// SPDX-License-Identifier: Apache-2.0
//! Lower a [`SubtileIR`] to an **IBM Spyre SuperDSC (SDSC) bundle** — the
//! TILE-LEVEL, work-divided IR that DeepTools' backend (`dxp_standalone
//! --bundle`) compiles into a device program. This is the PERFORMANCE path:
//! unlike the high-level sengraph emitter ([`lower_subtile_tape_to_sengraph`]),
//! which hands DeepTools an op-graph and lets `CompileGraph` derive the
//! per-core split automatically (on-card PROVEN to leave the 32 cores idle —
//! `SENCORES 1→14.4ms vs 32→9.9ms`, only 1.45×), the SuperDSC emitter OWNS the
//! 32-core work-division itself (exactly what torch-spyre's Inductor front-end
//! does). SubtileIR is already tile-level, so this is a tile→tile lowering.
//!
//! GROUND TRUTH: the DeepTools scheduler test fixtures
//! `/project_src/deeptools/dcg/dcg_fe/scheduler/test/sdsc_{add,bmm_autoBuffer,
//! …}.json` (on the pod) are real, valid SuperDSC; this module mirrors them
//! field-for-field. Ingest PROVEN on-card 2026-06-25: `L3DlOpsScheduler_standalone
//! -s <sdsc>.json` → "Success!" EXIT 0. See memory `superdsc-emitter-design.md`.
//!
//! NAMED ITERATION DIMS (NOT generic c0/c1): the SuperDSC iteration space uses
//! fixed slots `in_/out_/mb_/i_/j_/ki_/kj_/x_/x1_/y_/r_/c_/ij_/…` (-1 = unused).
//! For a matmul A[mb,in] · W[in,out] → O[mb,out]: `in_`=K (reduction), `out_`=N,
//! `mb_`=M, `x_`=tiling factor. Pointwise add over [mb,out] uses `i_`/`j_`/`ij_`.
//!
//! HARDWARE: 32 cores, 2 corelets/core; STICK = 128 B = 64 fp16 elems; per-core
//! span limit 256 MiB. fp16 everywhere → dataFormat_ "SEN169_FP16", wordLength 2,
//! stickSize_ 64.

// The SuperDSC struct fields mirror the DeepTools fixture JSON keys VERBATIM
// (trailing-underscore / camelCase, e.g. `coreFoldProp_`, `numWkSlicesPerDim_`),
// so the serde field name IS the JSON key. snake_case renaming would break the
// wire format — silence the lint for the whole module.
#![allow(non_snake_case)]

pub use crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off;
use crate::ir::bridge::tiled_op_sdsc_op::{
    assemble_attn, assemble_pointwise_broadcast_off_from_tile, assemble_pointwise_seeded_from_tile,
    assemble_reduce_seeded, assemble_rmsnorm,
};
use crate::ir::bridge::tiled_op_sdsc_op::{
    pointwise_broadcast_opspec_from_tile, pointwise_opspec_from_tile,
};
// Re-exports: external crates reference these matmul-family functions directly through this
// module's path (`scratchy-forward-compiler-macro`'s codegen: `superdsc::assemble_matmul`;
// `scratchy-sdsc`'s emit.rs: `mono::assemble_matmul_split`) — keep those paths resolving after
// the matmul family's real home moved to `ir::bridge::tiled_op_sdsc_op::matmul`.
//
// `assemble_matmul_seeded` / `assemble_matmul_off` moved from the private `use` above to this
// `pub use` (2026-07-28): tests/element_torchspyre_parity.rs calls both through this module path,
// so a private import made that ENTIRE test target fail to compile with E0603 — it has not built
// since 57dfac42 (07-26), silently killing 6 tests. One of them,
// `per_core_start_derives_from_spyre_layout`, is the ONLY thing that pins mb-split per-core start
// bytes against torch-spyre, i.e. exactly the machinery under suspicion for the composite-m
// garble. Nothing caught it because CI never builds this crate with `--features superdsc`.
pub use crate::ir::bridge::tiled_op_sdsc_op::{
    assemble_matmul, assemble_matmul_off, assemble_matmul_seeded, assemble_matmul_split,
};
/// ⭐ THE BAKED BUNDLE'S TYPES. This lowering's OUTPUT is a `bundle::BundleCode` — the value the
/// `#[forward]` macro puts in the binary and the runtime reads back.
pub use scratchy_spyre_bundle as bundle;
use scratchy_subtile::model_geometry::{with_config_attn_geometry, with_config_head_dim};
use scratchy_subtile::sdsc_abstract::{
    FlatTag, KernelTag, KindTag, RowBlockedTag, StickKind, StickLayout, Stk,
};
use scratchy_subtile::subtile_ir::{EwKind, RopeForm, SubOp, SubtileIR, SubtileNode};
use scratchy_subtile::superdsc_opspec::{
    Allocation, AnyTensorArg, ArgView, DataFormat, DeviceTileLayout, ItDim, MaxCores, OpFunc,
    OpInfo, OpSpec, Role, Scale, SdscFoldSet, StickExtent, TensorArg, WorkPlan,
    assert_df_stick_multiple,
};
use serde::Serialize;
use std::collections::BTreeMap;

/// This lowering's admission to the proven batch-inner bmm walk. The walk module charges a
/// site witness at [`SharedKernelBmmForm::batch_inner_proven`]'s door, and this module holds the
/// mint for the lowering's own wrapper sites (dense/fp8 projections, the RoPE rotation, the
/// krep/vrep identity replications) — none of which can carry request rows.
pub(crate) mod bmm_site {
    /// The tape lowering's [`NonAttnBmmSite`](crate::ir::bridge::tiled_op_sdsc_op::matmul::walk::NonAttnBmmSite).
    /// The private field seals construction into [`Self::witness`], whose visibility is this
    /// lowering module: the attention emitter cannot mint one, so it cannot state the proven walk.
    pub(crate) struct TapeLoweringSite(());

    impl TapeLoweringSite {
        /// Mintable only inside the tape lowering — the wrapper sites named on the module.
        pub(super) fn witness() -> Self {
            TapeLoweringSite(())
        }
    }
}

// Re-export the hardware constants from the typed core so the whole emitter
// shares ONE source of truth (the typed witnesses live in `superdsc_opspec`).
// (`DataFormat` itself is brought into scope by the main `use` block above.)
pub use scratchy_subtile::superdsc_opspec::{ACTIVE_CORELETS, MAX_CORES};
/// fp16 data format alias (the default). `Fp32` is used only by the fp32-SFP-merge split-K path;
/// `SenInt8` only by the packed-int8 (SENINT8) quant weight path.
pub use scratchy_subtile::superdsc_opspec::{Df, Fp8, Fp16, Fp32, SenInt8};

// Per-tensor dtype (`wordLength` / `stickSize_` / `dataFormat_`) is now the arg's typed [`Df`]
// (`ArgView::df`), NOT a `_fp8`/`_fp32`/`_senint8` name substring. The old `word_length_for` /
// `stick_elems_for` / `dataformat_for` string-sniffers are DELETED — see `Df` in superdsc_opspec.

pub const CORELETS_PER_CORE: u32 = 2;
pub const STICK_BYTES: u32 = 128;
pub const FP16_ELEMS_PER_STICK: u32 = 64; // 128 B / 2 B
pub const MAX_SPAN_BYTES: u64 = 256 * 1024 * 1024;

// ── EMIT-TIME CONSTANTS THAT USED TO BE ENVIRONMENT READS ───────────────────
//
// ⛔⛔⛔ THE EMITTER MUST NOT READ THE SHELL. `#[forward]` compiles
// `(config.json, math DSL)` into an artifact. An `env::var` in here makes it
// `(config.json, math DSL, whoever's shell ran cargo)` — two builds of identical
// source emit different bundles, and nothing downstream can be called a constant.
//
// ⛔ THE COST IS ON THE RECORD, in `spyre_load.rs`'s own comment: a bundle baked
// WITHOUT a K-split flag and served by a worker that read the flag staged blocks
// the bundle had no room for — every completion came back EMPTY, RC=0, no error,
// and it "cost four runs and one wrong conclusion". The WORKER was then fixed to
// ask the bundle instead of the environment. The EMITTER was not, so the same
// two-processes-agreeing-by-convention hazard survived one level up.
//
// ⭐ EACH VALUE BELOW IS WHAT THE UNSET VARIABLE PRODUCED. None of these is set
// by the canonical pod env or its build wrapper, so the unset case IS the
// production case and this is a pure constant-ification: the bundle fingerprints
// do not move. Changing one is now a source edit that recompiles, which is what
// makes it a constant rather than a configuration.
//
// 🛑 `SCRATCHY_SUPERDSC_GROUP_SIZE` IS DELIBERATELY NOT HERE. It is the one knob
// the canonical env DOES set (2048), and the two authorities disagree: this file
// says "2048 was too aggressive … 512 is the real ceiling, not 2048", while the
// canonical pod env says "G=2048 = whole-body = ~14.6 tok/s … pod DEBUG scripts use
// 512; DON'T — it tanks throughput". Picking either silently is a crash or a 2.2×
// throughput regression, so it stays a read until someone settles it on a card.

// ───────────────────────────────────────────────────────────────────────────
// Work-division (the core NEW logic — the whole point of this emitter).
// Ported from torch-spyre torch_spyre/_inductor/work_division.py. We use the
// SPYRE 256MB/core span + 64-fp16 stick model — NOT the KTIR emitter's H100
// pick_k/n_block formulas (wrong hardware).
// ───────────────────────────────────────────────────────────────────────────

/// Largest divisor of `size` that does not exceed `max_cores` (so the split is
/// always even). `core_split(384, 32) = 32`; `core_split(320, 32) = 16` (320 =
/// 2^6·5, largest divisor ≤32 is 16). Mirrors work_division.py `core_split`.
pub fn core_split(size: u32, max_cores: u32) -> u32 {
    let mut i = max_cores.min(size.max(1));
    while i >= 1 {
        if size.is_multiple_of(i) {
            return i;
        }
        i -= 1;
    }
    1
}

/// Stick count for an extent on the stick axis: ceil(size / 64) for fp16.
pub fn stick_count(elems: u32) -> u32 {
    elems.div_ceil(FP16_ELEMS_PER_STICK)
}

/// The 32-core split of one op's `[rows, cols]` output: `row_cores × stick_cores`, each core owning a
/// contiguous row-chunk × stick-chunk. CANONICAL here (subtile) so the emitter and the Kani-verifying tower
/// (`scratchy-sdsc`, which RE-EXPORTS this + proves it disjoint+covering) share ONE definition — no
/// duplicate brain. Output dims are split largest-first (rows vs stick-count), each by `core_split`; a
/// reduction dim is never split. `row_cores * stick_cores ≤ MAX_CORES`. (Was ported from `distribute_cores`,
/// which the pointwise/reduce ops already use — so making matmul use THIS makes the whole emit one split.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoreSplit {
    pub row_cores: u32,
    pub stick_cores: u32,
}

impl CoreSplit {
    /// Distribute up to `MAX_CORES` over `[rows, cols]`: split the larger output extent first (rows by
    /// row-count, cols by stick-count), each by `core_split`, with the cores left after the first.
    pub fn plan(rows: u32, cols: u32) -> CoreSplit {
        // ⭐ THE CAP IS `MAX_CORES`, AND IT IS THE ONLY CAP. The 1-core / N-core bisection
        // knobs that used to narrow it were emit-time env reads, so the Kani proofs had to
        // call `plan_capped(_, _, MAX_CORES)` explicitly to keep `cap` concrete — CBMC models
        // `var_os` as nondeterministic, which unbounded the `core_split` loop. With the cap a
        // constant, what the proofs check and what the emitter does are the same expression.
        let cap = MAX_CORES;
        Self::plan_capped(rows, cols, cap)
    }

    /// The PURE planning math (no env, no I/O): distribute up to `cap` cores over `[rows, cols]`,
    /// splitting the larger output extent first. Kani-provable — the only loop (`core_split`) is
    /// bounded by the concrete `cap`. `plan` = this with the env-derived cap.
    pub fn plan_capped(rows: u32, cols: u32, cap: u32) -> CoreSplit {
        let sc = stick_count(cols);
        let (first_is_rows, big, small) = if rows >= sc {
            (true, rows, sc)
        } else {
            (false, sc, rows)
        };
        let big_split = core_split(big, cap);
        let small_split = core_split(small, (cap / big_split).max(1));
        if first_is_rows {
            CoreSplit {
                row_cores: big_split,
                stick_cores: small_split,
            }
        } else {
            CoreSplit {
                row_cores: small_split,
                stick_cores: big_split,
            }
        }
    }

    pub fn ncores(&self) -> u32 {
        self.row_cores * self.stick_cores
    }

    /// The half-open region `(r0, r1, c0, c1)` (rows, cols) owned by core `(cr, cs)` of an `[R,C]` output.
    /// Row chunks are even (`core_split`); stick chunks even in STICK units, last stick's cols clamp to `C`.
    pub fn region(&self, cr: u32, cs: u32, rows: u32, cols: u32) -> (u32, u32, u32, u32) {
        let r_chunk = rows / self.row_cores;
        let r0 = cr * r_chunk;
        let r1 = if cr + 1 == self.row_cores {
            rows
        } else {
            r0 + r_chunk
        };
        let sc = stick_count(cols);
        let s_chunk = sc / self.stick_cores;
        let s0 = cs * s_chunk;
        let s1 = if cs + 1 == self.stick_cores {
            sc
        } else {
            s0 + s_chunk
        };
        let c0 = (s0 * FP16_ELEMS_PER_STICK).min(cols);
        let c1 = (s1 * FP16_ELEMS_PER_STICK).min(cols);
        (r0, r1, c0, c1)
    }
}

/// Pad a 64-aligned OUTPUT/KERNEL width `n64` so its STICK COUNT is core-splittable to the util floor
/// (≥8 cores). A PRIME/awkward stick count (the granite lm_head: `49216/64 = 769`, prime) has
/// `core_split(769, 32) = 1` ⇒ the gemm strands on ONE core (the util-floor #11 guard / the build-#3
/// failure). Rounding the stick count UP to a multiple of 8 makes `core_split(8k, 32) ≥ 8` (8 | 8k). A
/// no-op when the count is already splittable (all granite weights except the lm_head). SHARED by the
/// emitter (applied when the matmul's MACs cross the util-floor) and the worker's weight zero-pad, so the
/// staged buffer width matches the emitted device width by construction. `n64` MUST be a 64-multiple.
pub fn bump_sticks_to_splittable(n64: u32) -> u32 {
    let sticks = n64 / FP16_ELEMS_PER_STICK;
    let cur = core_split(sticks, MAX_CORES);
    if cur >= MAX_CORES {
        return n64; // already fills the machine
    }
    // ── FULL-OCCUPANCY PAD ───────────────────────────────────────────────────────────────────────
    // The floor of 8 below fills the UTIL FLOOR, not the machine. granite's lm_head is the case that
    // matters: vocab 49155 → 769 sticks, which is PRIME, so the ≥8 rule pads to 776 = 8·97 and
    // `core_split(776,32)` is 8. MEASURED in the emitted suffix: the lm_head matmul is 97 trips of
    // out=512, every one on 8 of 32 cores, with a wordLength=2 (fp16) kernel — i.e. the single
    // largest tensor in the model (2048·49664·2 B ≈ 203 MB/token, ~7.7% of all weight bytes) streams
    // through a quarter of the machine.
    //
    // Rounding to a multiple of MAX_CORES instead makes `core_split` exact: 800 = 32·25 ⇒ 32 cores.
    // A matmul this shape is BANDWIDTH-bound in its core count, so 4× the cores is worth 3.1% more
    // bytes (+24 sticks). The emitter's own util-floor guard calls under-occupancy "the 1.45×
    // sengraph regression this emitter exists to fix" — this raises that floor to the whole machine.
    //
    // The ≤1/8 padding test keeps this from firing where it would BACKFIRE: k_proj/v_proj are 8
    // sticks (n=512), and padding those to 32 would quadruple their weight to buy 4× cores — a wash
    // at best. Only a wide, awkward width (lm_head) passes.
    let full = sticks.next_multiple_of(MAX_CORES);
    if (full - sticks) * 8 <= sticks {
        return full * FP16_ELEMS_PER_STICK;
    }
    if cur >= 8 {
        n64
    } else {
        sticks.next_multiple_of(8) * FP16_ELEMS_PER_STICK
    }
}

/// A matmul OUTPUT/KERNEL **device stick width** — 64-aligned, and (for a FLOP-heavy gemm) core-splittable
/// to ≥8 cores — BY CONSTRUCTION. TYPE-SAFE LOCK-DOWN of the padding/alignment invariant: the SOLE
/// constructor [`DeviceWidth::for_output`] applies the padding rule ONCE, so the emitter's `n_dev`, the
/// kernel `RetileDescriptor`, and the worker's weight zero-pad CANNOT diverge — a raw `u32` is unusable
/// where a `DeviceWidth` is required (any mismatch is unconstructable, not caught at runtime by the shim).
/// The bump fires ONLY for `macs ≥ 2^20` so a small op (RoPE-rotate `[.,128]`) is NOT over-padded to 512;
/// every real WEIGHT matmul crosses that threshold for ANY `m ≥ 1`, so the worker (which stages at m=1)
/// and the emitter (prefill m=seq / decode m=1) always agree. Invariant Kani-proven (`kani_proofs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceWidth(u32);

impl DeviceWidth {
    /// The device width of a matmul output `[m, logical_n]` with contraction `k`: round `logical_n` up to a
    /// 64-stick, then (only if `m·n64·k ≥ 2^20`) bump the stick count to ≥8-splittable.
    pub fn for_output(m: u32, logical_n: u32, k: u32) -> DeviceWidth {
        let n64 = logical_n.next_multiple_of(FP16_ELEMS_PER_STICK);
        let macs = m as u64 * n64 as u64 * k as u64;
        DeviceWidth(if macs >= (1 << 20) {
            bump_sticks_to_splittable(n64)
        } else {
            n64
        })
    }

    /// The device width of a POINTWISE op's tensor `[·, logical_n]` (a ScalarMul/elementwise on the padded
    /// logits): round to a 64-stick, then bump a prime/awkward stick count to ≥8-splittable UNCONDITIONALLY.
    /// This EQUALS `for_output(m, logical_n, k)` for any producer matmul with `macs ≥ 2^20` (every real
    /// producer of a padded tensor — lm_head/o_proj/down_proj), so a pointwise CONSUMER addresses the exact
    /// same device layout its matmul PRODUCER emitted. (Kani: `devwidth_pointwise_matches_matmul`.)
    pub fn for_pointwise(logical_n: u32) -> DeviceWidth {
        DeviceWidth(bump_sticks_to_splittable(
            logical_n.next_multiple_of(FP16_ELEMS_PER_STICK),
        ))
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

/// Distribute up to `max_cores` across an op's iteration space (Pass 3 of the
/// planner: split OUTPUT dims first by decreasing size, then at most ONE
/// reduction dim with the cores that remain). Returns the split map (dim →
/// >1-split) — the `splitter` shape consumed by [`WorkPlan::divide`], which adds
/// > the (b)/(f) proof on top. Pass 2 (matmul cost model) refines this; this
/// > baseline already beats the sengraph auto-split (which leaves ~31 idle).
///
/// Example (the real `sdsc_bmm_autoBuffer.json`): dims mb=384(out), out=384(out),
/// in=64(reduction) → split `out` 2 and `mb` 16 → 32 cores
/// (`numWkSlicesPerDim_:{out:2, mb:16}`), reduction `in` left unsplit.
pub fn distribute_cores(dims: &[ItDim], max_cores: u32) -> BTreeMap<&'static str, u32> {
    let mut splits: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut remaining = max_cores;

    // ── ROW (mb/free) dim FIRST — unify with the matmul's mb-first partition (`matmul_split_map` ⭐). ──
    // A FLAT (row-major, `mq_flat_activation`) mq>1 activation lives in HBM row-major: the per-core start
    // is `r·cols` and the coordInfo walks the row's features CONTIGUOUSLY (stick-groups `eps` apart). If
    // instead the OUT dim is split and mb is left WHOLE (the old largest-first order), a core owns one
    // 64-stick across ALL rows but the walk still advances each row by `eps` — so row `r` of out-core `c`
    // lands at `(c+r)·eps` (ROW-MIXING: scalarmul wrote only `mb+out/eps−1` of `mb·out/eps` stick-groups,
    // measured `nz=3968=(c+r)·64`). Splitting `mb` first makes each core own WHOLE rows (≤1 row/core for
    // mq≤MAX_CORES), so the flat start alone places the row and there is no mb-stride to mis-walk — the
    // SAME reason the matmul (mb-split) is correct while the pointwise (out-split) was not. Byte-identical
    // at mb==1 (decode: `core_split(1,·)=1`, falls through to the out loop) and for single-stick ops (a
    // 1-stick `out` can't split, so mb was split anyway). NOTE: mq>MAX_CORES leaves >1 row/core → the
    // mb-walk row-mixing returns; that regime needs the per-tensor flat `stride_map` (see
    // `element_arrangement_design.md`), not just the split.
    if let Some(mb) = dims
        .iter()
        .find(|d| d.name == "mb" && !d.is_reduction && !d.is_stick)
    {
        let split = core_split(mb.size, remaining);
        if split > 1 {
            splits.insert(mb.name, split);
            remaining /= split;
        }
    }

    // Then the remaining OUTPUT dims, largest extent first (most parallelism, no PSUM merge).
    let mut outs: Vec<&ItDim> = dims
        .iter()
        .filter(|d| !d.is_reduction && d.name != "mb")
        .collect();
    outs.sort_by_key(|d| std::cmp::Reverse(d.size));
    for d in outs {
        if remaining <= 1 {
            break;
        }
        // Stick dims split by STICK COUNT at the dim's OPERAND format (fp16=64, fp8/int8=128) —
        // NOT a hardcoded 64. A fp8-output op (the qfp8ch activation quantizer) whose stick dim is
        // split at the fp16 64-basis hands a core a 64-wide slab = HALF a 128-fp8 stick (dxp
        // `L3DlOpsScheduler:1070`). Deriving from `df` keeps each core ≥1 whole DF-stick. fp16
        // dims are byte-identical (df.elems_per_stick()==64==the old stick_count basis).
        let basis = if d.is_stick {
            d.size.div_ceil(d.df.elems_per_stick())
        } else {
            d.size
        };
        let split = core_split(basis, remaining);
        if split > 1 {
            splits.insert(d.name, split);
            remaining /= split;
        }
    }

    // NO reduction-dim split. Splitting the reduced dim makes each core compute a
    // PARTIAL result that must accumulate into the SAME output address — but the
    // #50 per-core addressing gives those cores one shared output byte with no
    // confirmed PSUM-accumulate semantic for a standalone reduce (unlike matmul,
    // which rides its own `matmul_split_map`). That is exactly the collision the
    // #50 build guard rejects (observed: RMSNorm mean `[1,576]→[1,1]` over 32
    // cores → all write 0x4_0000_0000). So an op routed through `distribute_cores`
    // (reduce / pointwise / silu) splits ONLY its disjoint OUTPUT dims; a reduce
    // whose output is one stick (decode m=1) correctly lands on 1 core (it is tiny
    // — the 211 matmuls are where the 32-core win lives). Cross-core PSUM reduce is
    // a future optimization that needs the accumulate addressing, not this path.
    splits
}

/// Per-core slice index along each split dim, for `coreIdToWkSlice_`. Cores are
/// numbered row-major over the split dims in `splits` insertion order: for
/// `{out:2, mb:16}`, core c → out = c/16, mb = c%16 (matches the real fixture:
/// cores 0..15 → out0/mb0..15, cores 16..31 → out1/mb0..15). `splits` is the
/// [`WorkPlan`]'s validated split map (≤32 by type).
pub fn core_to_wk_slice(
    plan: &WorkPlan,
    all_dims: &[&'static str],
) -> BTreeMap<String, BTreeMap<&'static str, scratchy_subtile::superdsc_opspec::SliceIndex>> {
    use scratchy_subtile::superdsc_opspec::SliceIndex;
    let cores = plan.cores_used().get().max(1);
    // Split dims in a stable order (largest split first matches the fixture's
    // outer→inner core numbering: out (×2) outer, mb (×16) inner).
    let mut split_dims: Vec<(&&'static str, &u32)> = plan.splits().iter().collect();
    split_dims.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let mut out = BTreeMap::new();
    for c in 0..cores {
        let mut idx = c;
        let mut slice: BTreeMap<&'static str, SliceIndex> = BTreeMap::new();
        // Inner-most (smallest split) varies fastest. EVERY core's slice index comes from
        // this one mixed-radix decomposition — cores 2..n are enumerated by the same
        // digits as cores 0..1, never extrapolated from them.
        for (name, split) in split_dims.iter().rev() {
            slice.insert(**name, SliceIndex::of_core_decomposition(idx % **split));
            idx /= **split;
        }
        for d in all_dims {
            slice.entry(d).or_insert(SliceIndex::UNSPLIT);
        }
        out.insert(c.to_string(), slice);
    }
    out
}

/// How a matmul/bmm is split across cores per dim (batch/M/N/K split counts;
/// product = cores used). The cost-model Pass 2 (vs the generic Pass-3
/// `distribute_cores`): for the real `sdsc_bmm_autoBuffer.json` (M=384, N=384,
/// K=64, batch=16) it picks M×16, N×2 (=32), NOT M×32 — balancing the PT array
/// occupancy + HBM traffic per the analytic cost.
#[derive(Clone, Debug, PartialEq)]
pub struct MatmulSplit {
    pub b: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}
impl MatmulSplit {
    pub fn cores(&self) -> u32 {
        self.b * self.m * self.n * self.k
    }
}

fn divisors_upto(size: u32, cap: u32) -> Vec<u32> {
    (1..=cap.min(size.max(1)))
        .filter(|d| size.is_multiple_of(*d))
        .collect()
}

/// Pass-2 matmul cost-model core split (port of torch-spyre work_division.py
/// `cost_model_matmul_division`). Enumerates feasible `(b,m,n,k)` split combos
/// with `b*m*n*k <= max_cores`, prices each with the analytic runtime estimate,
/// and returns the lowest-cost (ties → most cores, then largest M-split). `m_t`
/// is the per-core M-tile target (8 rows = the PT array row group).
pub fn matmul_cost_split(b: u32, m: u32, n: u32, k: u32, max_cores: u32) -> MatmulSplit {
    let (bf, mf, nf, kf) = (b as f64, m as f64, n as f64, k as f64);
    let mut best = MatmulSplit {
        b: 1,
        m: 1,
        n: 1,
        k: 1,
    };
    let mut best_cost = f64::INFINITY;
    for &bs in &divisors_upto(b, max_cores) {
        for &ms in &divisors_upto(m, max_cores / bs) {
            for &ns in &divisors_upto(n, max_cores / (bs * ms)) {
                {
                    // NO K (reduction / INPUT-stick dim) split. A K-split makes each
                    // core compute a PARTIAL product that must PSUM-accumulate into
                    // the SAME output address — but the #50 per-core addressing gives
                    // those cores one shared output byte with no confirmed accumulate
                    // semantic on this path, which the #50 disjoint-output build guard
                    // rejects (observed: decode m=1 matmuls k-split → collide at the
                    // output segment base). So matmuls FILL cores via OUTPUT splits
                    // (b/m/n) ONLY. Decode (m=1) small-N matmuls land on fewer cores
                    // (still multi-core via the n-split); the big matmuls (lm_head /
                    // MLP, large N) still fill 32. K-split PSUM is a future opt that
                    // needs the accumulate addressing, not this path.
                    let ks = 1u32;
                    let cores = bs * ms * ns * ks;
                    if cores == 0 || cores > max_cores {
                        continue;
                    }
                    // STICK constraint (proven on-card 2026-06-25): N is the OUTPUT/
                    // KERNEL stick dim and K the INPUT stick dim — their PER-CORE
                    // extent must stay a whole multiple of the 64-fp16 stick, else
                    // the dxp scheduler rejects the tile ("valid lower and upper
                    // bound", L3DlOpsScheduler.cpp:1040: per-core < the 64 stick
                    // lBound). M (mb) is not a stick dim, so it may split freely.
                    if !(n / ns).is_multiple_of(FP16_ELEMS_PER_STICK)
                        || !(k / ks).is_multiple_of(FP16_ELEMS_PER_STICK)
                    {
                        continue;
                    }
                    let (mm, nn) = ((m / ms) as f64, (n / ns) as f64);
                    // PT efficiency falls off when the per-core M tile < 8 rows.
                    let pt_eff = (1.0_f64)
                        .min((1.0_f64).max(mm / 8.0) / 8.0)
                        .sqrt()
                        .max(1e-3);
                    let compute = (bf * mf * nf * kf / cores as f64) / (1536.0 * pt_eff);
                    let hbm = (bf * mf * kf + bf * kf * nf + bf * mf * nf) * 2.0 / (204.8 * 1000.0)
                        * (1.0_f64).max(mm.max(nn) / 8.0);
                    let psum = (0.0_f64).max(ks as f64 - 1.0) * bf * mf * nf * 1.4e-4;
                    let m_target = 8.0;
                    let tie = ((mm / m_target).log2()).abs() * 50.0;
                    let cost = (compute + hbm + psum + tie) * (bs as f64).powf(1.4);
                    // PRIMARY objective: FILL the cores — the whole point of this
                    // emitter is owning the 32-way split (the sengraph 1.45× was
                    // under-utilization). So MORE cores always wins; the analytic
                    // cost only breaks ties between equal-core splits, then prefer
                    // the larger M-tile. (Earlier the cost was primary, and the
                    // pt_eff/hbm terms — which reward bigger per-core tiles —
                    // mis-picked a 24-core split over the 32-core fill.)
                    let bc = best.cores();
                    let better = cores > bc
                        || (cores == bc && cost < best_cost - 1e-9)
                        || (cores == bc && (cost - best_cost).abs() <= 1e-9 && ms > best.m);
                    if better {
                        best_cost = cost;
                        best = MatmulSplit {
                            b: bs,
                            m: ms,
                            n: ns,
                            k: ks,
                        };
                    }
                }
            }
        }
    }
    best
}

/// All positive divisors of `n` ascending (`divisors(12) = [1,2,3,4,6,12]`). torch-spyre
/// `sympy.divisors`. `n<=0 → [1]`.
fn divisors(n: u32) -> Vec<u32> {
    let n = n.max(1);
    (1..=n).filter(|d| n.is_multiple_of(*d)).collect()
}

/// FAITHFUL port of torch-spyre `work_division.py::_matmul_split_cost` — the analytic 9-term
/// hardware cost (µs) of running `[B,M,K]@[B,K,N]` under the split `(b,m,n,k)`. Each axis is
/// `(size, split)`; sizes are ELEMENTS (M,N,K) / count (B). Lower is better; `inf` if infeasible.
/// Constants are torch-spyre's verbatim (AIU HW limits + measured kernel-time coefficients) so the
/// split scratchy picks is the one torch-spyre picks — NOT a rewritten "fill the cores" heuristic.
#[allow(clippy::too_many_arguments)]
fn matmul_split_cost(
    b_axis: (u32, u32),
    m_axis: (u32, u32),
    n_axis: (u32, u32),
    k_axis: (u32, u32),
    max_cores: u32,
    shared_weight: bool,
) -> f64 {
    const PT_ROWS: f64 = 8.0;
    const TARGET_PT_PASSES: f64 = 5.0;
    const TARGET_M_TIE_PASSES: u32 = 4;
    const PT_EFFICIENCY_EXPONENT: f64 = 0.25;
    const M_MIN: f64 = 4.0; // PT_ROWS // 2
    const PEAK_MACS_US_CORE: f64 = 98.304e12 / 2.0 / 32.0 / 1e6; // = 1_536_000.0
    const HBM_BW_GBS: f64 = 204.8;
    const DTYPE_BYTES: f64 = 2.0;
    const PSUM_PER_CORE_ELEM_US: f64 = 1.0e-3;
    const BMM_PSUM_PER_CORE_ELEM_US: f64 = 1.0e-4;
    const COHORT_LIMIT: f64 = 8.0;
    const COHORT_PENALTY_EXPONENT: f64 = 0.75;
    const M_LANE_UNDERUSE_PENALTY_US: f64 = 10.0;
    const M_TILE_UNDERFILL_TARGET: f64 = 16.0;
    const M_TILE_UNDERFILL_PENALTY_US: f64 = 30.0;
    const TARGET_N_TILE_ELEMS: f64 = 512.0;
    const WIDE_N_TILE_PENALTY_US: f64 = 25.0;
    const CORE_UNDERUSE_PENALTY_US: f64 = 150.0;
    const BMM_BATCH_SPLIT_PENALTY_US: f64 = 10.0;
    const LARGE_M_TILE_SHAPE_PENALTY_US: f64 = 20.0;
    const SHARED_DOWN_N_SPLIT_PENALTY_US: f64 = 10.0;
    const SHARED_NARROW_OUTPUT_REF: f64 = TARGET_N_TILE_ELEMS * COHORT_LIMIT; // 4096
    const SHARED_N_TILE_TARGET: f64 = TARGET_N_TILE_ELEMS / 4.0; // 128

    let ((big_b, b), (big_m, m), (big_n, n), (big_k, k)) = (b_axis, m_axis, n_axis, k_axis);
    let cores_used = b * m * n * k;
    if cores_used == 0 || cores_used > max_cores {
        return f64::INFINITY;
    }
    let (bf, mf, nf, kf) = (big_b as f64, big_m as f64, big_n as f64, big_k as f64);
    let (bs, ms, ns, ks) = (b as f64, m as f64, n as f64, k as f64);

    // Compute: per-core MACs over peak, derated when the per-core M tile is too short to fill the PT.
    let m_t = big_m.checked_div(m).map_or(1.0, |t| t as f64);
    let pt_passes = (1.0f64).max(m_t / PT_ROWS);
    let pt_eff = (1.0f64).min((pt_passes / TARGET_PT_PASSES).powf(PT_EFFICIENCY_EXPONENT));
    let compute_us = (bf * mf * nf * kf / cores_used as f64) / (PEAK_MACS_US_CORE * pt_eff);

    // HBM: operands broadcast to the cohort splitting the orthogonal dim; past _COHORT_LIMIT contends.
    let weight_batches = if shared_weight { 1.0 } else { bf };
    let bytes_total = (bf * mf * kf + weight_batches * kf * nf + bf * mf * nf) * DTYPE_BYTES;
    let fanout_split = if shared_weight { ms.max(ns) } else { ns };
    let cohort_penalty = (1.0f64).max((fanout_split / COHORT_LIMIT).powf(COHORT_PENALTY_EXPONENT));
    let hbm_us = bytes_total / (HBM_BW_GBS * 1000.0) * cohort_penalty;

    // PSUM: a K-split costs (k-1) partial-sum hops over each core's output tile.
    let psum_coeff = if shared_weight {
        PSUM_PER_CORE_ELEM_US
    } else {
        BMM_PSUM_PER_CORE_ELEM_US
    };
    let output_elems_per_core = (bf * mf * nf) / (1.0f64).max((b * m * n) as f64);
    let psum_us = (0.0f64).max(ks - 1.0) * output_elems_per_core * psum_coeff;

    // Tie-break: expose enough M lanes to feed the stationary weight, and avoid under-filled M tiles.
    let target_m = M_MIN.max(
        ((max_cores / 2) as f64)
            .min((1.0f64).max((big_m / (TARGET_M_TIE_PASSES * 8)).max(1) as f64)),
    );
    let m_lane_underuse_us =
        (0.0f64).max((target_m / (1.0f64).max(ms)).log2()) * M_LANE_UNDERUSE_PENALTY_US;
    let m_tile_underfill_us = (0.0f64).max((M_TILE_UNDERFILL_TARGET / (1.0f64).max(m_t)).log2())
        * M_TILE_UNDERFILL_PENALTY_US;

    // Very wide per-core output tiles lose schedule efficiency.
    let n_t = if n != 0 { nf / ns } else { nf };
    let wide_n_us =
        (0.0f64).max(((1.0f64).max(n_t) / TARGET_N_TILE_ELEMS).log2()) * WIDE_N_TILE_PENALTY_US;

    // Tile-shape preferences (ratios, not op names): true-BMM value split, shared-narrow, shared-down.
    let filled_m_tile_factor = if m_t >= M_TILE_UNDERFILL_TARGET {
        1.0
    } else {
        0.0
    };
    let true_bmm_value_split_us = if shared_weight || n <= 1 {
        0.0
    } else {
        filled_m_tile_factor
            * (0.0f64).max(((1.0f64).max(kf) / (1.0f64).max(nf)).log2())
            * ns.log2()
            * LARGE_M_TILE_SHAPE_PENALTY_US
    };
    let shared_narrow_tile_us = if !shared_weight {
        0.0
    } else {
        filled_m_tile_factor
            * (0.0f64).max((SHARED_NARROW_OUTPUT_REF / (1.0f64).max(nf)).log2())
            * (0.0f64).max(((1.0f64).max(n_t) / SHARED_N_TILE_TARGET).log2())
            * (LARGE_M_TILE_SHAPE_PENALTY_US / 4.0)
    };
    let shared_down_n_split_us = if !shared_weight || n <= 1 {
        0.0
    } else {
        (0.0f64).max(((1.0f64).max(kf) / (1.0f64).max(nf)).log2())
            * ns.log2()
            * SHARED_DOWN_N_SPLIT_PENALTY_US
    };
    let large_m_tile_shape_us =
        true_bmm_value_split_us + shared_narrow_tile_us + shared_down_n_split_us;

    // Prefer using the full core budget (soft).
    let core_underuse_us =
        (0.0f64).max((max_cores as f64 / cores_used as f64).log2()) * CORE_UNDERUSE_PENALTY_US;

    // True BMMs: small additive batch-split overhead.
    let batch_split_us = if shared_weight {
        0.0
    } else {
        (1.0f64).max(bs).log2() * BMM_BATCH_SPLIT_PENALTY_US
    };

    compute_us
        + hbm_us
        + psum_us
        + m_lane_underuse_us
        + m_tile_underfill_us
        + wide_n_us
        + large_m_tile_shape_us
        + core_underuse_us
        + batch_split_us
}

/// FAITHFUL port of torch-spyre `_cost_model_matmul_planner`: enumerate every `(b,m,n,k)` where `m`
/// divides M (elements), `n` divides the N stick count, `k` divides the K stick count, `b` divides the
/// batch, with `b·m·n·k ≤ max_cores`, and pick the split with the LOWEST [`matmul_split_cost`]. `n`/`k`
/// range over STICK counts (torch-spyre `n_divs = divisors(n_sticks)`), so every per-core N/K slice is a
/// whole stick by construction. K-split is DISABLED (`k=1`): scratchy has no K-time PSUM-accumulate
/// addressing (#50), and for the granite projections torch-spyre's own cost model picks `k=1` anyway
/// (the PSUM penalty dominates), so this is faithful in practice, not a shortcut.
pub(crate) fn matmul_split_plan(
    m: u32,
    n_elems: u32,
    k_elems: u32,
    batch: u32,
    stick: u32,
    max_cores: u32,
    shared_weight: bool,
) -> MatmulSplit {
    let n_sticks = n_elems.div_ceil(stick).max(1);
    let k_sticks = k_elems.div_ceil(stick).max(1);
    // K-SPLIT RE-ENABLED (2026-07-28) now that its blocker is fixed. It was disabled because
    // splitting the reduction dim emitted a DESCRIPTOR THAT CONTRADICTED ITS OWN ADDRESSING for fp8
    // operands: `gen_fp8_kernel_in_fold` / `gen_fp8_input_in_fold` hardcoded
    // `{"factor_": 1, "label_": "core_fold"}` for the `in` dim (written under the older invariant
    // that the reduction is never split), so the coordInfo reconstructed only K/nsplits while
    // `numWkSlicesPerDim_["in"]` and `per_core_addr` encoded a real nsplits-way split. Both
    // generators now take `nsplits` and emit it as core_fold, exactly as their fp16 sibling
    // `gen_coord_info_value` always has, so the descriptor and the addressing agree again.
    //
    // This matters for prefill THROUGHPUT, not just tidiness: measured on hardware, prefill is
    // compute-bound (75.4 GMAC in 82.9 ms = 909 GMAC/s) while decode is weight-bandwidth-bound
    // (2.43 GB / 29 ms = 84 GB/s). With k unsplit, down_proj runs its whole k=8192 reduction on one
    // core; the cost model picks (n=8, k=4) for every major granite projection when allowed to.
    // Only mb>1 is affected -- decode's m=1 takes the out-split-only branch of `matmul_split_map`,
    // where k is 1 regardless, so decode's emission is unchanged either way.
    let k_divs = divisors(k_sticks);
    // NOTE on a DISPROVEN theory, recorded so it is not re-derived: batched prefill is coherent at
    // prefill_m=31 and 17 (both PRIME) and garbage at 16 (composite), which looks like "the cost
    // model only mb-splits at composite m, so mb-split is broken". That premise is FALSE. mb-split
    // is NOT gated by primality or by `m_divs`: matmul_split_map's single-stick branch calls
    // core_split(mb, max_cores) directly, so at the COHERENT m=31 the per-head attention score/value
    // matmuls already split mb 31 ways, distribute_cores splits mb 31 ways for every [31,2048]
    // pointwise op, and decode's attn_krep/vrep split mb 32 ways. mb-splitting is hardware-proven at
    // both extremes (nsplits=m with 1 row/core, and nsplits=1 with m rows/core), and the emitted
    // coordInfo + per-core start bytes are byte-identical to IBM's own dxp-validated 8-way mb-split
    // fixture (deeptools ddc/test/l0_tethering/.../sdsc_alxs_input_MatMul_49.json). Do NOT "fix"
    // gen_coord_info_value's non-stick alphas or per_core_addr -- they are golden.
    // What ACTUALLY changes only at m=16 is three inseparable things: gate/up/down pick
    // {mb:2,out:8,in:2}; those same ops then overflow USABLE_LX_BYTES and take time_tile=2 (the
    // first fp8 time-tile this backend has ever emitted); and the token-stream pointwise ops gain a
    // second split dim, {mb:16,out:2}, because core_split(16,32)=16 leaves remaining=2 (at a prime m
    // remaining is 1 and the loop breaks). Any of those three -- not mb-split per se -- is the
    // suspect.
    let (m_divs, n_divs, b_divs) = (
        divisors(m.max(1)),
        divisors(n_sticks),
        divisors(batch.max(1)),
    );
    let mut best = MatmulSplit {
        b: 1,
        m: 1,
        n: 1,
        k: 1,
    };
    let mut best_cost = f64::INFINITY;
    for &bb in &b_divs {
        for &mm in &m_divs {
            for &nn in &n_divs {
                for &kk in &k_divs {
                    if bb * mm * nn * kk > max_cores {
                        continue;
                    }
                    let c = matmul_split_cost(
                        (batch.max(1), bb),
                        (m, mm),
                        (n_elems, nn),
                        (k_elems, kk),
                        max_cores,
                        shared_weight,
                    );
                    if c < best_cost {
                        best_cost = c;
                        best = MatmulSplit {
                            b: bb,
                            m: mm,
                            n: nn,
                            k: kk,
                        };
                    }
                }
            }
        }
    }
    best
}

/// The `N_` iteration space for a matmul/bmm: mb=M, out=N, in=K (reduction),
/// x=batch, y=1 (decoded from the fixture's primaryDsInfo_ layoutDimOrder_).
pub fn matmul_iter_space(m: u32, n: u32, k: u32, batch: u32) -> IterSpace {
    let mut it = IterSpace::empty();
    it.mb_ = m as i64;
    it.out_ = n as i64;
    it.in_ = k as i64;
    it.y_ = 1;
    // The output-tile counts MUST be 1 (single tile), NOT the -1 "unused"
    // sentinel. `bmm.ddl` reads i_/j_/ij_ as its tile-loop bounds; -1 makes the
    // loop walk garbage → output orthogonal to A@W and invariant to addresses
    // (the exact failure signature). Match the dxp-VALIDATED MatMul_49 N_
    // (fp16_32core_nonmx): i_=1, j_=1, ij_=1, with the batch slot x_ left -1 for
    // a plain (non-batched) matmul. Only a true bmm (batch>1) populates x_.
    it.i_ = 1;
    it.j_ = 1;
    it.ij_ = 1;
    if batch > 1 {
        it.x_ = batch as i64;
    }
    it
}

// ───────────────────────────────────────────────────────────────────────────
// SuperDSC JSON wire structs — the FRONTEND-MINIMAL field set mirroring
// torch-spyre `compute_ops.py generate_sdsc`. These are the ONLY types that
// derive `Serialize`; the typed `OpSpec`/`TensorArg`/`WorkPlan` core (the
// witnesses) lowers into them via `emit_sdsc`. The over-CONSTRAINING fields
// (register-file memOrg, hand-set fold factors, hand-set exUnit) are unset
// BY CONSTRUCTION upstream — they cannot reach these structs. Trailing
// underscores on keys are MANDATORY (DeepTools reads them verbatim).
// ───────────────────────────────────────────────────────────────────────────

/// `{"factor_": N, "label_": "core"}` — a fold (core/corelet/time) descriptor.
#[derive(Serialize, Clone, Debug)]
pub struct FoldProp {
    pub factor_: i64,
    pub label_: &'static str,
}

/// Skip an `IterSpace` dim slot when it is the -1 "unused" sentinel — torch-spyre's
/// REAL frontend `N_` emits ONLY the dims an op uses (e.g. `{name_:"n", out_:64}`),
/// NOT the full 21-slot table with -1 fillers. Verified on compiled `torch.rsqrt`/
/// `add`/`mul`. The verbose -1 fillers + empty scheduler maps below were the LAST
/// difference between scratchy's rsqrt SDSC and torch-spyre's (the compiled programs
/// were 99.7% identical — only 6 constant bytes differed); dxp's SFP path reads the
/// empty `peSfpSplit_`/`coreletSplit_` scheduler maps that torch-spyre OMITS.
fn iter_slot_unused(v: &i64) -> bool {
    *v == -1
}
fn map_is_empty_i64(m: &BTreeMap<String, i64>) -> bool {
    m.is_empty()
}
fn map_is_empty_veci64(m: &BTreeMap<String, Vec<i64>>) -> bool {
    m.is_empty()
}

/// The full SuperDSC named iteration space. -1 = unused slot, OMITTED on serialize
/// (torch-spyre emits only used dims). The empty scheduler maps are also omitted —
/// torch-spyre's frontend never emits them (they are scheduler outputs).
#[derive(Serialize, Clone, Debug)]
pub struct IterSpace {
    pub name_: &'static str,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub in_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub out_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub mb_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub i_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub j_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub ki_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub kj_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub x_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub x1_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub y_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub r_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub c_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub ij_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub rc_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub kij_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub sij_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub zij_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub si_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub sj_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub zi_: i64,
    #[serde(skip_serializing_if = "iter_slot_unused")]
    pub zj_: i64,
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub symbolicDimInfo_: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub maxSymbolicVolume_: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "map_is_empty_veci64")]
    pub coreletSplit_: BTreeMap<String, Vec<i64>>, // {dim: [per_corelet0, per_corelet1]}
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub rowSplit_: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub peSfpSplit_: BTreeMap<String, i64>,
    #[serde(skip_serializing_if = "map_is_empty_i64")]
    pub paddingSizes_: BTreeMap<String, i64>,
}

impl IterSpace {
    /// All slots -1 (unused) except `name_`. Set the dims an op uses afterward.
    pub fn empty() -> Self {
        IterSpace {
            name_: "n",
            in_: -1,
            out_: -1,
            mb_: -1,
            i_: -1,
            j_: -1,
            ki_: -1,
            kj_: -1,
            x_: -1,
            x1_: -1,
            y_: -1,
            r_: -1,
            c_: -1,
            ij_: -1,
            rc_: -1,
            kij_: -1,
            sij_: -1,
            zij_: -1,
            si_: -1,
            sj_: -1,
            zi_: -1,
            zj_: -1,
            symbolicDimInfo_: BTreeMap::new(),
            maxSymbolicVolume_: BTreeMap::new(),
            coreletSplit_: BTreeMap::new(),
            rowSplit_: BTreeMap::new(),
            peSfpSplit_: BTreeMap::new(),
            paddingSizes_: BTreeMap::new(),
        }
    }
}

/// `exUnit`: "pt" for matmul/bmm, "sfp" for everything else. DEPRECATED string
/// helper — the typed [`OpFunc::ex_unit`] (a sealed `ExUnit` newtype with no
/// public constructor) is now the SOLE producer (witness #6). Kept only so any
/// out-of-module caller still resolves; new code goes through `OpFunc`.
pub fn ex_unit(op_func: &str) -> &'static str {
    match op_func {
        "matmul" | "batchmatmul" => "pt",
        _ => "sfp",
    }
}

/// Affine-fold leaf for `dim_prop_func` entries. The fixture uses `{"Affine":
/// {alpha_,beta_}}` for linear per-index maps, `{"Const":{}}` for a fixed value,
/// and `{"Map":{}}` for an explicit `data_` lookup table (per-core addresses).
#[derive(Serialize, Clone, Debug)]
pub enum FoldFunc {
    Affine { alpha_: i64, beta_: i64 },
    Const {},
    Map {},
}

/// `memOrg_` component presence for one tensor. FRONTEND-MINIMAL: torch-spyre
/// `compute_ops.py` emits ONLY `{"isPresent": 1}` per component — the size/pad/
/// offset/allocateNode fields are scheduler outputs (over-specifying them was a
/// DtException 1535 source). Just the presence flag here.
#[derive(Serialize, Clone, Debug)]
pub struct MemPresence {
    pub isPresent: u8,
}

/// `memOrg_` = the components a tensor is resident in. SEALED to EXACTLY hbm/lx
/// (witness: NO register-file key like `pelrf` can be added — the frontend never
/// emits one; the L3 scheduler assigns l0/ptxrf/pelrf). Serializes to the minimal
/// `{"hbm":{"isPresent":1},"lx":{"isPresent":1}}` (or hbm-only / lx-only).
#[derive(Clone, Debug)]
pub struct MemOrg {
    hbm: Option<MemPresence>,
    lx: Option<MemPresence>,
}
impl MemOrg {
    /// Default residency: present in BOTH hbm and lx (the matmul/pointwise case).
    pub fn hbm_lx() -> Self {
        MemOrg {
            hbm: Some(MemPresence { isPresent: 1 }),
            lx: Some(MemPresence { isPresent: 1 }),
        }
    }
    /// HBM-only (index tensors must reside in HBM — no LX indirect addressing).
    pub fn hbm_only() -> Self {
        MemOrg {
            hbm: Some(MemPresence { isPresent: 1 }),
            lx: None,
        }
    }
    /// LX-only (an op-local scratchpad dataspace).
    pub fn lx_only() -> Self {
        MemOrg {
            hbm: None,
            lx: Some(MemPresence { isPresent: 1 }),
        }
    }
}
// Custom Serialize so absent components are simply omitted (never a `null`),
// matching torch-spyre's conditional dict; and so NO register-file key exists.
impl Serialize for MemOrg {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let n = self.hbm.is_some() as usize + self.lx.is_some() as usize;
        let mut m = ser.serialize_map(Some(n))?;
        if let Some(p) = &self.hbm {
            m.serialize_entry("hbm", p)?;
        }
        if let Some(p) = &self.lx {
            m.serialize_entry("lx", p)?;
        }
        m.end()
    }
}

/// One tensor descriptor in `labeledDs_`. FRONTEND-MINIMAL: torch-spyre emits
/// only `{ldsIdx_, dsName_, dsType_, scale_, wordLength, dataFormat_, memOrg_}`.
/// The dropped fields (density_/segment_/isStatic_/level/hbm*Size_/lx*Size_/…)
/// are scheduler-filled — emitting them mis-sized the LX chunk (1535). `scale_`:
/// 1 = active, -1 = reduction (non-stick), -2 = stick-aligned reduction (sealed
/// to the `Scale` enum upstream — never a free integer).
#[derive(Serialize, Clone, Debug)]
pub struct LabeledDs {
    pub ldsIdx_: u32,
    pub dsName_: String,
    pub dsType_: &'static str, // "INPUT" | "KERNEL" | "OUTPUT"
    pub scale_: Vec<i64>,
    pub wordLength: u32,           // 2 for fp16
    pub dataFormat_: &'static str, // "SEN169_FP16"
    pub memOrg_: MemOrg,
}

/// `computeOp_` entry. `exUnit` is a sealed `ExUnit` value from `OpFunc::ex_unit`
/// only (witness #6) — never hand-set on a matmul.
#[derive(Serialize, Clone, Debug)]
pub struct ComputeOp {
    pub exUnit: &'static str, // "pt" | "sfp"
    pub opFuncName: String,
    pub attributes_: ComputeAttrs,
    pub location: &'static str, // "Inner"
    // torch-spyre's REAL frontend emits ONLY the 6 core fields above + the two
    // labeledDs lists — VERIFIED on compiled `torch.rsqrt`/`F.rms_norm`/`add`/`mul`
    // (every op = `exUnit/opFuncName/attributes_/location/inputLabeledDs/outputLabeledDs`,
    // NOTHING else). A prior me ADDED the loop-placement flags below citing the dxp
    // `sdsc_silu.json` *fixture*, but the real frontend OMITS them and computes rsqrt
    // CORRECTLY on-card — and an explicit `auxLoopName: ""` plausibly tells dxp "no aux
    // loop" → suppresses the SFP Newton-Raphson refine (rsqrt→1/x). So every field below
    // is Option/skip-if-empty: the default emission is byte-equal to torch-spyre (omitted),
    // and a populated interim/indirect list (paged ops) still serializes when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxLoopName: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isAtMainLoop: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isAtTop: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub coreExclude: Vec<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub coreClExclude: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opConsts: Option<serde_json::Value>,
    pub inputLabeledDs: Vec<String>, // ["Tensor0-idx0", …]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub interimLabeledDs: Vec<String>,
    pub outputLabeledDs: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub indirectAccessIndexLabeledDs: Vec<String>,
}
#[derive(Serialize, Clone, Debug)]
pub struct ComputeAttrs {
    pub dataFormat_: &'static str,
    pub fidelity_: &'static str, // "regular"
}

/// `primaryDsInfo_` entry (per role INPUT/KERNEL/OUTPUT): the tile layout.
/// Matches torch-spyre: `layoutDimOrder_`, `stickDimOrder_`, `stickSize_` only
/// (no `stickRepl_` — torch-spyre does not emit it).
#[derive(Serialize, Clone, Debug)]
pub struct LayoutInfo {
    pub layoutDimOrder_: Vec<&'static str>,
    pub stickDimOrder_: Vec<&'static str>,
    pub stickSize_: Vec<u32>, // [64] for fp16
}

/// `scheduleTree_` allocate node: a per-tensor HBM (or LX) buffer. FRONTEND-
/// MINIMAL, matching torch-spyre's `generate_sdsc` allocate node: `nodeType_`,
/// `name_`, `prev_`, `ldsIdx_`, `component_`, `layoutDimOrder_`, `maxDimSizes_`,
/// the indirect-access fields, `startAddressCoreCorelet_`, `coordinates_`, and
/// `backGapCore_` ONLY when a back-gap is present. The mechanical defaults the
/// bmm SCHEDULER fixture carried (numBuffers_/constIdx_/nonUnifiedAllocInHBM_/…)
/// are dropped — they are scheduler outputs. `isStartAddrSymbolic_` is emitted
/// ONLY for symbolic HBM addresses; `relatedIndirectAccessAlloc_`/
/// `indexTensorType_` ONLY for indirect access (`skip_serializing_if` on the
/// `Option`s keeps byte-faithfulness for both presence and absence).
#[derive(Serialize, Clone, Debug)]
pub struct AllocNode {
    pub nodeType_: &'static str, // "allocate"
    pub name_: String,
    pub prev_: String,
    pub ldsIdx_: u32,
    pub component_: &'static str, // "hbm" | "lx"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isStartAddrSymbolic_: Option<u8>,
    pub layoutDimOrder_: Vec<&'static str>,
    // Opaque: the ONLY way to build one is `StickLayout::device_walk` (from the same layout as the start),
    // so a walk that disagrees with the start — or has the wrong rank — is unconstructable. Serializes to
    // the `[-1; rank]` / pinned-extents array dxp reads.
    pub maxDimSizes_: scratchy_subtile::sdsc_abstract::DeviceWalk,
    pub indirectAllocType_: &'static str, // "no_indirection"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relatedIndirectAccessAlloc_: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexTensorType_: Option<&'static str>,
    pub startAddressCoreCorelet_: AddrFold,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backGapCore_: Option<serde_json::Value>,
    pub coordinates_: serde_json::Value, // {coordInfo:{…}, coreIdToWkSlice_:{}}
}
/// The `startAddressCoreCorelet_` fold: `dim_prop_func` = `[Map, Const, Const]`
/// over `[core, corelet, time]`, with the per-core HBM byte address in `data_`
/// keyed `"[c, 0, 0]"` (SPACES, per torch-spyre `_start_addr_data`). The
/// `[core, corelet]` factors are COPIED from the op's [`SdscFoldSet`] via
/// [`AddrFold::new`] (witness (c)): there is no public field setter, so a factor
/// that diverges from `coreFoldProp_`/`coreletFoldProp_` is unrepresentable.
#[derive(Serialize, Clone, Debug)]
pub struct AddrFold {
    pub dim_prop_func: Vec<FoldFunc>,
    pub dim_prop_attr: Vec<FoldProp>,
    /// "[c, 0, 0]" → HBM byte address (a string, per the torch-spyre frontend).
    pub data_: BTreeMap<String, String>,
}
impl AddrFold {
    /// Build the address fold, copying the core/corelet/time factors STRAIGHT from
    /// `folds` (witness (c)). `per_core_addr[c]` is the byte address for core `c`.
    /// CONCRETE-address path (`isStartAddrSymbolic_` absent) — the time=1 case.
    pub fn new(per_core_addr: &[u64], folds: &SdscFoldSet) -> AddrFold {
        AddrFold {
            // torch-spyre emits [Map, Const, Const] over [core, corelet, time].
            dim_prop_func: vec![FoldFunc::Map {}, FoldFunc::Const {}, FoldFunc::Const {}],
            dim_prop_attr: vec![
                FoldProp {
                    factor_: folds.core_fold() as i64,
                    label_: "core",
                },
                FoldProp {
                    factor_: folds.corelet_fold() as i64,
                    label_: "corelet",
                },
                FoldProp {
                    factor_: folds.time_fold() as i64,
                    label_: "time",
                },
            ],
            data_: per_core_addr
                .iter()
                .enumerate()
                .map(|(c, a)| (format!("[{c}, 0, 0]"), a.to_string()))
                .collect(),
        }
    }

    /// Build the SYMBOLIC-address fold for a TIME-TILED HBM tensor: the per-core
    /// `data_` value is a NEGATIVE symbol-id string (e.g. `"-1"`) instead of a
    /// concrete byte address, because the per-iteration HBM advance is computed by
    /// `affine.apply` in `bundle.mlir` (torch-spyre `compute_ops.py:491`,
    /// `local_symbols[addr]` under `use_symbols=True`). The fold factors are still
    /// copied from `folds` (witness (c)); only the `data_` values differ. Pairs
    /// with `isStartAddrSymbolic_: Some(1)` on the AllocNode (the **SymbolicWhenTiled**
    /// witness). `per_core_sym_ids[c]` is the negative id for core `c`.
    pub fn symbolic(per_core_sym_ids: &[i64], folds: &SdscFoldSet) -> AddrFold {
        AddrFold {
            dim_prop_func: vec![FoldFunc::Map {}, FoldFunc::Const {}, FoldFunc::Const {}],
            dim_prop_attr: vec![
                FoldProp {
                    factor_: folds.core_fold() as i64,
                    label_: "core",
                },
                FoldProp {
                    factor_: folds.corelet_fold() as i64,
                    label_: "corelet",
                },
                FoldProp {
                    factor_: folds.time_fold() as i64,
                    label_: "time",
                },
            ],
            data_: per_core_sym_ids
                .iter()
                .enumerate()
                .map(|(c, id)| (format!("[{c}, 0, 0]"), id.to_string()))
                .collect(),
        }
    }
}

/// `dataStageParam_["0"]` = per-core steady-state (`ss_`) + epilogue (`el_`)
/// dim sizes = N_ dims ÷ the work-division split (out 384/2=192, mb 384/16=24).
#[derive(Serialize, Clone, Debug)]
pub struct StageParam {
    pub ss_: IterSpace,
    pub el_: IterSpace,
}

/// `sdscFolds_` — the time-fold descriptor (Affine identity + per-step map).
#[derive(Serialize, Clone, Debug)]
pub struct SdscFolds {
    pub dim_prop_func: Vec<FoldFunc>,
    pub dim_prop_attr: Vec<FoldProp>,
    pub data_: BTreeMap<String, String>,
}

/// One `dscs_` entry — the per-DesignSpaceConfig tile schedule. FRONTEND-MINIMAL,
/// matching torch-spyre's `generate_sdsc` inner dict: `numCoresUsed_`,
/// `numCoreletsUsed_:1`, `coreIdsUsed_`, `N_`, `coordinateMasking_`,
/// `maskingConstId_`, `dataStageParam_`, `primaryDsInfo_`, `scheduleTree_`,
/// `labeledDs_`, `constantInfo_`, `computeOp_`. The bmm SCHEDULER fixture's
/// extras (unpadN_/T_/Tel_/P_/Pel_/B_/ChipD_/CoreD_/CoreletD_/pdsRelation_/pcfg_/
/// dscN_/numCoreletsUsed_DSC2_/ChipletD_/auxLoopOrder_/dimToSymbolMapping_/
/// gtrIdsUsed_/l0TetheredMode_/scheduleTreeHeadDenId_/loopOrder_/loopProperties_)
/// are all scheduler outputs — DROPPED. `coreIdToDscSchedule` stays (op-level).
#[derive(Serialize, Clone, Debug)]
pub struct Dsc {
    pub numCoresUsed_: u32,
    pub numCoreletsUsed_: u32, // 1 (ACTIVE_CORELETS) — frontend constant
    pub coreIdsUsed_: Vec<u32>,
    pub N_: IterSpace,
    pub coordinateMasking_: BTreeMap<String, Vec<[i64; 2]>>,
    pub maskingConstId_: i64,
    pub dataStageParam_: BTreeMap<String, StageParam>,
    pub primaryDsInfo_: BTreeMap<&'static str, LayoutInfo>,
    pub scheduleTree_: Vec<AllocNode>,
    pub labeledDs_: Vec<LabeledDs>,
    // EMPTY → the JSON *string* "{}"; POPULATED → a dict. VERIFIED on torch-spyre's
    // REAL frontend (compiled `torch.rsqrt`/`F.rms_norm`): rsqrt/add/mul emit
    // `"constantInfo_": "{}"` (a str), the `mean` op's scaling_factor a dict. The earlier
    // claim that the string was an "old bug" (citing the bmm SCHEDULER fixture, NOT the
    // frontend) was WRONG and INVERTED: the empty OBJECT {} is falsy in dxp's Python so
    // the SFP constant-table / NR-refine setup is skipped → transcendentals collapse to
    // their seed (rsqrt→1/x). The empty→string conversion lives at the `constant_info`
    // build site. serde_json::Value carries either form.
    pub constantInfo_: serde_json::Value,
    pub computeOp_: Vec<ComputeOp>,
}

/// Top-level SuperDSC op (one `sdsc_{idx}.json`, serialized `{ "{op}_{id}": .. }`).
/// FRONTEND-MINIMAL, matching torch-spyre's `generate_sdsc` outer dict:
/// `sdscFoldProps_` (time=1), `sdscFolds_`, `coreFoldProp_`, `coreletFoldProp_`
/// (factor 1), `numCoresUsed_`, `coreIdToDsc_`, `numWkSlicesPerDim_`,
/// `coreIdToWkSlice_`, `coreIdToDscSchedule`, `dscs_`. The bmm SCHEDULER fixture's
/// extras (folded_sdsc_name_/fold_coord_/unpadN_/N_ at the op level, datadscs_/
/// opFuncsUsed_/target_/symbolDefinitions_/ldsShareInfo_/prodConsList/
/// inputSymbolsAndTags_/dimToSymbolMappingOpcodeCorrection_) are scheduler
/// outputs — DROPPED.
#[derive(Serialize, Clone, Debug)]
pub struct SdscOp {
    pub sdscFoldProps_: Vec<FoldProp>,
    pub sdscFolds_: SdscFolds,
    pub coreFoldProp_: FoldProp,
    pub coreletFoldProp_: FoldProp,
    pub numCoresUsed_: u32,
    pub coreIdToDsc_: BTreeMap<String, u32>,
    pub numWkSlicesPerDim_: BTreeMap<&'static str, u32>,
    /// Per-core slice indices, typed: a wire value here is provably a `SliceIndex` minted
    /// by `core_to_wk_slice`'s decomposition (serializes as the bare integer, wire-identical).
    pub coreIdToWkSlice_:
        BTreeMap<String, BTreeMap<&'static str, scratchy_subtile::superdsc_opspec::SliceIndex>>,
    pub coreIdToDscSchedule: BTreeMap<String, Vec<[i64; 4]>>, // {core:[[-1,0,0,0]]}
    pub dscs_: Vec<BTreeMap<String, Dsc>>,
}

/// `coreIdToDscSchedule`: one schedule entry `[[-1,0,0,0]]` per core (the fixture
/// form — a single dsc-0 step per core, no chunk/time sub-schedule).
fn core_dsc_schedule(cores: u32) -> BTreeMap<String, Vec<[i64; 4]>> {
    (0..cores.max(1))
        .map(|c| (c.to_string(), vec![[-1, 0, 0, 0]]))
        .collect()
}

/// `gen_coord_info_value` — PORTED VERBATIM from torch-spyre `compute_ops.py`.
/// The per-dim tile-fold for one allocate-node dim. `size` is the per-core extent
/// (full // split) for an active dim, else 1; `nsplits` is the split for active
/// dims, else 1; `is_stick_reduction` selects the reduction-stick variant.
fn gen_coord_info_value(
    size: i64,
    nsplits: i64,
    elems_per_stick: i64,
    is_stick_dim: bool,
    is_stick_reduction: bool,
    // Byte/elem stride between consecutive STICK-GROUPS along this dim. NATURAL layout =
    // `elems_per_stick` (sticks are contiguous). A RowBlocked tensor ([cols/eps, rows, eps])
    // interleaves all `rows` inside each stick-group, so its stick-groups are `rows·eps`
    // apart — a reduce over such a dim MUST step by that, else it sums across the wrong rows
    // (the mq>1 fp32 rmsnorm `mean(x²)` NaN: it read `sq32`'s stick-groups at stride 32 while
    // they physically sit 992 apart, gathering 64 different rows' first sticks + OOB).
    group_stride: i64,
) -> serde_json::Value {
    if !is_stick_dim {
        serde_json::json!({
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {"Affine": {"alpha_": size, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 1, "beta_": 0}},
                ],
                "dim_prop_attr": [
                    {"factor_": nsplits, "label_": "core_fold"},
                    {"factor_": 1, "label_": "corelet_fold"},
                    {"factor_": 1, "label_": "row_fold"},
                    {"factor_": size, "label_": "elem_arr_0"},
                ],
            },
        })
    } else {
        serde_json::json!({
            "spatial": 3,
            "temporal": 0,
            "elemArr": 2,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {"Affine": {"alpha_": if is_stick_reduction { elems_per_stick } else { size }, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": 0, "beta_": 0}},
                    {"Affine": {"alpha_": group_stride, "beta_": 0}},
                    {"Affine": {"alpha_": if is_stick_reduction { 0 } else { 1 }, "beta_": 0}},
                ],
                "dim_prop_attr": [
                    {"factor_": nsplits, "label_": "core_fold"},
                    {"factor_": 1, "label_": "corelet_fold"},
                    {"factor_": 1, "label_": "row_fold"},
                    {"factor_": if is_stick_reduction { 1 } else { size / elems_per_stick }, "label_": "elem_arr_1"},
                    {"factor_": elems_per_stick, "label_": "elem_arr_0"},
                ],
            },
        })
    }
}

/// The `coordinates_` object: `coordInfo` per layout dim (via the ported
/// `gen_coord_info_value`) + an empty `coreIdToWkSlice_`. `arg` is the view of the
/// tensor whose layout this allocate node owns; `plan` provides the extents +
/// splits. Mirrors compute_ops.py's per-dim loop: for an ACTIVE dim (scale == 1)
/// the fold uses the per-core extent (size // split) and the split; for a
/// reduction dim, size=1/nsplits=1 with the reduction-stick variant when -2.
fn build_coordinates(
    arg: &ArgView<'_>,
    // THE SAME layout `per_core_addr` gets (view_stick_layout, computed once at the call site) — now
    // DRIVES group_stride (StickLayout::group_stride) instead of a parallel hand-derivation from
    // `arg.scale`/`arg.row_blocked`. One proven decision, not two.
    layout: &scratchy_subtile::sdsc_abstract::StickLayout,
    plan: &WorkPlan,
    fp8_matmul: bool,
    is_reduction: bool,
) -> serde_json::Value {
    // Stick width from the operand's device format (fp8/int8 = 128, fp16/bf16 = 64, fp32 = 32) so the
    // coordInfo fold matches the emitted `stickSize_`; else dxp folds the fp8 dataspace at 64 (wrong).
    let eps = arg.df.elems_per_stick() as i64;
    let mut ci = serde_json::Map::new();
    for (i, &d) in arg.layout.iter().enumerate() {
        let scale = arg.scale[i];
        let split = plan.split_of(d) as i64;
        let full = plan.extent(d) as i64;
        let (size, nsplits) = if scale == Scale::Active {
            ((full / split.max(1)).max(1), split.max(1))
        } else {
            (1, 1)
        };
        let is_stick = arg.stick == d;
        // The fp8 W8A8 matmul operands use the AIU fp8 PE's NATIVE nested tile geometry (verified against
        // IBM's l0_tethering fixture, extracted via L3DlOpsScheduler_standalone): the WEIGHT's N-stick is
        // sub-tiled 8×8 and its K is 2-packed; the ACTIVATION's K-stick (128) is sub-tiled 8×2×8. A flat
        // single-stick fold (the fp16 shape) makes dsc2's fold-contiguity walk demand a "Loop split"
        // (dsc2.cpp:6379). These generators emit the exact nested folds so the walk stays contiguous.
        // Stick-group step: a RowBlocked tensor `[feat/eps, rows, eps]` interleaves all `rows` INSIDE each
        // stick-group, so consecutive stick-groups sit `rows·eps` apart, NOT `eps`. This is true of EVERY
        // mq>1 stick-major activation — the fp16 rank-2 stick-major (RowBlocked via `for_view_df`) and the
        // fp32 `row_blocked` island — for BOTH the producer (pointwise/rmsnorm write) and the consumer
        // (matmul-A / reduce read), not just the `is_reduction` reduce. Emitting `eps` (contiguous) made a
        // core read each row's 2nd..Nth stick from the NEXT row (row-mixing): the mq>1 matmul/reduce then
        // summed a single stick across 32 consecutive rows (+ zero padding) instead of one row's 32 sticks →
        // Σx²≈0 → amax≈0 → every projection inf. `rows==1` (decode) keeps `eps` byte-identical (rb_rows=1).
        // KERNELS are re-tiled by their RetileDescriptor (not this coordInfo), so they stay `eps`. The
        // `is_stick_reduction` accum collapses to `alpha_=0` regardless, so it also stays `eps`.
        // The stick-GROUP step: ONE call, `StickLayout::group_stride`, replacing the old two-fact
        // hand-derivation (`rb_rows` product + the `arg.row_blocked` gate). Proven fp16-safe /
        // fp32-eligible on the SAME layout `per_core_addr` uses for the start — see its doc.
        let group_stride = if is_stick && !scale.is_stick_reduction() {
            layout.group_stride(is_reduction)
        } else {
            eps
        };
        let info = match (fp8_matmul, arg.role, d) {
            (true, Role::Kernel, "out") => gen_fp8_kernel_out_fold(size, nsplits),
            (true, Role::Kernel, "in") => gen_fp8_kernel_in_fold(size, nsplits),
            (true, Role::Input, "in") => gen_fp8_input_in_fold(size, nsplits),
            _ => gen_coord_info_value(
                size,
                nsplits,
                eps,
                is_stick,
                scale.is_stick_reduction(),
                group_stride,
            ),
        };
        ci.insert(d.to_string(), info);
    }
    serde_json::json!({ "coordInfo": serde_json::Value::Object(ci), "coreIdToWkSlice_": {} })
}

/// fp8 WEIGHT `out` (N) fold: the 64-elem N-stick is sub-tiled 8×8 (elem_arr_1×elem_arr_0), with
/// `per_core_N/64` such sticks (elem_arr_2), split `nsplits` ways across cores. Alphas chain
/// contiguously (1, 8, 64, per_core) so dsc2's `alpha[curr]==alpha[next]·card[next]` holds.
fn gen_fp8_kernel_out_fold(per_core: i64, nsplits: i64) -> serde_json::Value {
    serde_json::json!({
        "spatial": 3, "temporal": 0, "elemArr": 3, "padding": "nopad",
        "folds": {
            "dim_prop_func": [
                {"Affine": {"alpha_": per_core, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 64, "beta_": 0}},
                {"Affine": {"alpha_": 8, "beta_": 0}},
                {"Affine": {"alpha_": 1, "beta_": 0}},
            ],
            "dim_prop_attr": [
                {"factor_": nsplits, "label_": "core_fold"},
                {"factor_": 1, "label_": "corelet_fold"},
                {"factor_": 1, "label_": "row_fold"},
                {"factor_": (per_core / 64).max(1), "label_": "elem_arr_2"},
                {"factor_": 8, "label_": "elem_arr_1"},
                {"factor_": 8, "label_": "elem_arr_0"},
            ],
        },
    })
}

/// fp8 WEIGHT `in` (K) fold: K is 2-PACKED (2 fp8 per fp16-width slot) — elem_arr_0=2 (α1), with
/// `K/2` such packs (elem_arr_1, α2). Contiguous (2=1·2).
///
/// `k` is the PER-CORE extent (`full/nsplits`, as build_coordinates passes it) and `nsplits` is the
/// `in` split, so the reconstructed extent is `core_fold · Π elem_arr = nsplits · (k/2) · 2 =
/// nsplits·k` = the full K. `core_fold` was previously HARDCODED to 1, written under the old
/// invariant that the reduction dim is never split; once the cost model started K-splitting, the
/// descriptor claimed 1/nsplits of K while `numWkSlicesPerDim_["in"]` and `per_core_addr` both
/// encoded a real nsplits-way split — the descriptor and the addressing disagreed. Its fp16 sibling
/// `gen_coord_info_value` has always carried the real `nsplits` here.
fn gen_fp8_kernel_in_fold(k: i64, nsplits: i64) -> serde_json::Value {
    serde_json::json!({
        "spatial": 3, "temporal": 0, "elemArr": 2, "padding": "nopad",
        "folds": {
            "dim_prop_func": [
                {"Affine": {"alpha_": k, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 2, "beta_": 0}},
                {"Affine": {"alpha_": 1, "beta_": 0}},
            ],
            "dim_prop_attr": [
                {"factor_": nsplits, "label_": "core_fold"},
                {"factor_": 1, "label_": "corelet_fold"},
                {"factor_": 1, "label_": "row_fold"},
                {"factor_": (k / 2).max(1), "label_": "elem_arr_1"},
                {"factor_": 2, "label_": "elem_arr_0"},
            ],
        },
    })
}

/// fp8 ACTIVATION `in` (K) fold: the 128-elem fp8 K-stick is sub-tiled 8×2×8 (elem_arr_0=8 α1,
/// elem_arr_1=2 α64, elem_arr_2=8 α8), with `K/128` sticks (elem_arr_3 α128).
/// The non-monotonic alphas (1,64,8,128) are the fp8 activation's physical interleave (from IBM's fixture).
///
/// `k` is the PER-CORE extent and `nsplits` the `in` split, so the reconstructed extent is
/// `core_fold · Π elem_arr = nsplits · (k/128) · 8 · 2 · 8 = nsplits·k` = the full K. See
/// [`gen_fp8_kernel_in_fold`] for why `core_fold` was wrongly hardcoded to 1.
fn gen_fp8_input_in_fold(k: i64, nsplits: i64) -> serde_json::Value {
    serde_json::json!({
        "spatial": 3, "temporal": 0, "elemArr": 4, "padding": "nopad",
        "folds": {
            "dim_prop_func": [
                {"Affine": {"alpha_": k, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 0, "beta_": 0}},
                {"Affine": {"alpha_": 128, "beta_": 0}},
                {"Affine": {"alpha_": 8, "beta_": 0}},
                {"Affine": {"alpha_": 64, "beta_": 0}},
                {"Affine": {"alpha_": 1, "beta_": 0}},
            ],
            "dim_prop_attr": [
                {"factor_": nsplits, "label_": "core_fold"},
                {"factor_": 1, "label_": "corelet_fold"},
                {"factor_": 1, "label_": "row_fold"},
                {"factor_": (k / 128).max(1), "label_": "elem_arr_3"},
                {"factor_": 8, "label_": "elem_arr_2"},
                {"factor_": 2, "label_": "elem_arr_1"},
                {"factor_": 8, "label_": "elem_arr_0"},
            ],
        },
    })
}

// (The former `lx_chunk_bytes` / `assert_chunk_stick_multiples` emit-time guard is
// GONE: the DtException-1535 chunk-size rule is now enforced BY CONSTRUCTION in
// `TensorArg::new` (the MaterializedStick invariant — a stick dim can't be phantom-
// scaled), so a non-128-multiple LX chunk is unrepresentable. No need to recompute
// `getBufferCapacityForNode` on the generated SDSC and assert on it.)

/// Per-arg HBM SEGMENT base addresses (torch-spyre `constants.py:44`
/// `SEGMENT_OFFSETS`, 16 GiB = `0x4_0000_0000` stride). Each I/O arg lives in its
/// OWN 16 GiB segment keyed by its `arg_index` (== `ldsIdx_` == the tensor's
/// position in `op.args`), so two cores' tiles of DIFFERENT tensors can never
/// alias, and a tiled op's per-trip tiles of the SAME tensor only ever advance
/// WITHIN one segment (task #50 — this is what lets GUARD #14 lift). There are 7
/// segments; an op with >7 distinct dataspaces is a build-time `Err` (we cannot
/// give every tensor a non-aliasing segment), surfaced by [`segment_base`].
pub const SEGMENT_OFFSETS: [u64; 7] = [
    0x0,
    0x4_0000_0000,
    0x8_0000_0000,
    0xC_0000_0000,
    0x10_0000_0000,
    0x14_0000_0000,
    0x18_0000_0000,
];
/// 16 GiB segment stride (torch-spyre `constants.py:55` `SEGMENT_SIZE`).
pub const SEGMENT_SIZE: u64 = 0x4_0000_0000;

/// Decompose an HBM byte address into its `(segment_index, intra_segment_offset)`.
/// Every HBM address our emitter produces is `SEGMENT_OFFSETS[seg] + intra` with
/// `intra < SEGMENT_SIZE` (each dataspace lives in its own 16 GiB segment), so the
/// decomposition is exact: `seg = addr / SEGMENT_SIZE`, `off = addr % SEGMENT_SIZE`,
/// and `SEGMENT_OFFSETS[seg] + off == addr`. This is what lets the SYMBOLIC bundle
/// emit each address as `arith.addi %segbase_{seg}, off` — the torch-spyre
/// `kernel_derived` scheme (`generate_bundle`), whose SHARED per-value operand SSA
/// (vs a fresh `arith.constant` per per-core symbol) is what dxp's symbol machinery
/// accepts. Kani-verified (`hbm_seg_off_reconstructs_address`).
pub fn hbm_seg_off(addr: u64) -> (u64, u64) {
    (addr / SEGMENT_SIZE, addr % SEGMENT_SIZE)
}

/// The HBM segment base for arg `arg_index`. A build-time `Err` (not a panic /
/// silent wrap) when `arg_index >= 7`: there are only 7 non-aliasing segments, so
/// an 8th distinct dataspace would have to share a segment and could alias — that
/// must be a `cargo build` failure, never a silently-wrong on-card address.
fn segment_base(arg_index: usize) -> Result<u64, SuperDscError> {
    SEGMENT_OFFSETS.get(arg_index).copied().ok_or_else(|| {
        SuperDscError(format!(
            "per-core HBM addressing: arg_index {arg_index} exceeds the {} available 16 GiB \
             HBM segments (SEGMENT_OFFSETS, torch-spyre constants.py:44). An op with >{} \
             distinct dataspaces cannot give each its own non-aliasing segment — refusing to \
             bake (would alias on-card). Fuse/spill some operands or split the op.",
            SEGMENT_OFFSETS.len(),
            SEGMENT_OFFSETS.len()
        ))
    })
}

// ───────────────────────────────────────────────────────────────────────────
// GLOBAL bundle memory layout (task #55) — the fix for the multi-op address bug.
//
// The per-op `segment_base(arg_index)` above places each op's args in segments by
// their PER-OP-LOCAL position (arg 0→seg0, arg 1→seg1, …). That is correct for a
// SINGLE op but WRONG for the 697-op fused bundle: every op reuses seg0..6, so
// op-5's weight and op-300's activation both bake to seg1 and ALIAS. The fix is a
// GLOBAL layout the emitter owns: every distinct tensor (by SubtileIR id) gets ONE
// (segment, byte-offset) for the whole bundle, keyed by ROLE not arg position. A
// single fused program then addresses (segment, offset); the executor binds ≤7
// segment regions (one per role) — defeating the 7-tensor-segment HW cap by packing
// many tensors into one region at interior offsets (flex `defines.hpp` segment_id =
// top 3 bits of the operand DMVA; one region = one segment_id).
// ───────────────────────────────────────────────────────────────────────────

/// RESERVED tensor id for the RoPE rotate-half permutation matrix `P` (in-bundle
/// RoPE). It is NOT a SubtileIR tensor (no model source produces it) — the emitter
/// references it as `t{ROPE_P_TID}` in `lower_rope_node`'s `matmul(x, P)`, places it
/// as a seg0 ACTIVATION in [`compute_bundle_layout`] (re-bound per step like RMS_SEED —
/// a seg1 weight is only H2D'd at PrepareModel, which binds ONLY the manifest weights,
/// so a synthetic seg1 P would stay ZERO), and the WORKER recognizes this id to
/// synthesize + bind the fixed matrix each step. Chosen far above any real tensor id.
pub const ROPE_P_TID: u32 = u32::MAX - 1;

/// RESERVED tensor id for the attention `scale` constant (`1/sqrt(head_dim)`), a
/// `[1,1]` fp16 the worker synthesizes + binds (seg1). Shared across all layers.
pub const ATTN_SCALE_TID: u32 = u32::MAX - 2;

/// RESERVED tensor id for the attention additive PREFIX length-mask `[mq, cap]` (0
/// on valid prefix cols `[0..p)`, −inf elsewhere) — a per-step ACTIVATION (seg0) the
/// worker binds each step. Broadcast over the nqh batch. Shared across layers.
pub const ATTN_MASK_TID: u32 = u32::MAX - 3;

/// RESERVED tensor id for the attention additive CAUSAL mask `[mq, mq_pad]` over the
/// NEW chunk (0 where new token i attends new token j ≤ i, −inf above the diagonal
/// and on the `[mq..mq_pad)` stick-pad) — a per-step ACTIVATION (seg0) the worker
/// binds (the `triu(-inf,diag=1)` of the reference SDPA). Broadcast over nqh. For
/// decode (mq=1) it is `[1, mq_pad]` with col 0 = 0 (the token attends itself).
pub const ATTN_CAUSAL_TID: u32 = u32::MAX - 4;

/// RESERVED tensor id for the RMSNorm Newton-rsqrt SEED FLOOR constant (`≈1e-4`), a
/// `[1,1]` fp16 the worker synthesizes + binds (seg0 activation). The decomposed rmsnorm
/// computes `inv = 1/√meps` by a range-safe Newton iteration `y ← y·(1.5 − 0.5·meps·y²)`
/// (uses `meps` DIRECTLY — no `reciprocal(meps)` that underflows fp16 for large residuals).
/// Its seed is `max(reciprocal(meps), SEED_CONST)`: `reciprocal(meps)` is a good seed for
/// small meps (early layers) and underflows→~0 for large meps, where this floor takes over —
/// so ~10 iterations converge across the whole residual-stream dynamic range. The iteration
/// constants `1.0/0.5/1.5` are derived ON-CARD from this one bound value via bounded
/// reciprocals (`1 = SEED·recip(SEED)`, `½ = recip(1+1)`, `1½ = 1+½`), so only ONE const
/// needs binding. (Sengraph gets a working native `Rsqrt` free from CompileGraph; the
/// SuperDSC per-op-dxp `rsqrt`/`sqrt` return the unrefined seed, so we build it explicitly.)
pub const RMS_SEED_TID: u32 = u32::MAX - 5;

/// RESERVED tensor id for the RMSNorm Newton constant `0.5` (`[1,stick]`, worker-bound).
/// Bound DIRECTLY (not derived from [`RMS_SEED_TID`] via `reciprocal`): the seed floor is
/// ~5e-6, and `reciprocal(5e-6)=2e5` OVERFLOWS fp16 (max 65504)→inf, so deriving `1.0 =
/// seed·recip(seed)` produced inf consts. `0.5` is exact in fp16; `1.5` is `0.5+0.5+0.5`.
pub const RMS_HALF_TID: u32 = u32::MAX - 6;

/// RESERVED tensor id for the RMSNorm `1/cols` constant (`[1,stick]`, worker-bound = `1/hidden`). Used
/// ONLY by the mq>1 (prefill) SUM-BASED amax: reduce-MAX is unusable on a multi-row TENSOR (returns seed 0
/// even with a per-row mb=1 slice — CONFIRMED, Kani `rmsnorm_max_reduce_seeds_on_multirow_tensor`), so amax
/// = `sum|x|` via a MULTI-ROW SUM. Pre-scaling `|x|·(1/cols)` before the sum keeps every partial ≤ max|x| ≤
/// fp16-max (no overflow, Kani `rmsnorm_sumbased_amax_finite`); `ramax = recip(mean|x|)·(1/cols) = 1/sum|x|`.
pub const RMS_INVCOLS_TID: u32 = u32::MAX - 11;

/// DIAGNOSTIC (2026-07-08): a persistent seg0 probe tid holding LAYER-0's `new_v` (the V-proj output,
/// PRE-selector) `[mq_pad, nkvh·hd]`. The structural mq>1 inf is in the shared M=8 proj-matmul/selector
/// path (K+V caches both inf, rope exonerated). This splits it: the emitter copies layer-0 (k_id==9) new_v
/// here BEFORE the selector; the worker reads it. INF ⇒ the V-proj M=8 matmul is the source; FINITE ⇒ the
/// selector (which reads new_v's mqp-padded [mqu..mqp) rows) is. Persistent (never reused) so it survives.
pub const NEW_V_PROBE_TID: u32 = u32::MAX - 12;

/// DIAGNOSTIC (mq>1 fp32 rmsnorm root-cause, gated on `SCRATCHY_SUPERDSC_ISLAND_PROBE`): persistent
/// fp32 probe tids holding the reduce output `var32` and the final Newton `rsqrt` (per row). The
/// approved measurement (audit) reads these to localize the batched `V=0`: `var32` tiny ⇒ the reduce
/// output-side is broken (⇒ matmul-by-ones fix); `var32` finite but `rsqrt` huge ⇒ the Newton
/// broadcast diverges. Persistent (never reused) so the value survives to the worker readback.
pub const RMS_VAR_PROBE_TID: u32 = u32::MAX - 18; // MAX-16 collides with ONES_REDUCE_TID
// ⛔ MAX-17, NOT MAX-19 — MAX-19 IS `IDENTITY_TID`. The probe moved, not
// IDENTITY: IDENTITY is on the live KV-matmul path and any already-baked bundle
// encodes MAX-19 as IDENTITY, so moving IT would silently reinterpret existing
// artifacts. MAX-17 was the one free slot in MAX-1..MAX-19.
// Locked by `sentinel_tids_are_pairwise_distinct`, which was RED on this line.
pub const RMS_RSQRT_PROBE_TID: u32 = u32::MAX - 17;

/// fp8 W8A8 activation-quant constants (worker-bound f16, seg0, shape [1, stick]): the E4M3 clamp bounds
/// `+448` / `-448` and `1/448` (for `a_scale = amax·(1/448)`). Placed when the tape has an fp8 (arity-3)
/// MatmulTile. `qfp8ch` requires its input CLAMPED into [-448,448] (fp16 rounding can push a per-token-scaled
/// value past 448 — the bound is NOT free, per the clamp guard on [`scratchy_subtile::superdsc_opspec::OpFunc::Qfp8ch`]).
pub const FP8_POS448_TID: u32 = u32::MAX - 13;
pub const FP8_NEG448_TID: u32 = u32::MAX - 14;
pub const FP8_INV448_TID: u32 = u32::MAX - 15;

/// RESERVED tid for the mq>1 (prefill) MATMUL-BY-ONES row-sum weight — a `[hidden, stick]` all-ones fp16
/// seg0 ACTIVATION (worker-bound flat like the head-major selectors).
/// The on-card reduce returns the SEED (0) for any tensor with >1 PHYSICAL row (#33), so the multi-row
/// rmsnorm `sum|x|`/`mean(xs²)` reduces (and the softmax denom) yield 0 → recip(0)=inf → the whole prefill
/// (NEW_V, K/V) goes inf. matmul IS proven correct AND multi-row on-card (it is the q/k/v/o proj path), so
/// `matmul(A[rows,cols], ones[cols,stick])` gives Σ_c A[r,c] in EVERY output col — a drop-in row-sum. n=stick
/// (=64, one output stick) ⇒ the device retile is identity of row-major, and an all-ones tile is
/// retile-invariant, so it needs NO RetileDescriptor (exactly like the selectors). `cols≤hidden` ⇒ w_off=0
/// reads the first `cols` all-ones rows. Placed + bound iff the bundle has a multi-row (prefill) rmsnorm;
/// a decode (m=1) bundle omits it, byte-identical to the pre-fix baseline. Prefill-only (mq>1). Free tid
/// (u32::MAX-16; -17/-18 = Claude2's block_table/slot_mapping, -19 = IDENTITY_TID, up to SCALARMUL_BASE=-20).
pub const ONES_REDUCE_TID: u32 = u32::MAX - 16;

/// RESERVED tensor id for the mq>1 PREFILL KV-CACHE-WRITE matmul-by-IDENTITY (#3) — a `[hd,hd]`
/// identity, worker-bound seg0 const (bound like [`ONES_REDUCE_TID`]). The mq>1 per-slot cachewr
/// used `nqh·mqu·2` single-row `[1,hd]` `slot_no_fuse` copies (SlotSolo singletons) because the
/// on-card multi-row `[mqu,hd]` copy is BROKEN (writes nothing). Mirroring #0's reduce→matmul-by-ones,
/// the copy becomes `matmul(kh[mqu,hd], I[hd,hd]) = kh` — matmul is the ONE m>1-correct primitive — so
/// ONE fusable op per head replaces `mqu` singletons (1,984→64 for granite-2b prefill). For hd==64 the
/// identity is single-stick (retile == row-major, like the all-ones), so no `kernel_weights` descriptor.
/// Placed only when the tape has an mq>1 `AttnDecode` AND `SCRATCHY_SUPERDSC_KV_MATMUL` (default-off A/B).
pub const IDENTITY_TID: u32 = u32::MAX - 19;

/// RESERVED tensor id for the mq>1 PREFILL HEAD-MAJOR SELECTOR weights — `nqh` stacked one-hot
/// `[nqh·hd, hd]` column-selectors (total `[nqh·nqh·hd, hd]`, worker-bound f16 const, seg0). For head
/// `h`, `Sel_h` (row-block `h`, element offset `h·nqh·hd·hd`) has `Sel_h[i,j]=1` iff
/// `i == `[`selector_head_src_col`](sdsc_abstract::selector_head_src_col)`(h,hd,j)` — so
/// `matmul(q[mq,nqh·hd], Sel_h) = q_h[mq,hd]` = head h's columns, written head-major at `h·mq·hd`. The
/// ONLY deployed way to head-major-ize row-major q/k/v for the per-head mq>1 attention ops (a strided
/// read / 3D reshape / restickify all fail — see [`kcache_kt_write_offset`] + Kani `selector_extracts_head_column`).
/// Fixed (position/layer-independent) ⇒ bound ONCE at prepare, like [`ROPE_P_TID`]. Placed only when the
/// tape has an mq>1 `AttnDecode` (the prefill bundle); the mq=1 decode bundle never references it.
pub const SEL_HEADMAJOR_TID: u32 = u32::MAX - 7;

/// RESERVED tid for the mq>1 prefill KV-width head-major selector — `nkvh` stacked one-hot
/// `[nkvh·hd, hd]` selectors (worker-bound f16 const, seg0). Same role as [`SEL_HEADMAJOR_TID`] but for
/// the `nkvh`-headed new-K/new-V `[mq_pad, nkvh·hd]` → head-major `[nkvh, mq_pad, hd]` (a distinct k-dim
/// `nkvh·hd`, so a distinct selector). Placed only for the prefill (mq>1) bundle.
pub const SEL_KV_HEADMAJOR_TID: u32 = u32::MAX - 8;

/// RESERVED tid for the mq>1 prefill INVERSE head-major selector — `nqh` stacked one-hot `[hd, nqh·hd]`
/// selectors (`SelT_h[j,c]=1` iff `c == h·hd+j`, worker-bound f16 const, seg0). Scatters the head-major
/// attention output `out_h[mq,hd]` back to ROW-MAJOR `out[mq, nqh·hd]` (columns `[h·hd,(h+1)·hd)`) that
/// `o_proj` reads: `out += matmul(out_h[mq,hd], SelT_h[hd, nqh·hd])`. Placed only for the prefill bundle.
pub const SELT_HEADMAJOR_TID: u32 = u32::MAX - 9;

/// RESERVED tid for the mq>1 prefill ZERO-PAD const — a worker-bound `[mqp, hd]` f16 ZERO buffer (seg0).
/// The head-major K/V selector writes only the `mqu` REAL rows of `kh`/`vh` (`[nkvh, mqp, hd]`); rows
/// `[mqu..mqp)` (stick padding) are UNINITIALIZED. The restickify+score then reads those as garbage K in
/// the masked padding columns `[mqu..mqp)`, and the NO-MAX softmax `exp(garbage·scale + MASK_NEG)`
/// OVERFLOWS to inf (Kani `prefill_softmax_padding_must_be_zeroed`, fail-first) → the whole prefill goes
/// inf. Copying this zero into `kh`/`vh` padding rows makes the padding-col score 0 ⇒ `exp(MASK_NEG)=0`
/// (safe). Placed only when the tape has an mq>1 AttnDecode.
pub const ATTN_ZERO_TID: u32 = u32::MAX - 10;

/// RESERVED tensor id for the mq>1 PREFILL LAST-ROW HIDDEN `[1, hidden]` — the last prompt token's
/// final-norm output, extracted from the `[mq, hidden]` residual stream so the lm_head tail can run at
/// `m=1` INSIDE the prefill bundle (folding away the separate decode-of-the-last-token forward, which
/// existed only to make the first generated token's logits and re-paid the whole weight-stream floor).
///
/// NOT worker-bound: it is written ON-CARD by [`lower_one_node`]'s per-stick copies and read by the
/// re-lowered lm_head. It is registered as a SYNTH under this name (`t{LAST_HIDDEN_TID}`), so it costs
/// one intermediate-segment reservation and NO manifest placement — the decode bundle never references
/// it and is byte-identical.
///
/// WHY A COPY AND NOT A ONE-HOT MATMUL: `hidden[mq, hidden]` is `RowBlocked`, so row `mq-1` lives in
/// `hidden/64` chunks of 64 elements at stride `mq·64` — a `sel[1,mq] @ hidden[mq,hidden]` extraction
/// would need `k = mq` to be a whole 64-stick (the rungs are 15/23/31/39/47/63/80/96 — never), and
/// reading `hidden` at a row offset in ONE op would need a `rows·lanes` coordInfo stride, which
/// [`scratchy_subtile::sdsc_abstract::StickLayout::group_stride`] documents as unrepresentable for fp16 (dxp
/// `LX_MODLRFIMM` at exactly `rows·lanes = 31·64`). So the extraction is `hidden/64` single-stick
/// `[1,64]` identity copies — each self-consistently addressed at `lanes`, the one proven form.
pub const LAST_HIDDEN_TID: u32 = u32::MAX;

/// ⛔ THE SENTINEL TIDS MUST BE PAIRWISE DISTINCT, AND NOTHING WAS CHECKING.
///
/// These are hand-assigned `u32::MAX - N` magic numbers naming synthetic
/// tensors (constants, probes, selectors) that the emitter places and the worker
/// binds. Two names sharing one value means two DIFFERENT tensors resolve to one
/// placement: whichever is bound last wins and the other silently reads someone
/// else's bytes. There is no crash — the id is valid, it is just not yours.
///
/// This is not hypothetical. `RMS_VAR_PROBE_TID` carries the comment
/// "MAX-16 collides with ONES_REDUCE_TID", so the collision was hit ONCE and
/// fixed by hand — and then `RMS_RSQRT_PROBE_TID` was given MAX-19, which
/// `IDENTITY_TID` already had. A hand-assigned space with no uniqueness check
/// re-collides as soon as someone adds a name.
#[test]
fn sentinel_tids_are_pairwise_distinct() {
    // Every sentinel, by name, so a new one added without a slot shows up here.
    const SENTINELS: &[(&str, u32)] = &[
        ("LAST_HIDDEN", LAST_HIDDEN_TID),
        ("ROPE_P", ROPE_P_TID),
        ("ATTN_SCALE", ATTN_SCALE_TID),
        ("ATTN_MASK", ATTN_MASK_TID),
        ("ATTN_CAUSAL", ATTN_CAUSAL_TID),
        ("RMS_SEED", RMS_SEED_TID),
        ("RMS_HALF", RMS_HALF_TID),
        ("SEL_HEADMAJOR", SEL_HEADMAJOR_TID),
        ("SEL_KV_HEADMAJOR", SEL_KV_HEADMAJOR_TID),
        ("SELT_HEADMAJOR", SELT_HEADMAJOR_TID),
        ("ATTN_ZERO", ATTN_ZERO_TID),
        ("RMS_INVCOLS", RMS_INVCOLS_TID),
        ("NEW_V_PROBE", NEW_V_PROBE_TID),
        ("FP8_POS448", FP8_POS448_TID),
        ("FP8_NEG448", FP8_NEG448_TID),
        ("FP8_INV448", FP8_INV448_TID),
        ("ONES_REDUCE", ONES_REDUCE_TID),
        ("RMS_VAR_PROBE", RMS_VAR_PROBE_TID),
        ("RMS_RSQRT_PROBE", RMS_RSQRT_PROBE_TID),
        ("IDENTITY", IDENTITY_TID),
    ];
    let mut seen: std::collections::BTreeMap<u32, &str> = Default::default();
    let mut clashes: Vec<String> = Vec::new();
    for (name, tid) in SENTINELS {
        if let Some(prev) = seen.insert(*tid, name) {
            clashes.push(format!(
                "{name} and {prev} are both u32::MAX - {}",
                u32::MAX - *tid
            ));
        }
    }
    assert!(
        clashes.is_empty(),
        "sentinel tids collide — two synthetic tensors would share one placement: {clashes:?}"
    );
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE RESERVED TID SPACE — declared, bounded, and proven disjoint AT COMPILE TIME
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One reserved region of the tid space: a base and how many slots it owns, counting DOWN.
///
/// ⛔⛔⛔ THIS EXISTS BECAUSE THE REGIONS WERE RAW `u32` SUBTRACTION. Every reserved tid was
/// `SOME_BASE - index`, with each base defined in terms of the previous one, no bound on any
/// index, and nothing anywhere proving the regions do not overlap. `scalarmul_scale_tid(idx)` owns
/// exactly 100 slots before it walks into `KSPLIT_BLOCK_BASE`, and it took `idx: usize` unchecked;
/// `downproj_block_tid` multiplies a REAL TENSOR ID by a stride and subtracts that.
///
/// Two tensors landing on one tid is not an error anyone sees. They get ONE placement, so the
/// second one's producer writes over the first one's bytes and its consumer reads them — on a
/// device with no stack to attach to, which surfaces as wrong output or as a hang.
///
/// A region cannot be added or resized without [`RESERVED_REGIONS_ARE_DISJOINT`] re-proving the
/// whole layout, and no index can leave its region without [`Self::at`] refusing during the bake.
#[derive(Clone, Copy, Debug)]
pub struct TidRegion {
    /// For diagnostics — named so a refusal says WHICH region overflowed.
    pub name: &'static str,
    /// Highest tid in the region; slots count downward from here.
    pub base: u32,
    pub slots: u32,
}

impl TidRegion {
    /// Lowest tid this region owns.
    pub const fn floor(self) -> u32 {
        self.base - (self.slots - 1)
    }

    /// The `idx`-th tid, or a BUILD FAILURE. `idx >= slots` would silently alias the region below.
    pub const fn at(self, idx: u32) -> u32 {
        assert!(
            idx < self.slots,
            "reserved tid region overflow: this index is past the region's last slot and would \
             alias the region below it, giving two tensors one placement",
        );
        self.base - idx
    }

    const fn overlaps(self, other: TidRegion) -> bool {
        self.floor() <= other.base && other.floor() <= self.base
    }
}

/// Every reserved region, in one place. Order is high tid → low.
pub const RESERVED_REGIONS: [TidRegion; 3] = [
    // The single sentinels (`ROPE_P_TID` .. `IDENTITY_TID`) occupy MAX-1 .. MAX-19.
    TidRegion {
        name: "sentinels",
        base: u32::MAX - 1,
        slots: 19,
    },
    TidRegion {
        name: "scalarmul_scale",
        base: u32::MAX - 20,
        slots: 100,
    },
    // ⛔ THE GAP FROM MAX-120 TO MAX-1_000_170 IS DELIBERATELY LEFT EMPTY. It held the K-split
    // block/zero/down_proj regions, which are gone with the K-split itself. `kct_resident` keeps
    // its ABSOLUTE base rather than sliding up into the hole: every reserved id is a number that
    // has been baked into artifacts, and moving one to tidy the map would silently repoint it.
    // ⛔ BOUNDED, WHERE IT USED TO BE OPEN-ENDED. `kct_resident_tid(k_id) = BASE - k_id` had no
    // floor at all: a model with enough tensors walked it downward without limit. One million
    // slots is far past any real tid count and is now a REFUSAL rather than a wrap.
    TidRegion {
        name: "kct_resident",
        base: u32::MAX - 1_000_171,
        slots: 1_000_000,
    },
];

/// ⭐ THE PROOF, EVALUATED AT COMPILE TIME. Adding or resizing a region re-runs it; an overlap is
/// a build error naming both regions, not a tensor that quietly answers to two names.
#[allow(clippy::let_unit_value)]
const _: () = {
    // Referencing both proofs is what makes the compiler EVALUATE them; an unused `const` item is
    // not const-evaluated, so a lock nothing mentions is a lock that never runs.
    let _ = SENTINELS_ARE_INSIDE_THEIR_REGION;
    let _ = RESERVED_REGIONS_ARE_DISJOINT;
};

pub const SENTINELS_ARE_INSIDE_THEIR_REGION: () = {
    // ⛔ THE ONE-OFF SENTINELS ARE DECLARED SEPARATELY (`ROPE_P_TID` .. `IDENTITY_TID`), so the
    // region table would happily describe a span they had already outgrown. Adding a 20th sentinel
    // now fails the build instead of silently taking `scalarmul_scale_tid(0)`'s slot.
    let r = RESERVED_REGIONS[0];
    assert!(r.base == u32::MAX - 1, "sentinels start at MAX-1");
    assert!(
        IDENTITY_TID >= r.floor(),
        "a one-off sentinel has fallen below the sentinel region and into the scalarmul scales",
    );
    assert!(
        SCALARMUL_SCALE_BASE < r.floor(),
        "the scalarmul region overlaps the one-off sentinels",
    );
};

pub const RESERVED_REGIONS_ARE_DISJOINT: () = {
    let mut i = 0;
    while i < RESERVED_REGIONS.len() {
        let mut j = i + 1;
        while j < RESERVED_REGIONS.len() {
            assert!(
                !RESERVED_REGIONS[i].overlaps(RESERVED_REGIONS[j]),
                "two reserved tid regions overlap — some tid names two different tensors",
            );
            j += 1;
        }
        // Regions descend, and none may run below the last one's floor into REAL tensor ids.
        assert!(
            RESERVED_REGIONS[i].floor() < RESERVED_REGIONS[i].base + 1,
            "a reserved region wrapped past zero",
        );
        i += 1;
    }
};

/// Base of the RESERVED tensor-id block for granite ScalarMul scale constants. The `i`-th DISTINCT
/// scale (see [`BundleLayout::scalarmul_scales`]) is bound to `SCALARMUL_SCALE_BASE - i` — a
/// worker-bound `[1,1]` fp16 the pointwise `mul` reads broadcast (the ATTN_SCALE mechanism). Well
/// below the other reserved ids (u32::MAX-1..-6) with room for many distinct scales.
pub const SCALARMUL_SCALE_BASE: u32 = RESERVED_REGIONS[1].base;

/// Reserved const TID for the `i`-th distinct ScalarMul scale.
pub fn scalarmul_scale_tid(idx: usize) -> u32 {
    RESERVED_REGIONS[1].at(idx as u32)
}

/// Base of the RESERVED tid block for the RESIDENT per-layer Kᵀ kernel `kct` (the "kill the O(active)
/// restickify" residency fix). The Kᵀ kernel the score matmul reads was a per-step SYNTH scratch that
/// re-transposed the whole active K each step; making it a RESIDENT seg2 tensor (persists across
/// steps and layers, so the incremental slab restickify only re-transposes the current slab) needs a
/// real tid with a per-layer seg2 placement. KEYED ON THE K-CACHE SOURCE TID `k_id` (like [`downproj_block_tid`])
/// so the emitter (which has layer-0 `k_id`) and the layout/guard (which iterate every layer's `k_id`)
/// compute the SAME kct tid with no layer-index handoff. A 1M gap below [`DOWNPROJ_BLOCK_BASE`] keeps it
/// disjoint from the down_proj blocks (which descend only ~tens-of-K) AND from real source tids (~thousands).
pub const KCT_RESIDENT_BASE: u32 = RESERVED_REGIONS[2].base;
/// Reserved tid for the resident per-layer Kᵀ kernel of the K-cache source tid `k_id`.
pub fn kct_resident_tid(k_id: u32) -> u32 {
    RESERVED_REGIONS[2].at(k_id)
}

/// HBM SEGMENT each tensor ROLE owns in the packed bundle (≤7; seg7 is the
/// flex-reserved program segment, never a data operand). Indexes [`SEGMENT_OFFSETS`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize)]
pub enum SegRole {
    // DIAGNOSTIC SWAP (2026-06-28): Intermediate↔Activation (was 0/3) to land the SFP
    // transcendental's operands (meps/inv = Intermediates) in REGION 0 like torch-spyre's
    // working rsqrt (input region1 / output region0), vs scratchy's region3. Tests whether
    // the operand region is the lever for the SFP Newton-Raphson refine (the SOLE remaining
    // byte-difference vs torch-spyre's byte-identical program). All segs are MemoryType::Tensor
    // so this SHOULD be inert — if it refines, the region matters despite same memory type.
    /// Per-step graph inputs (embeddings / cos / sin / positions) — DMA'd each step.
    Activation = 3,
    /// Model weights — packed, uploaded ONCE, resident across decode steps.
    Weight = 1,
    /// Paged K/V cache — resident (reserved; attention is host-routed today, see
    /// `lower_graph_to_superdsc` — populated once attention moves on-device).
    Kv = 2,
    /// Produced-and-consumed scratch — lifetime-colored (slots reused across
    /// non-overlapping live ranges, sized to max-concurrent-live, not the sum).
    Intermediate = 0,
    /// The single graph output (logits) — D2H'd each step.
    Logits = 4,
}

/// Pages one request's block table can address — the validity rows the mask reserves.
///
/// 32 is chosen so the reservation is `32 * 256 * 2` = 16384 B, byte-identical to the pre-paged
/// mask (`nqh * mq * cap * 2` at mq=1), which keeps every activation placed after it at the offset
/// it had before paging. A larger reserve pushes seven activation tensors along by its excess, and
/// this path is documented as placement-sensitive. 32 pages is 8192 positions per request; the
/// binding limit in practice is the runtime pool.
pub const MAX_PAGES_PER_REQUEST: u64 = 32;

/// Fold passes a batched-decode bundle reserves prefix-mask room for.
///
/// A pass is one `(request, page)`, and EVERY page comes from the one shared pool, so the passes in
/// flight can never exceed the pool's size however the batch is shaped: 32 requests of one page and
/// one request of 32 pages are the same 32 passes. Sizing the mask as
/// `requests * MAX_PAGES_PER_REQUEST` instead counts a batch where every request is simultaneously
/// at full context, which the pool cannot hold — at 32 requests that is 537 MB of mask against 17 MB
/// of actual worst case, and it is re-uploaded every step.
pub const MAX_FOLD_PASSES: u64 = 64;

impl SegRole {
    pub fn segment(self) -> usize {
        self as usize
    }
}

/// One tensor's global placement in the packed bundle.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TensorPlacement {
    /// SubtileIR tensor id (the stable global name is `t{id}`).
    pub tid: u32,
    pub role: SegRole,
    pub segment: usize,
    /// Byte offset WITHIN the segment region (128 B aligned).
    pub offset: u64,
    /// Byte size (f16 = 2 B/elem, rows*cols*2).
    pub size: u64,
}

/// The device-tile re-tile spec the executor (C++ shim) consumes to stage a matmul
/// KERNEL weight in the PT array's device layout. DERIVED SOLELY from the
/// [`DeviceTileLayout`] witness (`host_retile_descriptor`-style), so the shim's
/// re-tile and the emitter's per-core address read the SAME device layout — they
/// cannot diverge. `device(coord)` reads `host[Σ coord·stride_map]`, written in
/// contiguous `device_size` order.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RetileDescriptor {
    /// Device extents in dim_map order, e.g. `[out/64, in, 64]` for a `[in,out]` KERNEL.
    pub device_size: Vec<u64>,
    /// Host-element stride per device axis, e.g. `[64, out, 1]`.
    pub stride_map: Vec<u64>,
    /// Elements per stick (64 for fp16).
    pub stick_size: u32,
    /// Bytes per element (2 for fp16).
    pub word_length: u32,
}

/// The whole-bundle memory plan: every tensor → (segment, offset, size, role),
/// plus each segment's total packed bytes. Exported as `bundle_layout.json` for
/// the resident executor (which allocates one region per occupied segment).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BundleLayout {
    /// Placement per tensor id, ordered by id for determinism.
    pub placements: std::collections::BTreeMap<u32, TensorPlacement>,
    /// IDENTITY BY OPERAND SPELLING — populated wherever a name is MINTED.
    ///
    /// ⛔ THIS EXISTS TO DELETE A PARSE. `resolve_seg_base` used to recover a tensor's id with
    /// `name.strip_prefix('t').and_then(|s| s.parse::<u32>())` — the emitter rendering `t{tid}`
    /// at one site and reading the number back out of the rendering at another, which is the
    /// compile-time-fact-through-a-string round trip in its purest form. An operand carries a
    /// STRING because the layout handle it comes from (`scratchy_subtile`'s `Stk`) is shared,
    /// target-neutral code that must not learn a Spyre type; so spyre keeps its own table from
    /// that spelling to its own identity, filled where the spelling is created.
    #[serde(skip)]
    pub ids: std::cell::RefCell<std::collections::BTreeMap<String, bundle::PlaceId>>,
    /// Bytes occupied per segment (index = segment id 0..6).
    pub segment_bytes: [u64; 7],
    /// Matmul KERNEL weights → their device-tile [`RetileDescriptor`] (built ONLY via
    /// `DeviceTileLayout`, the same witness the per-core address uses). The PT array
    /// reads the device TILE layout `[out/64, in, 64]`, NOT the row-major flat bytes;
    /// the shim re-tiles every weight in this map during H2D staging via the descriptor.
    /// Tensors absent here (activations, embed gather table, 1-D rmsnorm gains) stay
    /// flat — `[1,N]`/`[N]` are tiling-invariant. Includes per-layer copies.
    #[serde(default)]
    pub kernel_weights: std::collections::BTreeMap<u32, RetileDescriptor>,
    /// Distinct granite ScalarMul scale VALUES (embedding/residual/attn/logits multipliers), in a stable
    /// order. Index `i` ↔ reserved const TID [`scalarmul_scale_tid`]`(i)` (a worker-bound `[1,1]` const,
    /// exactly the ATTN_SCALE mechanism). The emitter (`lower_scalarmul_node`) looks up its scale's index
    /// here → the const TID it multiplies by; the WORKER reads this list (serde) and binds each
    /// `t{tid} = [scale]`. So the emitter and worker agree on scale↔TID by construction (no reward-hack
    /// fold, no host-route — the op is a real on-device pointwise `mul`).
    #[serde(default)]
    pub scalarmul_scales: Vec<f32>,
    /// SYNTHETIC intermediates created during lowering (silu's `{out}_silu`,
    /// rmsnorm's `{out}_sq/mean/meps/inv/tmp/eps`) are NOT SubtileIR tensors, so
    /// they have no `t{id}` placement. They are lazily assigned a STABLE offset in
    /// the Intermediate segment (seg3) ABOVE the colored intermediates, so a
    /// producer and consumer of the same synthetic name resolve to the SAME address
    /// and they NEVER collide with weights (seg1)/activations (seg0). Interior-
    /// mutable so `resolve_seg_base` can allocate on first sight through `&layout`. Exported to
    /// `bundle_layout.json` (segment is always [`SegRole::Intermediate`], offset is `synth.map[name]`)
    /// so a runtime diagnostic can look up a synthetic intermediate's address by NAME, the same way
    /// it looks up a `t{id}`'s.
    pub synth: std::cell::RefCell<SynthAlloc>,
    /// THE arrangement authority: `tensor name → its ONE device [`StickLayout`]`. The FIRST op to
    /// address a tensor declares its layout; every later op must address it EQUIVALENTLY
    /// ([`StickLayout::addr_eq`]) or `declare_arrangement` returns a build `Err` naming the tensor. A
    /// tensor has ONE physical layout — so a producer that writes it stick-major and a consumer that
    /// reads it flat (the M>1 prefill scramble) is a COMPILE-TIME failure, not silently-wrong bytes.
    /// Interior-mutable so the `&layout`-threaded `emit_sdsc` can declare on first sight.
    #[serde(skip)]
    pub arrangements: std::cell::RefCell<
        std::collections::BTreeMap<String, scratchy_subtile::sdsc_abstract::StickLayout>,
    >,
    /// BYTES BETWEEN TWO REQUESTS' KV inside one page+layer — [`PagedKvPool::request_stride`] in
    /// bytes, or 0 when this bundle has no paged KV.
    ///
    /// The EMITTER's number, travelling to the runtime rather than being re-derived there. Every
    /// baked KV address already has the request dimension in it (`block_index` puts `ROWS` requests
    /// inside each kv head), so the launch shifting the KV segment by `r * this` is the other half of
    /// one law. A runtime that computed its own would be free to disagree, and disagreeing means
    /// reading another request's keys under a mask that thinks they are this one's — fluent output,
    /// wrong tokens, nothing to catch it.
    #[serde(default)]
    pub kv_request_stride_bytes: u64,
}

/// Bump allocator for synthetic-intermediate offsets within the Intermediate
/// segment (task #55). `next` starts at the colored-intermediate high-water; each
/// distinct synthetic name gets one 128B-aligned offset, reused on later sight.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SynthAlloc {
    pub next: u64,
    pub map: std::collections::BTreeMap<String, u64>,
    /// Declared FULL footprint (bytes) per synth name — set by [`BundleLayout::synth`].
    /// Present ⇒ the tensor OWNS its shape: `resolve_seg_base` reserves this whole size up
    /// front (so a per-head write at offset h·stride can't land on the next tensor) and GUARDS
    /// that no access exceeds it — an out-of-footprint access is a shape mismatch and a
    /// `cargo build` Err, not silent on-card aliasing.
    pub sizes: std::collections::BTreeMap<String, u64>,
}

/// Footprint (bytes) of a synth tensor with shape `dims` (outermost→innermost; the inner dim
/// is the stick axis, padded up to the dtype's stick-elem multiple). The allocator reserves THIS —
/// the tensor's true size — not a single per-op view (which under-reserved multi-head tensors → HBM
/// overlap). The dtype is the tensor's typed [`Df`]: fp32 split-K merge partials are 4-byte /
/// 32-elem-stick, fp8/int8 are 1-byte / 128-elem-stick (½ fp16), everything else fp16 (2-byte /
/// 64-elem-stick). INERT for fp16 (the working 1318-emit build is unchanged).
fn synth_footprint_bytes(dims: &[u32], df: Df) -> u64 {
    let stick_elems = df.elems_per_stick();
    let word_length = df.word_length();
    let inner = dims.last().copied().unwrap_or(stick_elems);
    let inner_stick = inner.div_ceil(stick_elems) * stick_elems;
    let outer: u64 = dims
        .iter()
        .rev()
        .skip(1)
        .map(|&d| d as u64)
        .product::<u64>()
        .max(1);
    outer * inner_stick as u64 * word_length as u64
}

/// ⛔⛔⛔ THE ADDRESS AUDIT — A LAYOUT THAT COULD FAULT MAY NOT COMPILE.
///
/// Everything up to the SuperDSC emission runs in the `#[forward]` procmacro, so every placement,
/// every segment extent and therefore every DMA window this bundle will ever generate is a
/// COMPILE-TIME CONSTANT. There is no reason for the card to be the thing that discovers a bad
/// address: it discovers them as `0xa35e RAS::PCI::BusFence`, which arrives with no host stack, no
/// statement of which transfer caused it, and a bus that needs resetting.
///
/// I had been asserting these at CB-build time, on the card, against numbers the bake computed —
/// checking at runtime a fact that was constant. This is the same audit, moved to where the
/// constants are, where the failure is a `cargo build` error naming the tensor.
///
/// Each check is a fault mechanism, not a tidiness rule:
///
///  * **Flit alignment.** A device address is written in FLITS (`>> 7`). An offset that is not a
///    multiple of 128 truncates DOWNWARD, so the transfer addresses a different location than its
///    length field describes. IBM's own validator for this is commented out in `qg.h`.
///  * **Inside its segment.** A placement running past `segment_bytes[seg]` addresses memory the
///    segment does not own — I1's inner half, which the device checks only on one path.
///  * **No overlap.** Two placements sharing bytes means one tensor's producer overwrites the
///    other's, which is silent: wrong output, or a fault when the second is a kernel.
///  * **Non-empty.** A zero-length field is read by the device as `1 << 27` flits = 16 GiB, in both
///    `handleHostDMA` and `handleXLATentry`. Zero does not mean nothing.
fn audit_layout_addresses(places: &[bundle::Placement], segment_bytes: &[u64; 7], fp: &str) {
    let mut fail: Vec<String> = Vec::new();

    for p in places {
        let seg = p.segment as usize;
        if seg >= segment_bytes.len() {
            fail.push(format!("{}: segment {seg} does not exist", p.id));
            continue;
        }
        if p.offset % 128 != 0 {
            fail.push(format!(
                "{} is at offset {} in seg{seg}, which is not 128-byte aligned — its device \
                 address is written in flits and would truncate to {}",
                p.id,
                p.offset,
                (p.offset / 128) * 128
            ));
        }
        if p.size == 0 {
            fail.push(format!(
                "{} has size 0 in seg{seg} — the device reads a zero length as 2^27 flits (16 GiB)",
                p.id
            ));
        }
        if p.offset + p.size > segment_bytes[seg] {
            fail.push(format!(
                "{} spans [{}, {}) of seg{seg}, which is only {} B",
                p.id,
                p.offset,
                p.offset + p.size,
                segment_bytes[seg]
            ));
        }
    }

    // ── overlap, per segment — SYNTHETICS ONLY ──
    //
    // ⛔ COLORED ACTIVATIONS OVERLAP ON PURPOSE. The slot-coloring pass gives two tensors whose
    // LIFETIMES are disjoint the same address; that is the optimisation, not a defect, and
    // granite's `t447`/`t464`/`t1127` legitimately share one 4096 B slot. Liveness is not a fact
    // this function has, so it does not get to judge them.
    //
    // Synthetics are different: they come from a BUMP allocator, so disjointness IS their
    // invariant and an overlap means the allocator handed the same address out twice — which is
    // exactly what it did, by never advancing `next` for an undeclared name.
    for seg in 0..segment_bytes.len() {
        let mut in_seg: Vec<&bundle::Placement> = places
            .iter()
            .filter(|p| {
                p.segment as usize == seg
                    && p.size > 0
                    && matches!(p.id, bundle::PlaceId::Synth { .. })
            })
            .collect();
        in_seg.sort_by_key(|p| p.offset);
        for w in in_seg.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a.offset + a.size > b.offset {
                fail.push(format!(
                    "{} spans [{}, {}) and {} starts at {} in seg{seg} — they SHARE {} B, so one \
                     tensor's producer writes over the other's",
                    a.id,
                    a.offset,
                    a.offset + a.size,
                    b.id,
                    b.offset,
                    a.offset + a.size - b.offset
                ));
            }
        }
    }

    if !fail.is_empty() {
        let shown = fail.len().min(12);
        panic!(
            "\n⛔ [superdsc-layout] bundle {fp}: {} ADDRESS DEFECT(S). Refusing to bake.\n\n{}\n{}\n             Every number here is a compile-time constant, so this cannot be left for the card to \
             find — it finds them as `0xa35e RAS::PCI::BusFence`, with no host stack and no \
             statement of which transfer was responsible.\n",
            fail.len(),
            fail[..shown]
                .iter()
                .map(|f| format!("  • {f}"))
                .collect::<Vec<_>>()
                .join("\n"),
            if fail.len() > shown {
                format!("  … and {} more\n", fail.len() - shown)
            } else {
                String::new()
            },
        );
    }
}

/// Round `n` up to a 128-byte boundary (Spyre requires tensor start addresses to be
/// multiples of 128 B — `runtime_operation.hpp:365`). This is the ACTUAL primitive the segment `pack`
/// closure advances every tensor base by; `pub` so the SegLayout island's Kani proof
/// (`align128_emitter_equals_model`) can assert the on-card packing rests on the proven round-up.
pub fn align128(n: u64) -> u64 {
    (n + 127) & !127
}

/// Mint a synthetic's operand SPELLING and record its identity with the layout in one step.
///
/// Every `{tensor}_{role}` name goes through here, so `resolve_seg_base` can recover the identity
/// by LOOKUP. Minting and registering are one call because a name minted without being registered
/// is exactly the case the parse used to paper over.
pub(crate) fn syn(layout: Option<&BundleLayout>, id: bundle::PlaceId) -> String {
    let n = id.to_string();
    if let Some(l) = layout {
        l.ids.borrow_mut().insert(n.clone(), id);
    }
    n
}

impl BundleLayout {
    /// The identity behind an emitted operand's spelling, or `None` if nothing minted it.
    pub fn id_of(&self, name: &str) -> Option<bundle::PlaceId> {
        self.ids.borrow().get(name).copied()
    }

    /// Declare a SYNTHETIC intermediate at its FULL shape (`dims`, outermost→innermost),
    /// reserving its true footprint in the Intermediate-segment allocator UP FRONT. The tensor
    /// then OWNS its shape: a later per-head write at offset h·stride cannot land on the next
    /// synthetic (no under-reservation), and `resolve_seg_base` turns any access beyond the
    /// footprint into a `cargo build` Err (the typed-shape contract). Idempotent per name.
    ///
    /// This is the principled replacement for the old bump allocator that sized each synthetic
    /// from its FIRST op's per-op view (ONE head) — which aliased sp/spprod/mxp on-card and
    /// produced the garbage attention. The shape is now part of the tensor, not guessed per-op.
    pub fn synth(&self, id: bundle::PlaceId, dims: &[u32]) {
        self.synth_df(id, dims, Df::Fp16);
    }

    /// [`synth`](Self::synth) for a NON-fp16 intermediate — the packed fp8/int8 (1-byte / 128-stick)
    /// or fp32-merge (4-byte / 32-stick) tensors. The footprint is reserved at the given [`Df`]'s true
    /// width, so a fp8 activation reserves HALF the fp16 bytes (the residency win) — not a name-suffix
    /// guess. `synth(name, dims)` is exactly `synth_df(name, dims, Df::Fp16)`.
    pub fn synth_df(&self, id: bundle::PlaceId, dims: &[u32], df: Df) {
        self.synth_bytes(id, synth_footprint_bytes(dims, df));
    }

    /// ⛔⛔⛔ DECLARE A SYNTH AT THE FOOTPRINT OF THE TENSOR IT SHADOWS, NOT OF ONE CHUNK OF IT.
    ///
    /// A decomposition's intermediate usually has EXACTLY the shape of the node's output —
    /// `silu(gate) -> tmp` then `multiply(tmp, up) -> out`. But a wide op is CHUNKED: the tape splits
    /// granite's 12800-wide MLP intermediate into column blocks, and `lower_*_node` sees only its own
    /// block. Declaring `[rows, cols]` from the node therefore reserves the FIRST chunk's width, and
    /// since declaration is first-one-wins ([`Self::synth_bytes`]) every later chunk writes past it.
    ///
    /// On granite-3.1-8b that is chunk 2 — offset 8192, width 4608 of 12800 — landing 9216 B past a
    /// 16384 B reservation, straight into the next intermediate. The footprint guard turns it into a
    /// build error rather than on-card garbage, and this is the cure: take the width from the OUTPUT
    /// TENSOR's placement, which `compute_bundle_layout` sized from `ir.tensors` before any chunking.
    ///
    /// Falls back to `dims` when the tensor has no placement (a synth OF a synth, and every unit test
    /// that lowers without a layout) — the old behaviour, correct whenever the op is not chunked.
    pub fn synth_like(&self, id: bundle::PlaceId, shadows: u32, dims: &[u32], df: Df) {
        match self.placements.get(&shadows) {
            Some(p) => self.synth_bytes(id, p.size),
            None => self.synth_bytes(id, synth_footprint_bytes(dims, df)),
        }
    }

    /// The byte-exact core of [`Self::synth_df`]: bump-allocate `bytes` for `id`, once.
    fn synth_bytes(&self, id: bundle::PlaceId, bytes: u64) {
        let name = id.to_string();
        let a = self.synth.borrow_mut();
        // The IDENTITY is recorded even when the OFFSET already exists. `resolve_seg_base` bump-
        // allocates an intermediate the first time an op references it, which can happen BEFORE the
        // declaring `synth()` call; the early return below then skipped the id and the placement had
        // no identity to be built from. Declaration order decides the offset, never the identity.
        drop(a);
        self.ids.borrow_mut().insert(name.clone(), id);
        let mut a = self.synth.borrow_mut();
        if a.map.contains_key(&name) {
            return;
        }
        let o = align128(a.next);
        let fp = align128(bytes.max(1));
        a.next = o + fp;
        a.map.insert(name.clone(), o);
        a.sizes.insert(name, fp);
    }

    /// THE arrangement authority. Record that tensor `name` is addressed through `want`; a later op that
    /// addresses it NON-equivalently ([`StickLayout::addr_eq`]) is a build `Err` naming the tensor.
    /// `addr_eq` is true for the same layout, and for a byte-identical RESHAPE (same total elements, both
    /// contiguous — e.g. `[1,2048]` ↔ `[32,64]`), which IS safe: identical bytes. It is FALSE for the M>1
    /// same-shape Dense-vs-stick scramble AND for a different-TOTAL disagreement (`[4,64]` vs `[32,64]`),
    /// because a consumer that reads more elements than the producer wrote reads garbage. That is NOT
    /// silently allowed — it names the tensor at build time so the size mismatch is confronted, not baked.
    pub fn declare_arrangement(
        &self,
        name: &str,
        want: scratchy_subtile::sdsc_abstract::StickLayout,
    ) -> Result<scratchy_subtile::sdsc_abstract::StickLayout, SuperDscError> {
        let mut m = self.arrangements.borrow_mut();
        match m.get(name) {
            Some(existing) if !existing.addr_eq(&want) => Err(SuperDscError(format!(
                "tensor '{name}': device arrangement {want:?} conflicts with the earlier {existing:?} — a \
                 tensor has ONE physical layout, but a producer and consumer address it differently (a \
                 same-shape Dense-vs-stick scramble, OR a different-TOTAL size mismatch where the consumer \
                 reads bytes the producer never wrote). Address it ONE way, or insert an explicit \
                 restickify/replication op. This is the arrangement authority (declare_arrangement)."
            ))),
            Some(existing) => Ok(*existing),
            None => {
                m.insert(name.to_string(), want);
                Ok(want)
            }
        }
    }
}

/// Build the GLOBAL [`BundleLayout`] for one fused SuperDSC bundle (task #55).
///
/// Classification by liveness over `ir.nodes` (already a valid eval order):
/// - a SOURCE tensor (`id < num_sources`) in `weight_ids` → [`SegRole::Weight`]
///   (seg1, packed contiguously, resident — all weights are simultaneously live so
///   no reuse, just dense packing);
/// - a SOURCE not in `weight_ids` → [`SegRole::Activation`] (seg0, per-step input);
/// - `ir.result` → [`SegRole::Logits`] (seg4);
/// - every other tensor (produced by some op AND consumed) → [`SegRole::Intermediate`]
///   (seg3), assigned by a lifetime-aware linear scan that REUSES the byte range of an
///   intermediate whose `last_use` precedes the new tensor's `first_def`.
///
/// All sizes are f16 bytes (`rows*cols*2`), 128 B aligned. A build-time disjointness
/// guard ([`bundle_layout_aliases`]) asserts no two SIMULTANEOUSLY-LIVE tensors in a
/// segment overlap — a silent on-card aliasing write is a `cargo build` Err.
pub fn compute_bundle_layout<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // TRUE when this bundle's query rows are separate requests. The prefix mask's reservation
    // depends on it (a prompt reads one broadcast validity row, a batch reads one per query row),
    // and so does the attention pad's door: a decode width must be a baked ladder rung
    // (`PaddedMq::of_bundle`), which is the `Err` this returns.
    rows_are_requests: bool,
) -> Result<BundleLayout, SuperDscError> {
    // fp8 W8A8 weights (SEN143_FP8: 1-byte / 128-elem stick) are `input[1]` of any arity-3 MatmulTile.
    // Their device footprint is HALF the fp16 weight — this is the unfakeable 1-byte-read proxy: it is
    // what shrinks `seg_bytes[1]` (the weight segment). A dequant-to-f16 transient would NOT shrink it.
    let fp8_weight_tids: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.op, SubOp::MatmulTile { .. }) && n.inputs.len() == 3)
        .map(|n| n.inputs[1].tensor.index() as u32)
        .collect();
    let nbytes = |tid: u32| -> u64 {
        let s = ir.tensors[tid as usize];
        // Reserve the DEVICE footprint: a stick-last tensor pads its innermost stick dim up to a whole
        // stick on-device — fp16 is a 64-elem / 2-byte stick, fp8 is a 128-elem / 1-byte stick (HALF the
        // bytes). Width is DERIVED from the operand's `Df` (rung 1), never hardcoded. (granite lm_head
        // vocab 49159 → 49216 padding stays in-bounds for the on-card write / weight staging.)
        let df = if fp8_weight_tids.contains(&tid) {
            Df::Fp8
        } else {
            Df::Fp16
        };
        s.rows as u64
            * bump_sticks_to_splittable(s.cols.next_multiple_of(df.elems_per_stick())) as u64
            * df.word_length() as u64
    };
    // ── Liveness: first def (output of node i) and last use (input of node j). A
    //    source has no def (def = 0, live from the start); the result has no use
    //    (use = nodes.len(), live to the end). ──
    let n_nodes = ir.nodes.len();
    let mut first_def: std::collections::BTreeMap<u32, usize> = Default::default();
    let mut last_use: std::collections::BTreeMap<u32, usize> = Default::default();
    for (i, node) in ir.nodes.iter().enumerate() {
        let o = node.output.tensor.index() as u32;
        first_def.entry(o).or_insert(i);
        for inp in &node.inputs {
            last_use.insert(inp.tensor.index() as u32, i);
        }
    }

    let mut placements: std::collections::BTreeMap<u32, TensorPlacement> = Default::default();
    let mut seg_bytes = [0u64; 7];
    // Set by the paged-KV placement below, from the SAME `PagedKvPool` that sized the placement, so the
    // stride the runtime shifts by and the stride the addresses were baked with are one value.
    let kv_request_stride_bytes: u64 = 0;

    // ── Pack WEIGHTS (seg1) + ACTIVATIONS (seg0) + LOGITS (seg4): all simultaneously
    //    live within their role (weights resident the whole program; the single
    //    logits row lives to the end), so dense contiguous packing by tensor id. ──
    let pack = |tid: u32,
                role: SegRole,
                seg_bytes: &mut [u64; 7],
                place: &mut std::collections::BTreeMap<u32, TensorPlacement>| {
        let seg = role.segment();
        let off = seg_bytes[seg];
        let sz = nbytes(tid);
        place.insert(
            tid,
            TensorPlacement {
                tid,
                role,
                segment: seg,
                offset: off,
                size: sz,
            },
        );
        seg_bytes[seg] = align128(off + sz);
    };

    // The AttnDecode K/V cache tensors are SOURCES, but they get RE-PLACED into seg2 (SegRole::Kv) at
    // the batched cache size further below (search "Prefix K/V caches → seg2"). If the source loop ALSO
    // packs them into the Activation segment at their source size, that Activation slot is never used
    // (the seg2 placement overrides it) — a DEAD HOLE (80 tensors × [cap,kv_dim]·2 = 335 MB @cap=4096
    // for granite) that the per-step whole-Activation-segment H2D re-ships every decode step. That dead
    // re-upload IS the O(cap) decode preamble cliff (measured: 50 ms @cap=4096 vs 8 ms @cap=256). Skip
    // them here; their only live placement is seg2 (what the worker binds + attention reads).
    let prefix_kv_tids: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            SubOp::AttnDecode { layout: kv, .. } => Some([
                kv.cache_tensor().index() as u32,
                kv.v_cache_tensor().index() as u32,
            ]),
            _ => None,
        })
        .flatten()
        .collect();
    // Sources first (deterministic id order): weights → seg1, activations → seg3 (Activation role).
    for tid in 0..ir.num_sources {
        if prefix_kv_tids.contains(&tid) {
            continue; // re-placed into seg2 (Kv) below — no dead Activation slot / no per-step re-H2D
        }
        let role = if weight_ids.contains(&tid) {
            SegRole::Weight
        } else {
            SegRole::Activation
        };
        pack(tid, role, &mut seg_bytes, &mut placements);
    }
    // ── KSPLIT block WEIGHTS (seg1) ── the lm_head split's B block weights `ksplit_block_tid(b)` are added by
    // the `kernel0` loop (RetileDescriptors) + the worker (bytes), but NOT by the source loop above (they are
    // not manifest sources). Without a placement they get NO HBM address → staged to a default/aliased spot →
    // GARBAGE logits (on-card-observed: KSPLIT output empty/EOS while the bundle loaded fine). Place each in
    // seg1 (Weight) here — same condition as the emit branch + kernel0 (n_dev>16384 lm_head). Size = the
    // staged `[KB, n_dev]` f16 buffer (KB·n_dev·2, the retiled `[n_dev/64,KB,64]` footprint).
    // Logits (seg4) — the graph result (it is also produced by an op, but its role
    // is OUTPUT; classify it before intermediates so it is not pool-coalesced).
    let result = ir.result.index() as u32;
    if result >= ir.num_sources {
        pack(result, SegRole::Logits, &mut seg_bytes, &mut placements);
    }
    // ── ROPE permutation matrix P [hd,hd] (task: in-bundle RoPE) ── If the tape has
    // any RopeRotate/RopeAppend, `lower_rope_node` emits `rot = matmul(x, P)` (the
    // rotate-half as a 64-stick-aligned matmul, avoiding the 32-half sub-stick). P is
    // a FIXED permutation-sign matrix the worker synthesizes + binds (it is NOT a
    // model/safetensors weight, so it gets the reserved id ROPE_P_TID). Place it in
    // seg0 (ACTIVATION, re-bound per step like RMS_SEED/ATTN_SCALE) — NOT seg1 (WEIGHT):
    // a seg1 weight is only H2D'd at PrepareModel, where the worker binds ONLY the
    // manifest's model weights, so a synthetic seg1 P stays ZERO ⇒ rot=matmul(x,0)=0 ⇒
    // RoPE collapses to `x·cos` (rotate-half/sin term DROPPED) ⇒ wrong positional
    // encoding ⇒ wrong content. Same P for every head/position/layer.
    if let Some(hd) = ir.nodes.iter().find_map(|n| match &n.op {
        SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
            Some(head_dim.get() as u64)
        }
        _ => None,
    }) {
        let seg = SegRole::Activation.segment();
        let off = seg_bytes[seg];
        let sz = hd * hd * 2; // [hd,hd] fp16
        placements.insert(
            ROPE_P_TID,
            TensorPlacement {
                tid: ROPE_P_TID,
                role: SegRole::Activation,
                segment: seg,
                offset: off,
                size: sz,
            },
        );
        seg_bytes[seg] = align128(off + sz);
    }

    // ── DEAD seg0 CONSTS REMOVED (2026-07-28, TTFT) ── This block used to place, for every mq>1
    // (batched-prefill) bundle: the head-major one-hot selectors Sel_q [nqh·nqh·hd,hd],
    // Sel_kv [nkvh·nkvh·hd,hd], SelT [nqh·hd,nqh·hd], and the matmul-by-ones row-sum weight
    // ONES_REDUCE [max(hidden, max fp8-K), stick]. NOTHING reads any of them any more: the unified
    // emitter (ir::bridge::tiled_op_sdsc_op::attn) is head-contiguous and needs no selector matmul,
    // and rmsnorm/fp8-amax both use NATIVE reduces (assemble_reduce_seeded "mean"/"max"), not the
    // matmul-by-ones substitute. Verified: no op anywhere names these tids; only the const decls,
    // this placement, and doc comments referenced them.
    // They were not free. The shim re-uploads the whole dirty seg0 every forward, so each prefill
    // chunk was paying ~18.3 MB of H2D for never-written, never-read bytes:
    //   Sel_q 8.4 MB + SelT 8.4 MB + Sel_kv 0.5 MB + ONES_REDUCE 1.0 MB.
    // With the op count already cut 5x (RoPE row-batching), TTFT was dominated by a ~218 ms FIXED
    // cost that op-count work cannot touch; this dead H2D is the bulk of it.
    // The ONES_REDUCE placement also drove the worker's `uses_ones_reduce` binding (read back out of
    // bundle_layout.json), so unplacing it also stops the worker from staging the all-ones buffer.
    // Its qk-norm build guard (cols==hidden assert) went with it: that guard existed only to protect
    // the matmul-by-ones reduce's reuse of RMS_INVCOLS=1/hidden as a 1/cols scale, and the native
    // reduce it was replaced by folds 1/N itself, so the constraint no longer applies.
    // ── KV-cache-write / GQA-replicate MATMUL-BY-IDENTITY weight (#3, SCRATCHY_SUPERDSC_KV_MATMUL) ──
    // A [hd,hd] identity fp16 seg0 ACTIVATION, worker-bound flat: for hd==64 the retile is
    // row-major identity (single stick) ⇒ NO RetileDescriptor (like the all-ones). The mq>1 cachewr
    // becomes matmul(kh[mqu,hd], I[hd,hd]) = kh, replacing the mqu·nqh·2 SlotSolo copies.
    //
    // BUG #1 (found via a real pod trace, 20cf993a): `assemble_attn`'s GQA new_k/new_v replication
    // (ir::bridge::tiled_op_sdsc_op::attn.rs, `attn_krep{h}`/`attn_vrep{h}`) references this SAME
    // `ident` tensor UNCONDITIONALLY — every model with attention, every mq, no env-var check at
    // all (the on-card multi-row plain copy is proven broken, so this IS the copy mechanism, not
    // an optional #3 optimization for it). But this placement was gated ONLY on the opt-in
    // SCRATCHY_SUPERDSC_KV_MATMUL env var — unset in a normal build, IDENTITY_TID was never placed
    // at all, so the always-referenced `ident` fell through to an unrelated synthetic offset with
    // whatever garbage happened to be there.
    //
    // BUG #2 (found via a real pod HARD-FAIL, 2026-07-27): the fix for BUG #1 was placed INSIDE the
    // `if let Some((nqh,nkvh,hd)) = ir.nodes.find_map(... n.output.region.rows.len > 1)` block right
    // above — an mq>1-ONLY guard (the batched-prefill selector/ones-reduce consts). Decode's tape only
    // ever has mq=1 AttnDecode nodes, so that ENTIRE enclosing block — including this placement — never
    // ran for decode's own bundle at all. Confirmed by the worker's own hard-fail: decode's baked
    // bundle_layout.json genuinely never had IDENTITY_TID, exactly as this predicts. Moved OUTSIDE that
    // mq>1 guard so it runs for ANY AttnDecode node regardless of mq, matching the consumer's real,
    // unconditional need (assemble_attn_head references `ident` at every mq, decode included).
    if let Some(hd) = ir.nodes.iter().find_map(|n| match &n.op {
        SubOp::AttnDecode { geom, .. } => Some(geom.hd().get() as u64),
        _ => None,
    }) {
        let seg = SegRole::Activation.segment();
        let off = seg_bytes[seg];
        let sz = hd * hd * 2; // [hd,hd] fp16 identity
        placements.insert(
            IDENTITY_TID,
            TensorPlacement {
                tid: IDENTITY_TID,
                role: SegRole::Activation,
                segment: seg,
                offset: off,
                size: sz,
            },
        );
        seg_bytes[seg] = align128(off + sz);

        // ATTN_ZERO (BUG #3, same class as #1/#2 above, found by re-auditing this exact area 2026-07-28):
        // `new_k`/`new_v` (AttnDecode's inputs[3]/[4]) are declared "[mq_pad, nkvh·hd] (worker zero-pads
        // rows)" — but nothing ever actually zeroed rows [mq..mq_pad) at ANY mq. This tid's placement was
        // ALSO stuck inside the old mq>1-only selector guard (moved out alongside IDENTITY_TID above), and
        // even there it was placed but never CONSUMED anywhere in this file — dead. `new_k`/`new_v` are a
        // lifetime-reused seg3 intermediate (compute_bundle_layout's linear-scan reuse), never explicitly
        // re-zeroed between steps/layers, so the padding rows can alias whatever UNRELATED tensor last
        // lived at that byte range — potentially large magnitude, not bounded "stale K-vector" data. The
        // causal mask (mask_neg, a moderate ~-32752 fp16 constant) only reliably neutralizes BOUNDED
        // garbage; it does not guarantee correctness against arbitrary aliased memory. Placed here
        // (unconditional, any mq with attention) so `lower_attn_node` can emit a real zero-copy into the
        // padding rows before GQA-replicate ever reads them, instead of relying on masking alone.
        // SIZE IS `mq_pad`, NOT ONE STICK (2026-07-29). The worker binds this as `[mq_pad, hd]` zeros
        // (`vec![0.0f32; mq_pad*hd]`, narrowed to 2-byte f16), and `mq_pad = mq.div_ceil(64)*64` — so a
        // hardcoded one-stick reservation matched the bind ONLY while every chunk fit in 64 query rows.
        // At mq_pad=128 the bind is 16384 B into an 8192 B placement and spills 8192 B into the seg3
        // tensors that follow. Nothing catches it: the shim's refill only checks the bind against the
        // whole SEGMENT size, never against `p.size` (see the `src.bytes.size() > p.size` guard added
        // alongside this in sdsc_shim.cpp).
        //
        // WHICH NEIGHBOURS ACTUALLY STAY CORRUPTED is decided by bind ORDER, because the shim refills
        // `std::map<std::string, HostBuf> bound` in LEXICOGRAPHIC tid order. Reserved tids are
        // `u32::MAX - k`, so a LARGER k sorts EARLIER and is overwritten by this tensor's spill with no
        // chance to be rewritten. Measured at mq=127 (spill `[1845248, 1853440)`):
        //   MAX-11 RMS_INVCOLS, MAX-13 FP8_POS448, MAX-14 FP8_NEG448, MAX-15 FP8_INV448 → sort BEFORE
        //     ATTN_ZERO (MAX-10) ⇒ left ZEROED.
        //   MAX-6 RMS_HALF and MAX-3 pmask → sort AFTER ⇒ rebound, harmless.
        // So the fp8 activation-quant clamp constants are the real victims. `inv448 = 0` makes
        // `ascale = amaxfl · inv448 = 0`, and `fq_dqa = mul(raw, col(ascale))` is then exactly 0, so all
        // seven fp8 projections emit zeros in all 40 layers: prefill writes an ALL-ZERO KV cache while
        // the residual carries the embedding through untouched. Decode (its own seg3 is not aliased,
        // and its mq_pad is 64, so its own bind fits) stays numerically healthy but attends a zero
        // prefix — leaving it conditioned on the last prompt token alone. On hardware that read as
        // FLUENT, on-grammar output about entirely the wrong subject (asked about the history of
        // France, answered about fractions of an inch).
        // ⚠ An earlier version of this comment blamed pmask being zeroed to "valid". That is WRONG —
        // pmask is rebound after this tensor, and the only pmask bytes any op reads are the 512 B its
        // own bind restores. Recorded because that false story invites a "fix" that merely reorders
        // binds, which would leave the over-bind in place.
        //
        // Derive the extent from the WIDEST AttnDecode in the tape. `div_ceil` keeps it at exactly one
        // stick for every mq<=64 (decode's mq=1 included), so previously-baked bundles are unchanged.
        let mqp_z = ir
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                SubOp::AttnDecode { .. } => Some(n.output.region.rows.len as u64),
                _ => None,
            })
            .max()
            .unwrap_or(1)
            .div_ceil(Fp16::ELEMS_PER_STICK as u64)
            * Fp16::ELEMS_PER_STICK as u64;
        let off = seg_bytes[seg];
        let sz = mqp_z * hd * 2;
        placements.insert(
            ATTN_ZERO_TID,
            TensorPlacement {
                tid: ATTN_ZERO_TID,
                role: SegRole::Activation,
                segment: seg,
                offset: off,
                size: sz,
            },
        );
        seg_bytes[seg] = align128(off + sz);
    }
    // (No AttnDecode node at all means no consumer ever references `ident`/ATTN_ZERO, so
    // SCRATCHY_SUPERDSC_KV_MATMUL has nothing to do for such a model — already a no-op, not a case that
    // needs handling here.)

    // ── RMSNorm Newton const [1,stick] ── If the tape has any RmsNorm, place `RMS_HALF_TID` in seg0
    // (ACTIVATION, re-bound per step like ATTN_SCALE — a seg1 weight would only be H2D'd at prepare).
    // The worker binds 0.5 (EXACT in fp16); the rmsnorm derives 1.0/1.5/−1.0 on-card from it. The old
    // RMS_SEED_TID seed-floor placement was REMOVED: the amax-normalized reciprocal seed never
    // underflows, so no floor const is needed. See [`RMS_HALF_TID`].
    if ir
        .nodes
        .iter()
        .any(|n| matches!(n.op, SubOp::RmsNorm { .. }))
    {
        let a = SegRole::Activation.segment();
        // Bind as a [1, stick] row (64 copies) so every derived-constant op is a uniform [1, stick]
        // elementwise op (no [1,1]→stick broadcast bookkeeping).
        let sz = Fp16::ELEMS_PER_STICK as u64 * 2;
        let hoff = seg_bytes[a];
        placements.insert(
            RMS_HALF_TID,
            TensorPlacement {
                tid: RMS_HALF_TID,
                role: SegRole::Activation,
                segment: a,
                offset: hoff,
                size: sz,
            },
        );
        seg_bytes[a] = align128(hoff + sz);
        // RMS_INVCOLS `[1,stick]` = 1/cols (worker-bound), for the mq>1 sum-based amax pre-scale + un-scale.
        let ioff = seg_bytes[a];
        placements.insert(
            RMS_INVCOLS_TID,
            TensorPlacement {
                tid: RMS_INVCOLS_TID,
                role: SegRole::Activation,
                segment: a,
                offset: ioff,
                size: sz,
            },
        );
        seg_bytes[a] = align128(ioff + sz);
    }

    // ── fp8 W8A8 activation-quant consts [1,stick] (seg0, worker-bound like RMS_HALF) ── E4M3 clamp
    // bounds ±448 + 1/448 for `qfp8ch` (per-token amax → a_scale). Placed iff the tape has an fp8
    // (arity-3) MatmulTile. Unplaced (and thus value-0) was the "clamp consts default 0 → wrong quant"
    // gap; placing them here + binding in the worker gives the real E4M3 bounds.
    if ir
        .nodes
        .iter()
        .any(|n| matches!(n.op, SubOp::MatmulTile { .. }) && n.inputs.len() == 3)
    {
        let a = SegRole::Activation.segment();
        let sz = Fp16::ELEMS_PER_STICK as u64 * 2; // [1, stick] fp16
        for tid in [FP8_POS448_TID, FP8_NEG448_TID, FP8_INV448_TID] {
            let off = seg_bytes[a];
            placements.insert(
                tid,
                TensorPlacement {
                    tid,
                    role: SegRole::Activation,
                    segment: a,
                    offset: off,
                    size: sz,
                },
            );
            seg_bytes[a] = align128(off + sz);
        }
    }

    // ── INTERMEDIATES (seg3): lifetime-aware linear scan with byte-range reuse. ──
    // Collect every produced tensor that is NOT a source and NOT the logits, sorted
    // by first_def (def order == node order). Maintain `live` = currently-assigned
    // (offset, size, expiry=last_use) and `free` = reclaimed (offset, size) holes.
    // AttnDecode's new_k/new_v (inputs[3]/[4]) are RE-placed at the PADDED [mq_pad, nkvh·hd] size below,
    // so they must NOT also be placed by this general colored pool: a double-placement reserves a wasted
    // colored slot AND records a stale (unpadded) placement in the map that the later re-placement
    // overwrites — the coloring's live-set then reflects the wrong offset/size. Exclude them here; the
    // AttnDecode loop is their single placement.
    let replaced_kv: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter_map(|n| match &n.op {
            SubOp::AttnDecode { .. } if n.inputs.len() >= 5 => Some([
                n.inputs[3].tensor.index() as u32,
                n.inputs[4].tensor.index() as u32,
            ]),
            _ => None,
        })
        .flatten()
        .collect();
    let mut inter: Vec<u32> = first_def
        .keys()
        .copied()
        .filter(|&t| t >= ir.num_sources && t != result && !replaced_kv.contains(&t))
        .collect();
    inter.sort_by_key(|t| (first_def[t], *t));
    // ── SEGMENT COLORING (the multi-op stitching fix): dxp's ModuleStitcher connects
    //    producer→consumer by SEGMENT (the dxp ref test_softmax_1core puts each
    //    inter-op tensor in its OWN segment: sub→seg2, exp reads seg2). scratchy used
    //    to cram ALL intermediates into seg3 (offset-distinguished) — but the stitcher
    //    can't tell t337 from t338 (both seg3) ⇒ the PT matmul's output is mis-wired
    //    (single-op MatMul_49 works; multi-op all-seg3 orphans). FIX: spread
    //    intermediates across the free segments {3,5,6} (lifetime-aware reuse) so no
    //    two SIMULTANEOUSLY-LIVE intermediates share a segment. ≤3 live fits; on
    //    overflow we fall back to seg3 offset-packing (the old behavior) for that tid.
    let inter_segs = [SegRole::Intermediate.segment(), 5usize, 6usize];
    // Per-segment: the live tensor's expiry (usize::MAX = free) + the running offset.
    let mut seg_live_exp = [0usize; 3]; // 0 = free (no live tensor)
    for &tid in &inter {
        let def = first_def[&tid];
        let exp = *last_use.get(&tid).unwrap_or(&n_nodes);
        let sz = align128(nbytes(tid));
        // Free any segment whose live tensor expired before this def.
        for exp in &mut seg_live_exp {
            if *exp != 0 && *exp < def {
                *exp = 0;
            }
        }
        // Pick the first free segment in {3,5,6}; else overflow into seg3 (offset-packed).
        let pick = (0..3).find(|&s| seg_live_exp[s] == 0);
        let seg = match pick {
            Some(s) => {
                seg_live_exp[s] = exp;
                inter_segs[s]
            }
            None => inter_segs[0], // overflow: reuse seg3 (offset-packed, may alias)
        };
        let off = seg_bytes[seg];
        seg_bytes[seg] = off + sz;
        placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Intermediate,
                segment: seg,
                offset: off,
                size: nbytes(tid),
            },
        );
    }

    // ── Attention KV cache (seg2) + scale const (seg1) for in-bundle attention ──
    // The IR's AttnDecode K/V cache tensors are reused as the per-layer IDENTITY but
    // RE-placed (override) at the BATCHED size `[nqh,hd,cap]·2` (the worker fills them
    // transposed-K + GQA-replicated V each step — they are read-only INPUTS in-bundle,
    // never written by a SuperDSC op). One shared `scale` const `[1,1]` (worker
    // synthesizes `1/sqrt(hd)`). Placed LAST so it overrides any earlier source/
    // intermediate classification of the cache ids.
    let mut scale_placed = false;
    for node in &ir.nodes {
        if let SubOp::AttnDecode {
            geom, layout: kv, ..
        } = &node.op
        {
            let nqh = geom.nqh().get() as u64;
            let nkvh = geom.nkvh().get() as u64;
            let hd = geom.hd().get() as u64;
            let k_id = kv.cache_tensor().index() as u32;
            let v_id = kv.v_cache_tensor().index() as u32;
            let cap = ir.tensors[k_id as usize].rows as u64; // prefix cache capacity (rows)
            let mq32 = node.output.region.rows.len; // chunk query rows (1=decode, >1=prefill)
            let mq = mq32 as u64;
            // THE SAME DOOR AS THE EMIT — `PaddedMq::of_bundle`, the one parse boundary from a
            // bundle's runtime width to the pad law, so the placements and the ops they hold cannot
            // be sized by different pads (a decode width must be a baked ladder rung here exactly as
            // it must be in `lower_attn_node`). The placements below spend it in TWO roles and each
            // names its own: the staging tensors' ROW extent and the causal mask's SCORE width.
            let mq_pad = scratchy_subtile::sdsc_abstract::PaddedMq::of_bundle(
                mq32,
                rows_are_requests,
            )
            .ok_or_else(|| {
                SuperDscError(format!(
                    "AttnDecode t{}: a decode bundle at {mq32} query rows — not a width the \
                         decode ladder bakes (1, or PagedKvPool::BATCH_RUNGS), so its staging \
                         tensors and causal mask have no placeable pad.",
                    node.output.tensor.index() as u32
                ))
            })?
            .pad();
            // ── PAGED K/V POOL (`sdsc_abstract::PagedKvPool`) ──
            // This layer's slice of ONE page: Kᵀ, then V, then natural K. NOTHING here is sized by
            // the pool — the placement covers a single page+layer and the runtime adds
            // `layer + physical page` per launch, which is what takes the servable context out of
            // the baked program. All three planes sit in the KV segment because the re-roll executor
            // advances only that segment (and the weights) per layer.
            // A PAGE WIDER THAN THE SWEEP IS A WIN, NOT A REGRESSION — the reverse of what this
            // guard used to assert. It read: "a fold covers one page per launch, so a page wider than
            // the pre-paged capacity would sweep MORE KV for the same context — a regression, not a
            // feature." That is an argument about the sweep, and the sweep is not what a pass costs.
            //
            // Measured, eight requests, one page, varying ONLY the swept width through the ladder's
            // own rungs: 64 slots 39.5 ms, 128 slots 39.0 ms, 256 slots 42.0 ms. Four times the sweep
            // costs six percent. A pass is likewise flat in the ROWS it computes (256 rows to 32:
            // no measurable change). A fold pass is a fixed cost.
            //
            // What it is not flat in is the NUMBER of passes, and `reps = pages * requests`, so the
            // page width sets the pass count: at 576 tokens a 256-slot page is three passes per
            // request, and that is 137 ms of a bs=8 step against 39 ms at one page. Making the page
            // wider than the sweep is how the pass count comes down.
            //
            // The real ceiling is MEMORY: a page is `nkvh * hd * slots * 2 * layers`, so the pool has
            // to be sized for the batch width actually run. That is the constraint a 4-bit KV cache
            // lifts, by making a page denser instead of bigger.
            let pool =
                scratchy_subtile::sdsc_abstract::PagedKvPool::new(nkvh as usize, hd as usize);
            let _ = cap;
            let seg = SegRole::Kv.segment();
            let plane_bytes = pool.plane_stride() as u64 * 2;
            // THE REQUEST STRIDE THIS PLACEMENT WAS SIZED FOR. `plane_stride` already carries the
            // `ROWS` factor, so the pool grew ×ROWS per page here and nowhere else; publishing the
            // stride from the same `pool` is what stops the runtime's shift from disagreeing with the
            // addresses the ops baked.
            // ⛔ NO REQUEST STRIDE TO PUBLISH. A page holds slots, not requests, so there is no
            // "bytes between two requests' KV" for the runtime to shift by — a request is reached by its
            // PAGE, through the host's block table. Left at 0, which is what an unbatched bundle always
            // published and what `LaunchPages` reads as "no request dimension".
            let layer_base = seg_bytes[seg];
            // Kᵀ at 0, V at `v_plane_base`, natural K at `knat_plane_base` — the offsets
            // `PagedKvPool` bakes into the ops' plane-relative bases.
            for (tid, plane_off) in [
                // natural K, V, transposed K — the pre-paged cache's own order.
                (k_id, pool.knat_plane_base() as u64 * 2),
                (v_id, pool.v_plane_base() as u64 * 2),
                (kct_resident_tid(k_id), pool.kt_plane_base() as u64 * 2),
            ] {
                placements.insert(
                    tid,
                    TensorPlacement {
                        tid,
                        role: SegRole::Kv,
                        segment: seg,
                        offset: layer_base + plane_off,
                        size: plane_bytes,
                    },
                );
            }
            // The per-layer stride the executor advances by MUST be exactly the three planes, with
            // no alignment padding creeping in between layers (hd and page_slots are 64-multiples,
            // so a plane is already 128-aligned — assert rather than trust).
            let layer_stride_bytes = pool.layer_stride() as u64 * 2;
            debug_assert_eq!(
                align128(layer_base + layer_stride_bytes),
                layer_base + layer_stride_bytes,
                "paged KV layer slice must be 128-aligned by construction"
            );
            seg_bytes[seg] = layer_base + layer_stride_bytes;
            let _ = nqh;
            // new_k/new_v (inputs[3]/[4]) are RE-placed at the PADDED size `[mq_pad,
            // nkvh·hd]·2` (fresh seg3 high-water) so the in-bundle transpose's
            // `[mq_pad,…]` read never aliases the next tensor (pad rows are masked).
            if node.inputs.len() >= 5 {
                let seg3i = SegRole::Intermediate.segment();
                for in_idx in [3usize, 4usize] {
                    let nid = node.inputs[in_idx].tensor.index() as u32;
                    let cols = ir.tensors[nid as usize].cols as u64; // nkvh·hd
                    let _ = nkvh;
                    let pbytes = mq_pad.rows().row_axis_extent() as u64 * cols * 2;
                    let off = seg_bytes[seg3i];
                    placements.insert(
                        nid,
                        TensorPlacement {
                            tid: nid,
                            role: SegRole::Intermediate,
                            segment: seg3i,
                            offset: off,
                            size: pbytes,
                        },
                    );
                    seg_bytes[seg3i] = align128(off + pbytes);
                }
            }
            if !scale_placed {
                // Attention MASKS → seg0 (activation, re-bound per step), worker-TILED to [nqh·mq, *]:
                // pmask=[nqh·mq, cap] (prefix validity), cmask=[nqh·mq, mq_pad] (causal triu). Shared across
                // layers (same shape every layer). The attention SOFTMAX SCALE is NOT placed here: it flows
                // through `scalarmul_scales` (config `attention_multiplier`, via `AttnDecode.scale`), placed
                // below with the other config scales and bound by the worker's scale loop — the emitter
                // NEVER recomputes 1/sqrt(hd), which would ignore the model's real attention_multiplier.
                //
                // NOTE (2026-07-28): pmask is OVER-RESERVED. Prefix validity is head- and
                // query-row-independent, the emitter reads it via In::mb_at (one-row mb-broadcast), and
                // BOTH workers now stage exactly one `[cap]` row — so `cap*2` would suffice and
                // [nqh·mq, cap] is ~507 KB of dead seg0 re-H2D'd per prefill forward. Deliberately NOT
                // shrunk here: pmask is placed for EVERY attention bundle, so resizing it shifts the
                // seg0 offsets of the DECODE bundle too (verified: all decode hashes move). Decode was
                // only just made coherent, and 507 KB is 2.7% of the ~19 MB of dead seg0 this pass
                // removes, so it is not worth perturbing a known-good bundle. Revisit once prefill TTFT
                // has settled and decode can be re-validated in the same run.
                // PREFIX-VALIDITY MASK stays in the ACTIVATION segment, ONE PAGE wide, describing
                // the TAIL page — the only page that is partially valid. Its own segment would let a
                // per-page shift select each page's row, but a bound tensor marks its WHOLE segment
                // for re-upload every step, and the only spare segments hold ~100-150 MB of
                // intermediates: that cost 1.8 ms per token. Full pages need no mask at all (see the
                // zero-masked fold variant), so no shift is needed and this can stay small.
                // ONE ROW PER PAGE, in the ACTIVATION segment: the fold re-launch walks along it,
                // so page `i`'s validity sits `i * PAGE_SLOTS` elements in.
                //
                // It lives here, and not in a segment of its own, because BINDING a tensor marks its
                // WHOLE segment for re-upload every step. The spare segments carry a couple of
                // hundred intermediates (~1.6 MB in the decode bundle), which at the preamble's
                // measured ~2.5 GB/s is ~0.64 ms per token of pure upload — most of the paged decode
                // regression. The activation segment is 0.04 MB and is uploaded every step anyway,
                // so the mask rides along for free.
                //
                // Shifting it per page is safe HERE and nowhere else: dumped from the baked bundle,
                // a fold group addresses exactly two things — the KV segment, and the mask. It
                // touches nothing else in the activation segment, so the shift moves nothing it reads.
                let mseg = SegRole::Activation.segment();
                let pmoff = seg_bytes[mseg];
                // ONE validity row over the max context, or one PER QUERY ROW when the rows are
                // separate requests and each carries its own history. The multiplier is the op row
                // count the attention sweeps (`nqh*mq`), which is what the per-row read addresses.
                // It stays 1 for a prompt chunk — at prefill's mq=96 the widened form would be
                // ~50 MB of mask re-uploaded every forward, and a prompt does not need it.
                // ONE BLOCK PER FOLD PASS when the rows are requests. A pass reads ONE request's
                // page, so every other row must be masked off for it — the mask is `[pass][row][page]`
                // and the runtime steps it by a whole `rows * page` block. A prompt keeps one
                // broadcast row per page, which is all its rows can need.
                let (pm_rows, pm_blocks) = if rows_are_requests {
                    (nqh * mq, MAX_FOLD_PASSES)
                } else {
                    (1, MAX_PAGES_PER_REQUEST)
                };
                let pmbytes = pm_rows
                    * pm_blocks
                    * scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u64
                    * 2;
                placements.insert(
                    ATTN_MASK_TID,
                    TensorPlacement {
                        tid: ATTN_MASK_TID,
                        role: SegRole::Activation,
                        segment: mseg,
                        offset: pmoff,
                        size: pmbytes,
                    },
                );
                seg_bytes[mseg] = align128(pmoff + pmbytes);
                let a = SegRole::Activation.segment();
                let cmoff = seg_bytes[a];
                // The mask is `[nqh·mq, mq_pad]`: `mq_pad` is its COLUMN count, the score width —
                // the row extent answers here only through the new-block identity.
                let cmbytes = nqh * mq * mq_pad.cols().score_axis_extent() as u64 * 2;
                placements.insert(
                    ATTN_CAUSAL_TID,
                    TensorPlacement {
                        tid: ATTN_CAUSAL_TID,
                        role: SegRole::Activation,
                        segment: a,
                        offset: cmoff,
                        size: cmbytes,
                    },
                );
                seg_bytes[a] = align128(cmoff + cmbytes);
                // DIAGNOSTIC probe (mq>1 only): persistent seg0 buffer for layer-0's pre-selector new_v
                // [mq_pad, nkvh·hd]. lower_attn_node copies layer-0 new_v here; the worker reads it to split
                // the structural inf (matmul vs selector). Never reused ⇒ survives to post-prefill readback.
                if mq > 1 {
                    let npoff = seg_bytes[a];
                    let npbytes = mq_pad.rows().row_axis_extent() as u64 * nkvh * hd * 2;
                    placements.insert(
                        NEW_V_PROBE_TID,
                        TensorPlacement {
                            tid: NEW_V_PROBE_TID,
                            role: SegRole::Activation,
                            segment: a,
                            offset: npoff,
                            size: npbytes,
                        },
                    );
                    seg_bytes[a] = align128(npoff + npbytes);
                }
                scale_placed = true;
            }
        }
    }

    // ⛔ BUILD GUARD (guard-every-crash-at-build-time, structural class): every RESERVED
    // synthetic constant (ROPE_P / ATTN_SCALE / ATTN_MASK / ATTN_CAUSAL / RMS_SEED /
    // RMS_HALF) is bound by the worker PER STEP via `acts`, so it MUST be placed in seg0
    // (ACTIVATION). A reserved tid placed in seg1 (WEIGHT) is only H2D'd at PrepareModel —
    // which binds ONLY the manifest's model weights — so it would stay ZERO on-card,
    // producing silent wrong numerics with NO crash to trace (this is exactly the RoPE-P
    // bug: P in seg1 ⇒ rot=matmul(x,0)=0 ⇒ RoPE collapsed to x·cos). A misplacement is a
    // pure-data emitter mistake; turn it into a `cargo build` panic instead of garbage out.
    {
        let act_seg = SegRole::Activation.segment();
        // NOTE: ATTN_SCALE_TID is intentionally absent — the attention scale is no longer a reserved
        // per-step const; it flows through `scalarmul_scales` (config attention_multiplier) whose
        // placements are proven seg0 by the scale loop above.
        for &rtid in &[
            ROPE_P_TID,
            ATTN_MASK_TID,
            ATTN_CAUSAL_TID,
            RMS_HALF_TID,
            RMS_INVCOLS_TID,
            ONES_REDUCE_TID,
        ] {
            if let Some(pl) = placements.get(&rtid) {
                assert!(
                    pl.segment == act_seg && matches!(pl.role, SegRole::Activation),
                    "reserved synthetic tid t{rtid} placed in segment {} (role {:?}) — it MUST be \
                     seg{act_seg} ACTIVATION: the worker binds it PER STEP via `acts`, so a \
                     WEIGHT-segment reserved tid is never bound (PrepareModel binds only manifest \
                     weights) ⇒ stays ZERO ⇒ silent wrong numerics",
                    pl.segment,
                    pl.role,
                );
            }
        }
    }

    // ── granite ScalarMul scale constants ── collect the DISTINCT scale values (embedding / residual /
    //    attention / logits multipliers) and place a `[1,1]` worker-bound const per scale in seg0
    //    (ACTIVATION, exactly like ATTN_SCALE). `lower_scalarmul_node` reads the index here → the const TID
    //    the pointwise `mul` multiplies by; the worker binds each `t{tid}=[scale]`. NO weight-fold, NO
    //    host-route — a real on-device pointwise multiply (the ATTN_SCALE mechanism).
    let mut scalarmul_scales: Vec<f32> = Vec::new();
    let push_scale = |scale: f32, scalarmul_scales: &mut Vec<f32>| {
        if !scalarmul_scales
            .iter()
            .any(|s| s.to_bits() == scale.to_bits())
        {
            scalarmul_scales.push(scale);
        }
    };
    for node in &ir.nodes {
        // EVERY config-derived on-device scalar flows through this ONE registry → a `[1,1]` worker-bound
        // const: the muP ScalarMul multipliers (embedding/residual/logits) AND the attention softmax scale
        // (`AttnDecode.scale` == config `attention_multiplier`, set by `attention_scale_for`). NO recompute
        // (the worker must never invent `1/sqrt(hd)` — that ignored the model's real attention_multiplier).
        match &node.op {
            SubOp::ScalarMul { scale } => push_scale(*scale, &mut scalarmul_scales),
            // torch-spyre `spyre__sdpa_overrideable`: scaling_factor = sqrt(scale), applied to BOTH q and K
            // (`query * scaling_factor`, `key * scaling_factor`). Register √scale for the prefill split; the
            // un-split `scale` stays for the decode qs.
            SubOp::AttnDecode { scale, .. } => {
                push_scale(*scale, &mut scalarmul_scales);
                push_scale(scale.sqrt(), &mut scalarmul_scales);
            }
            // RMSNorm epsilon (config `rms_norm_eps`) flows through the SAME registry — a `[1,1]`
            // worker-bound const the rmsnorm adds to the mean-of-squares (config value, not dropped).
            SubOp::RmsNorm { eps, .. } => push_scale(*eps, &mut scalarmul_scales),
            _ => {}
        }
    }
    for i in 0..scalarmul_scales.len() {
        let a = SegRole::Activation.segment();
        let off = seg_bytes[a];
        let tid = scalarmul_scale_tid(i);
        placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: a,
                offset: off,
                size: 2,
            },
        );
        seg_bytes[a] = align128(off + 2);
    }

    // Synthetic intermediates (assigned lazily during lowering) start ABOVE the
    // colored intermediates in seg3, so they never overlap a real intermediate.
    let synth = std::cell::RefCell::new(SynthAlloc {
        next: seg_bytes[SegRole::Intermediate.segment()],
        map: std::collections::BTreeMap::new(),
        sizes: std::collections::BTreeMap::new(),
    });
    // Every placed tensor's spelling, registered at the one site that knows the whole set.
    let ids = std::cell::RefCell::new(
        placements
            .keys()
            .map(|&tid| (crate::wiring::act_name(tid), bundle::PlaceId::Act(tid)))
            .collect::<std::collections::BTreeMap<_, _>>(),
    );
    Ok(BundleLayout {
        placements,
        ids,
        segment_bytes: seg_bytes,
        kernel_weights: std::collections::BTreeMap::new(),
        scalarmul_scales,
        synth,
        arrangements: std::cell::RefCell::new(std::collections::BTreeMap::new()),
        kv_request_stride_bytes,
    })
}

/// The deterministic synthetic source value (MUST match the worker self-test in
/// `spyre_worker.rs::superdsc_selftest`). `id` = source tid, `j` = element index.
#[inline]
pub fn dbg_synth_val(id: usize, j: usize) -> f32 {
    (((id * 131 + j * 7) % 197) as f32 / 197.0 - 0.5) * 0.1
}

/// DEBUG-ONLY numeric bisection oracle (codegen gates on `SCRATCHY_SUPERDSC_DBG`).
/// Runs `eval_dag` on the SAME deterministic synthetic sources the worker
/// self-test will bind, and writes, into `<body_bundle_dir>/dbg/`:
///   • `golden/t{tid}.bin` — eval_dag's f32 value for every produced tid,
///   • `order.json`        — `[{i,tid,op,rows,cols}]` in node order (so the
///                            self-test reports the FIRST op that diverges),
///   • `source_shapes.json`— `[{tid,rows,cols}]` for every source (the worker
///                            regenerates the synthetic value via `dbg_synth_val`),
///   • `weight_tids.json`  — the weight tids (the worker routes these into a
///                            synthetic SuperDscSession; the rest are activations).
/// Pre-attention tids (rmsnorm/qkv/rope) depend only on hidden+weights, so the
/// comparison needs NO mask/cache alignment — the bug localizes to one op.
pub fn write_eval_golden<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    dir: &std::path::Path,
) -> std::io::Result<()> {
    let dbg = dir.join("dbg");
    std::fs::create_dir_all(dbg.join("golden"))?;
    let nsrc = ir.num_sources as usize;
    let src_vals: Vec<Vec<f32>> = (0..nsrc)
        .map(|id| {
            let t = ir.tensors[id];
            let n = (t.rows as usize) * (t.cols as usize);
            (0..n).map(|j| dbg_synth_val(id, j)).collect()
        })
        .collect();
    let src_refs: Vec<&[f32]> = src_vals.iter().map(|v| v.as_slice()).collect();
    let bufs = scratchy_subtile::subtile_ir::eval_dag(ir, &src_refs);
    let mut order = String::from("[");
    let mut seen = std::collections::HashSet::new();
    for (ni, node) in ir.nodes.iter().enumerate() {
        let tid = node.output.tensor.index() as u32 as usize;
        if ni > 0 {
            order.push(',');
        }
        let oplabel: String = format!("{:?}", node.op).chars().take(48).collect();
        let ins: Vec<String> = node
            .inputs
            .iter()
            .map(|i| i.tensor.index().to_string())
            .collect();
        order.push_str(&format!(
            "{{\"i\":{ni},\"tid\":{tid},\"op\":{:?},\"rows\":{},\"cols\":{},\"ins\":[{}]}}",
            oplabel,
            ir.tensors[tid].rows,
            ir.tensors[tid].cols,
            ins.join(",")
        ));
        if seen.insert(tid) {
            let b: Vec<u8> = bufs[tid].iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dbg.join(format!("golden/t{tid}.bin")), b)?;
        }
    }
    order.push(']');
    std::fs::write(dbg.join("order.json"), order)?;
    let mut shapes = String::from("[");
    for id in 0..nsrc {
        if id > 0 {
            shapes.push(',');
        }
        shapes.push_str(&format!(
            "{{\"tid\":{id},\"rows\":{},\"cols\":{}}}",
            ir.tensors[id].rows, ir.tensors[id].cols
        ));
    }
    shapes.push(']');
    std::fs::write(dbg.join("source_shapes.json"), shapes)?;
    let mut wt: Vec<u32> = weight_ids.iter().copied().collect();
    wt.sort_unstable();
    let wtj: Vec<String> = wt.iter().map(|w| w.to_string()).collect();
    std::fs::write(dbg.join("weight_tids.json"), format!("[{}]", wtj.join(",")))?;
    // granite ScalarMul scale VALUES (index i ↔ `scalarmul_scale_tid(i)`). SAME first-seen walk order as
    // `compute_bundle_layout` (both iterate `ir.nodes`, dedup by bits) ⇒ index↔TID consistent. The worker
    // reads this + binds each `t{scalarmul_scale_tid(i)} = [scale_i]` so the on-device pointwise `mul` gets
    // the real value (unbound = 0 = wrong). Written alongside source_shapes.json (the run reads both).
    let mut sm: Vec<f32> = Vec::new();
    for node in &ir.nodes {
        if let SubOp::ScalarMul { scale } = &node.op
            && !sm.iter().any(|s| s.to_bits() == scale.to_bits())
        {
            sm.push(*scale);
        }
    }
    let smj: Vec<String> = sm.iter().map(|s| format!("{s}")).collect();
    std::fs::write(
        dbg.join("scalarmul_scales.json"),
        format!("[{}]", smj.join(",")),
    )?;
    Ok(())
}

/// REAL per-core HBM start addresses for one tensor's allocate node (task #50 —
/// replaces the former same-base-for-all-cores stub). Each core `c` gets:
///
///   `startAddr[c] = segment_base(arg_index) + intra_base + per_core_byte_offset(c)`
///
/// where `per_core_byte_offset(c)` is the byte offset of core `c`'s work-slice
/// within the tensor, summed over the tensor's OWN layout dims that the
/// work-division split (torch-spyre `compute_ops.py:71 core_idx_to_slice_offset`):
///
///   `Σ_{dim ∈ layout, ACTIVE}  wk_slice_idx[c][dim] · per_core_extent(dim) ·
///                              inner_full_product(dim) · 2 bytes`
///
/// `inner_full_product(dim)` = product of the FULL extents of the layout dims
/// AFTER `dim` (row-major innermost-last) — the element stride to step one whole
/// `dim` index. `wk_slice_idx[c][dim]` is core `c`'s slice index along `dim` (0 if
/// `dim` is unsplit). Only ACTIVE (scale ≥ 0) dims contribute — a reduction /
/// broadcast dim is fully resident per core, so it adds no per-core offset (exactly
/// `arg.scales[dim] > 0` in the reference). `intra_base` is the per-dataspace bump
/// the caller assigns (kept 0 within a segment now that each arg owns a segment,
/// but threaded so a future intra-segment sub-allocator can offset).
///
/// LX tensors get the SAME address on every core (no per-core HBM split — the
/// scratchpad is core-local), matching the reference's `"lx" in allocation` arm.
///
/// COLLISION GUARD (build-time): every emitted address must be unique across the
/// `(arg, core)` space within one op — two cores writing the same OUTPUT byte is
/// silently-wrong. We assert disjointness per output tensor here.
/// THE single classifier of a tensor view's device [`StickLayout`] — the ONE place a view's arrangement
/// (RowBlocked vs Flat) is decided. Shared by [`per_core_addr`] (which bakes the per-core START through
/// it) and [`emit_sdsc`] (which DECLARES it to the arrangement authority), so the byte a tensor is
/// addressed at and the layout the authority records are DERIVED FROM THE SAME FUNCTION — they cannot
/// drift. `row_blocked` forces stick-major (the fp32 island); otherwise [`StickLayout::for_view_df`]
/// classifies, folding phantom unit dims so the choice is rank-INDEPENDENT (a rank-3 `[mb,out,1]`
/// pointwise view and a rank-2 `[m,k]` matmul view of one tensor classify identically).
pub(crate) fn view_stick_layout(
    view: &ArgView<'_>,
    plan: &WorkPlan,
) -> scratchy_subtile::sdsc_abstract::StickLayout {
    // PHYSICAL extent per dim, not this op's iteration view of it. `TensorArg::device_extent` has
    // always documented itself as the thing "every stride/offset is derived from, NOT from
    // `iteration_space[dim]`" — torch-spyre's `arg.device_size` / `dev_dim_size > it_dim_size` case.
    //
    // What it buys: an op may ITERATE a slice while ADDRESSING the whole allocation. A batched decode's
    // attention needs exactly that — a per-request pass computing ONE row per head (`mb = 1`) while the
    // operand's heads are still `mq` rows apart, because the stride between heads is a property of the
    // TENSOR, not of how much of it this op touches.
    //
    // ⚠ Reading it HERE alone did NOT make it real, and that was measured, not assumed: this decides the
    // ARRANGEMENT, while `per_core_addr` built its own extent vector from `plan.extent` and folded the
    // per-core START over that — so a declared extent moved the classification and not one emitted
    // address (probe: a rank-3 `[y,in,out]` kernel with `device_extent` on `in` or `out` emitted a
    // per-`y` start stride of `in·out` either way, byte for byte). Both now go through the ONE
    // `DeviceExtents::of_view`, which is why the field cannot be honoured in one and dropped in the
    // other. `Span::Swept`: an arrangement is classified from the shape the op presents.
    let extents = scratchy_subtile::sdsc_abstract::DeviceExtents::of_view(
        &view
            .layout
            .iter()
            .map(|&d| plan.extent(d) as usize)
            .collect::<Vec<_>>(),
        &view
            .device_extent
            .iter()
            .map(|o| o.map(|p| p as usize))
            .collect::<Vec<_>>(),
        view.layout
            .iter()
            .position(|&d| d == view.stick)
            .unwrap_or(usize::MAX),
        &vec![true; view.layout.len()],
        scratchy_subtile::sdsc_abstract::Span::Swept,
    );
    let host_size_us: Vec<usize> = extents.dims().to_vec();
    let stick_idx = view
        .layout
        .iter()
        .position(|&d| d == view.stick)
        .unwrap_or(usize::MAX);
    if view.row_blocked && !host_size_us.is_empty() {
        let feat: usize = host_size_us[1..].iter().product::<usize>().max(1);
        return scratchy_subtile::sdsc_abstract::StickLayout::row_blocked_df(
            host_size_us[0],
            feat,
            view.df,
        );
    }
    // ── mq>1 activation: STICK-MAJOR (RowBlocked), torch-spyre-faithful ───────────────────────────────
    // A multi-stick `[m>1, k]` non-KERNEL activation is addressed STICK-MAJOR (RowBlocked) — the SAME
    // residency as decode and BYTE-IDENTICAL to torch-spyre's descriptor (`element_torchspyre_parity`):
    // per-core start = RowBlocked `c·eps` (128 B for the [31,2048] mb-split), `layoutDimOrder_ = [mb,in]`
    // (mb-outermost host order VERBATIM — dxp applies `get_generic_stick_layout` INTERNALLY when it
    // reconstructs the stick-blocked `device_size` from `-1`; the mb-outermost label already yields the
    // stick-major walk on hardware), `maxDimSizes_ = -1`. This REPLACES the former `Flat` (row-major)
    // band-aid, whose start was `c·k` (4096 B): self-consistent for the pointwise/rmsnorm producer (each
    // core owned a whole row) — which is why scalarmul/embedding read correctly under it — but DIVERGENT
    // from torch-spyre, and it fed the MATMUL a row-major activation the PT (systolic) array cannot consume
    // (it streams K stick-major), so every projection came out garbage/zero (residual V-proj=0, K-garbage).
    // `for_view_df` classifies a rank-2 stick-on-last view as RowBlocked and a rank-3 head-major view as
    // `Flat`, so head-major attention intermediates stay row-major (correct — they are genuinely row-major).
    // Decode (rows<=1) and single-stick (cols<=lanes) are byte-identical to before (RowBlocked≡Flat there).
    // KERNELS (weights / KV cache) re-tile via their RetileDescriptor.
    scratchy_subtile::sdsc_abstract::StickLayout::for_view_df(&host_size_us, stick_idx, view.df)
}

fn per_core_addr(
    seg_base: u64,
    view: &ArgView<'_>,
    // THE one device layout, handed in by `emit_sdsc` — NOT re-derived here — so the per-core START this
    // computes and the on-card WALK `build_coordinates` computes read the SAME `StickLayout` (task #11).
    sl: &scratchy_subtile::sdsc_abstract::StickLayout,
    plan: &WorkPlan,
    wk_slice: &BTreeMap<
        String,
        BTreeMap<&'static str, scratchy_subtile::superdsc_opspec::SliceIndex>,
    >,
    intra_base: u64,
    cores: u32,
) -> Result<Vec<u64>, SuperDscError> {
    let cores = cores.max(1);
    // LX residency is core-local — every core shares the base (reference parity).
    if view.allocation.is_lx() {
        let base = intra_base; // LX uses a scratchpad-local offset, no segment.
        return Ok(vec![base; cores as usize]);
    }
    // `seg_base` is the RESOLVED HBM segment base for this tensor — the GLOBAL
    // role-segment from the BundleLayout (task #55) when the walk supplies one,
    // else the per-op `segment_base(arg_index)` fallback (single-op fixtures). The
    // per-core work-slice offset is added on top.
    let base = seg_base + intra_base;

    // The per-core start address is the device offset of the core's work-slice CORNER
    // under the SHARED device model `dev_off` — the SAME model that produces the DSC
    // descriptors the card reads (`DeviceTileLayout::device_size`/`stride_map`). Both key
    // on the IDENTICAL condition (`len == 2 && stick_idx == 1`): a rank-2 tensor sticked
    // on its LAST dim is re-tiled `[b/64, a, 64]` (row stride = the 64-stick WIDTH, not
    // `cols`); EVERY other shape is stored FLAT row-major `[total/64, 64]` (stick-dim
    // stride = its row-major inner product, NOT the product-of-non-stick). `dev_off`
    // reproduces exactly this split, so the per-core address and the on-card descriptor
    // can never diverge — by construction, not by hand-tuning.
    //
    // The old per-dim `per_core_stride_elems` sum used the RE-TILED stick stride
    // (product-of-non-stick) for ALL ranks — correct for the re-tiled rank-2 case but
    // WRONG for the FLAT rank-3 matmul OUTPUT `[mb, out, y]` stick=out: it strode `out`
    // by `mb·y` instead of the flat `y`, so for granite's RoPE-rotate output
    // (`[mb=32, out=128, y=1]`, head_dim=128 = 2 sticks) a row-split core aliased a
    // stick-split core's byte (the #50 collision at 0x158180). Kernels/inputs are
    // UNCHANGED — they split only leading/kernel dims where the two models agree; only the
    // output's over-strided stick dim is corrected. `dev_off` is Kani-proven injective for
    // every distinct index (`dev_off_general_injective`), so any disjoint (even) split —
    // which the monolith's `wk_slice` already is — is collision-free by construction.
    //
    // `_tile` is kept for its VALIDATION side-effect only: `DeviceTileLayout::new` routes
    // the stick extent through `StickExtent`, so a non-64-multiple stick is a `cargo build`
    // Err here (the offset model itself now comes solely from `dev_off`).
    let layout = view.layout;
    // The DATASPACE's OWN device shape, not the op's iteration space: a dim this view does not range
    // over (`scale != Active` — an mb-broadcast weight like the rmsnorm gamma `[1, hidden]` or an fp8
    // per-channel dequant scale) has extent 1 in HBM, so its stick-GROUP stride is `1·stk`, not
    // `mb·stk`. `dev_off` keys the group stride off exactly this shape, so feeding it the op's `mb`
    // made a split stick-dim corner stride `mb×` too far — out-core 1 of a 2-D `{mb, out}` split read
    // gamma past the end of gamma. Same `Scale::Active` rule `build_coordinates` uses for the on-card
    // walk, so the per-core START and the WALK read ONE shape (the invariant this function documents).
    //
    // THE ONE EXTENT CONSTRUCTOR ([`DeviceExtents::of_view`], `Span::Materialized` — a non-ranged dim
    // occupies one row). This was a hand-rolled `plan.extent` fold, and because it was hand-rolled it
    // silently dropped the view's DECLARED physical extent (`TensorArg::with_device_extent`) that
    // `view_stick_layout` — deciding the arrangement of the SAME view of the SAME op — already read. So
    // declaring one moved the classification and left every emitted address untouched: measurably inert,
    // in the exact direction a batched decode needs (iterate a 64-slot window of a resident cache,
    // ADDRESS the whole allocation). `of_view` is the only vector `off_view` accepts, so the two cannot
    // drift apart again — the guarantee this function's own doc claimed and did not have.
    //
    // Only the NON-STICK dims form `dev_off`'s stick-group stride (`rows·stk`); the stick dim's own
    // extent drives the 64/128-multiple stick guard below, and an inactive stick dim adds nothing to the
    // address anyway (its corner stays 0), so `of_view` never collapses it.
    let extents = scratchy_subtile::sdsc_abstract::DeviceExtents::of_view(
        &layout
            .iter()
            .map(|&d| plan.extent(d) as usize)
            .collect::<Vec<_>>(),
        &view
            .device_extent
            .iter()
            .map(|o| o.map(|p| p as usize))
            .collect::<Vec<_>>(),
        layout
            .iter()
            .position(|&d| d == view.stick)
            .unwrap_or(usize::MAX),
        &(0..layout.len())
            .map(|li| view.scale.get(li).map(|s| s.to_i64()).unwrap_or(1) > 0)
            .collect::<Vec<_>>(),
        scratchy_subtile::sdsc_abstract::Span::Materialized,
    );
    let host_size: Vec<u64> = extents.dims().iter().map(|&e| e as u64).collect();
    // Stick guard at the OPERAND's device format: fp8/int8 = 128-elem stick (StickExtent<Fp8> rejects a
    // non-128-multiple at cargo build — the compile-time fp8 width guard), fp16/bf16 = 64, fp32 = 32.
    match view.df {
        Df::Fp8 | Df::SenInt8 => {
            DeviceTileLayout::<Fp8>::new(layout, view.stick, &host_size)?;
        }
        Df::Fp32 => {
            DeviceTileLayout::<Fp32>::new(layout, view.stick, &host_size)?;
        }
        Df::Fp16 | Df::Bf16 => {
            DeviceTileLayout::<Fp16>::new(layout, view.stick, &host_size)?;
        }
    }

    // The per-core start flows through [`view_stick_layout`] — the SAME classifier `emit_sdsc` declares to
    // the arrangement authority — so the byte a tensor is written at and the layout recorded for it can
    // never drift. The stick width is the operand's OWN format (`view.df`: fp8→128, fp16→64), so an fp8
    // read lands on the right bytes; the classifier folds phantom unit dims so a rank-3 `[mb,out,1]` view
    // addresses the SAME bytes a rank-2 stick-major op would. See the byte-parity note below (rank-2
    // sticked-on-last → RowBlocked; rows==1 / single-stick are byte-identical). `row_blocked` forces it at
    // rank>2 (the fp32 island). KERNELS are re-tiled by their RetileDescriptor (staged layout, excluded).
    // ⚠ The prior `mq_flat_activation` flat start (`r·cols`) DIVERGED from torch-spyre — it addressed HBM as
    // row-major while torch-spyre addresses the dense RowBlocked packing. It only appeared to work because
    // `maxDimSizes_=actual` disabled the reconstruction (the card then honored the coordInfo eps stride =
    // flat feat-groups); with `maxDimSizes_=-1` the card reconstructs RowBlocked and the flat start would
    // mis-address. Removed for byte-parity: RowBlocked start + `-1` reconstruction = the torch-spyre pair.
    let view_layout = *sl; // the ONE layout, not a re-derivation
    let mut addrs = Vec::with_capacity(cores as usize);
    // EVERY core 0..cores flows through the SAME corner → offset fold below — per-core
    // addresses are enumerated one by one from each core's own slice indices, never
    // extrapolated from a lower core's address.
    for c in 0..cores {
        let slice = wk_slice.get(&c.to_string());
        // The logical CORNER (start index per layout dim) of core `c`'s work-slice. Only
        // ACTIVE (reference `arg.scales[dim] > 0`), SPLIT dims move the corner; a
        // reduction/broadcast dim or an unsplit dim is fully resident per core ⇒ its
        // corner component stays 0 (so it adds no per-core offset — reference parity).
        // Each component is the typed `(axis, slice-index)` product `WorkPlan::corner_elems`
        // — a `SliceIndex` cannot be scaled by anything else.
        let mut corner = vec![0usize; layout.len()];
        for (li, &d) in layout.iter().enumerate() {
            if view.scale.get(li).map(|s| s.to_i64()).unwrap_or(1) <= 0 {
                continue;
            }
            if plan.split_of(d).max(1) <= 1 {
                continue;
            }
            let idx = slice
                .and_then(|m| m.get(d))
                .copied()
                .unwrap_or(scratchy_subtile::superdsc_opspec::SliceIndex::UNSPLIT);
            corner[li] = plan.corner_elems(d, idx) as usize;
        }
        // fp8 W8A8 KERNEL: the weight is staged OUT-STICK-MAJOR packed [N/64, K/2, 2, 64] (see the
        // RetileDescriptor), so the per-core corner's device offset must use THAT layout, not the flat
        // 128-stick `for_view_df` model — else the address diverges from the staged bytes → garbage.
        // The corner components enter by AXIS NAME (`Corner<InAxis>`/`Corner<OutAxis>`, positional over
        // the rank-2 `[in, out]` kernel walk), so the pack's strides are attached to named axes.
        let off_elems = if matches!(view.role, Role::Kernel) && matches!(view.df, Df::Fp8) {
            use crate::ir::bridge::tiled_op_sdsc_op::matmul::walk;
            walk::fp8_kernel_stage_off(
                walk::Corner::fp8_kernel_in(&corner),
                walk::Corner::fp8_kernel_out(&corner),
                walk::Fp8KernelKRows::of_device_extents(&host_size),
            )
        } else {
            view_layout.off_view(&extents, &corner) as u64
        };
        // Byte width from the arg's typed dtype (`view.df`): `Df::Fp32` merge-partials are 4-byte, `Df::Fp8`/
        // int8 1-byte, everything else fp16 2-byte. INERT for fp16 (Fp16::WORD_LENGTH); only fp32 tensors (the
        // KSPLIT fp32-merge partials) address at 4 B. Gated by `fp32_tensor_byte_offset_is_4x_and_in_footprint`.
        addrs.push(base + off_elems * view.df.word_length() as u64);
    }
    Ok(addrs)
}

/// The `N_` iteration space for a shape-preserving ELEMENTWISE op over
/// `[rows, cols]`: mb=rows, out=cols, y=1 (the SAME named dims as a matmul
/// OUTPUT — verified against the real `sdsc_silu.json`, which uses mb_/out_/y_,
/// NOT i_/j_). `out` is the 64-fp16 stick (innermost) axis.
pub fn pointwise_iter_space(rows: u32, cols: u32) -> IterSpace {
    let mut it = IterSpace::empty();
    it.mb_ = rows as i64;
    it.out_ = cols as i64;
    it.y_ = 1;
    it
}

// ───────────────────────────────────────────────────────────────────────────
// TYPED OpSpec builders + `emit_sdsc` — the FRONTEND core (the witnesses) is in
// `superdsc_opspec`; these thin builders produce a validated `OpSpec`, then the
// SINGLE `emit_sdsc` lowers it into the wire structs above. The three
// near-duplicate `assemble_*` bodies collapse to: pick OpFunc + Role layouts +
// scales, divide once, emit once.
// ───────────────────────────────────────────────────────────────────────────

/// Thin wrapper — the ONE implementation now lives in
/// [`crate::tile_op::pointwise_lx_resident_generic`] (`TileOp`'s dispatch target for
/// `TileOpKind::PointwiseOrReduce`), Kani-proven equal to this formula at every symbolic extent. Kept
/// as a free function so the 6 existing `time_tile_for_lx(|p,opt| pointwise_lx_resident(…), …)` call
/// sites are unchanged; new call sites should build a `TileOp` instead.
fn pointwise_lx_resident(plan: &WorkPlan, n_operands: u64, out_per_time: u32, df: Df) -> u64 {
    crate::tile_op::pointwise_lx_resident_generic(plan, n_operands, out_per_time, df)
}

/// Build the typed [`OpSpec`] for ONE shape-preserving ELEMENTWISE op over
/// `[rows, cols]`. All operands share the rank-3 `[mb,out,y]` OUTPUT layout
/// (`cols`/`out` is the stick axis); scales all active.
///
/// Thin wrapper: builds the `[mb,out,y]` `TileOp` (the TileIR declaration for this shape) and hands
/// it to [`pointwise_opspec_from_tile`] — the SAME two-step split every other caller of that function
/// goes through, whether their `TileOp` came from here or from
/// [`crate::subtile_tape_to_tile_ir::node_to_single_tile_op`] (the live per-node path).
fn pointwise_opspec(
    op: OpFunc,
    rows: u32,
    cols: u32,
    in_names: &[&str],
    o_name: &str,
    head_major: bool,
) -> Result<OpSpec, String> {
    let cols_ext = StickExtent::<Fp16>::new(cols)?;
    let dims = vec![
        ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: cols_ext.elems(),
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
    ];
    let n_operands = (in_names.len() + 1) as u32;
    let tile_op = crate::tile_op::TileOp {
        kind: crate::tile_op::TileOpKind::PointwiseOrReduce { n_operands },
        dims,
        df: Df::Fp16,
    };
    pointwise_opspec_from_tile(&tile_op, op, in_names, o_name, head_major)
}

/// How one elementwise operand maps onto the `[mb, out, y]` iteration space — its
/// per-dim [`Scale`]. The OUTPUT is always all-`Active`; an INPUT may BROADCAST a
/// dim: `mb` broadcast (a `[1, cols]` row vector like RmsNorm's gamma) → `mb` =
/// `RedNonStick` (-1, a non-stick size-1 dim); `out` broadcast (a `[m, 1]` column
/// vector like RmsNorm's inv_rms, or a `[1,1]` scalar const) → `out` = `RedStick`
/// (-2, the one-stick `alpha_=0` broadcast READ proven on-card 2026-06-25). Both
/// are MaterializedStick-safe. `y` is always `Active` (the unused trailing slot).
#[derive(Clone, Copy)]
pub struct EwOperand<'a> {
    // Fields are PRIVATE and there are NO raw ctors: the ONLY way to build an EwOperand is `In::ew()`,
    // which requires a typed `&Stk<K>` handle. So a raw `EwOperand { name: &str, .. }` or the old
    // `EwOperand::full(&str)` is UNCONSTRUCTABLE outside this module — the weakly-typed input path is
    // eliminated. Every pointwise/reduce input carries its layout kind in its type via `In<K>`.
    name: &'a str,
    mb_broadcast: bool,
    out_broadcast: bool,
    col_offset: u32,
}
impl<'a> EwOperand<'a> {
    pub(crate) fn scale(&self) -> [Scale; 3] {
        [
            if self.mb_broadcast {
                Scale::RedNonStick
            } else {
                Scale::Active
            },
            if self.out_broadcast {
                Scale::RedStick
            } else {
                Scale::Active
            },
            Scale::Active,
        ]
    }

    // Read-only accessors, `pub(crate)` so the TileIR→SdscOp bridge functions that read these
    // fields (`ir::bridge::tiled_op_sdsc_op`) can do so from outside this module WITHOUT making
    // the fields themselves `pub(crate)` — which would let crate-internal code brace-construct
    // an `EwOperand` directly, defeating the "no raw ctors, only `In::ew()`" invariant the private
    // fields exist to enforce (see the struct's own doc comment above).
    pub(crate) fn name(&self) -> &str {
        self.name
    }
    pub(crate) fn out_broadcast(&self) -> bool {
        self.out_broadcast
    }
    pub(crate) fn col_offset(&self) -> u32 {
        self.col_offset
    }
}

/// Build the typed [`OpSpec`] for ONE elementwise op over `[rows, cols]` with
/// PER-OPERAND broadcast (the generalization of [`pointwise_opspec`], which is the
/// all-`full` case). Inputs may broadcast over `mb` or `out` ([`EwOperand`]); the
/// output spans full `[mb,out,y]`. `out` is the 64-stick axis. Broadcast over the
/// stick (`out`) is proven on-card (RedStick `alpha_=0` fold). A TILED broadcast op
/// is REFUSED (build `Err`): a broadcast-over-`out` operand must NOT advance per
/// trip, which `concrete_trips` does not yet special-case — better a build failure
/// than a silently-wrong per-trip address.
/// fp16 wrapper (the 64-stick default): every existing caller. fp32 (the torch-spyre RMSNorm) uses
/// [`pointwise_broadcast_opspec_df`].
fn pointwise_broadcast_opspec(
    op: OpFunc,
    rows: u32,
    cols: u32,
    inputs: &[EwOperand<'_>],
    o_name: &str,
    out_offset: u32,
    head_major: bool,
) -> Result<OpSpec, String> {
    pointwise_broadcast_opspec_df(
        op,
        rows,
        cols,
        inputs,
        o_name,
        out_offset,
        head_major,
        Df::Fp16,
    )
}

#[allow(clippy::too_many_arguments)]
fn pointwise_broadcast_opspec_df(
    op: OpFunc,
    rows: u32,
    cols: u32,
    inputs: &[EwOperand<'_>],
    o_name: &str,
    out_offset: u32,
    head_major: bool,
    df: Df,
) -> Result<OpSpec, String> {
    // fp16 (default) keeps the proven 64-stick basis; Df::Fp32 (the torch-spyre fp32 rmsnorm) uses a
    // 32-elem/4-byte stick. The stick extent MUST be the ACTUAL dtype's (fp32 = 32-stick) — a per-row
    // scalar lane is cols=32, valid fp32 but NOT a 64-multiple, so an Fp16 extent would wrongly reject it.
    let stick_elems = if matches!(df, Df::Fp32) {
        StickExtent::<Fp32>::new(cols)?.elems()
    } else {
        StickExtent::<Fp16>::new(cols)?.elems()
    };
    let dims = vec![
        ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df,
        },
        ItDim {
            name: "out",
            size: stick_elems,
            is_reduction: false,
            is_stick: true,
            df,
        },
        ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df,
        },
    ];
    let n_operands = (inputs.len() + 1) as u32;
    let tile_op = crate::tile_op::TileOp {
        kind: crate::tile_op::TileOpKind::PointwiseOrReduce { n_operands },
        dims,
        df,
    };
    pointwise_broadcast_opspec_from_tile(
        &tile_op, rows, cols, op, inputs, o_name, out_offset, head_major,
    )
}

/// [`pointwise_broadcast_opspec`] → [`EmittedOp`].
/// RANK-1 SFP op over a SINGLE 64-stick dim `"mb"` — matching torch-spyre's REAL
/// emitted rmsnorm rsqrt SDSC (`N_={mb:64}`, `layoutDimOrder=stickDimOrder=["mb"]`,
/// scale `[1]`). The rmsnorm reduced scalar (mean+eps) is byte-identical whether the
/// emitter calls it `[mb=1, out=64-stick, y=1]` (scratchy's rank-3) or `[mb=64-stick]`
/// (torch-spyre's rank-1) — same 64 contiguous fp16 in one stick. But the SFP
/// Newton-Raphson REFINEMENT collapses to the seed (rsqrt→1/x) under the rank-3 form
/// (the phantom row/y dims wrapping the stick), and runs correctly under rank-1
/// (verified: torch-spyre compiles+runs the rank-1 form on the same dd2). So a
/// reduced-scalar SFP transcendental MUST be emitted rank-1.
fn sfp_rank1_opspec(
    op: OpFunc,
    stick_elems: u32,
    in_name: &str,
    out_name: &str,
) -> Result<OpSpec, String> {
    let ext = StickExtent::<Fp16>::new(stick_elems)?;
    // Dim name "out" (NOT "mb") — matches torch-spyre's REAL standalone hbm+lx rsqrt
    // (verified on-card correct). The prior "mb" matched the FUSED rmsnorm rsqrt, but that
    // one is lx-only; with hbm+lx (the per-op case) torch-spyre uses "out", and that is the
    // ONLY remaining byte-difference between scratchy's compiled rsqrt program and torch-
    // spyre's working one (the compiled programs are otherwise byte-identical — 6 bytes,
    // dim-name-derived). "mb"+hbm is the one combo torch-spyre never emits.
    let dims = vec![ItDim {
        name: "out",
        size: ext.elems(),
        is_reduction: false,
        is_stick: true,
        df: Df::Fp16,
    }];
    let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores)?;
    let device_dims = plan.iter_syms(["out"]);
    let mk = |is_input: bool, name: &str| {
        TensorArg::<1>::new(
            is_input,
            name.to_string(),
            Role::Output,
            [Scale::Active],
            device_dims,
            ["out"],
            "out",
            Allocation::Hbm,
        )
        .map_err(|e| e.0)
    };
    Ok(OpSpec {
        op,
        is_reduction: false,
        iter: plan,
        args: vec![
            AnyTensorArg::R1(mk(true, in_name)?),
            AnyTensorArg::R1(mk(false, out_name)?),
        ],
        op_info: OpInfo::None,
        tiled_symbols: vec![],
        time_tile: None,
    })
}

/// [`sfp_rank1_opspec`] → [`EmittedOp`].
pub fn assemble_sfp_rank1(
    op_name: &str,
    op_func: &'static str,
    stick_elems: u32,
    in_name: &str,
    out_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op = sfp_rank1_opspec(op_func_from_str(op_func), stick_elems, in_name, out_name)
        .unwrap_or_else(|e| panic!("assemble_sfp_rank1 {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_sfp_rank1 {op_name}: {e}"))
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_pointwise_broadcast(
    op_name: &str,
    op_func: &'static str,
    // TYPED row/width slots, same contract as `assemble_pointwise_broadcast_off`: every caller
    // states WHICH row quantity and WHICH width fills the slot through the types' named doors.
    rows: scratchy_subtile::sdsc_abstract::RowCount,
    cols: scratchy_subtile::sdsc_abstract::BlockCols,
    inputs: &[EwOperand<'_>],
    o: &Stk<RowBlockedTag>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // RowBlocked output ⇒ residual token stream (not head-major). head_major is derived from the type.
    assemble_pointwise_broadcast_off(
        op_name,
        op_func,
        rows,
        cols,
        inputs,
        o,
        scratchy_subtile::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    )
}

/// [`assemble_pointwise_broadcast`] for a PER-HEAD / row-structured attention op (`head_major`):
/// forces the legacy rank-3 flat layout regardless of `SCRATCHY_SUPERDSC_STICKMAJOR`, so the
/// stick-major residual seam fix NEVER rewrites an attention per-head pointwise (its `mb` axis is
/// heads/head·rows, NOT the residual token count). Byte-identical to the plain assembler on the
/// OFF path; on the ON path it keeps the attention path rank-3 (the fix targets residual ops only).
#[allow(clippy::too_many_arguments)]
pub fn assemble_pointwise_broadcast_hm(
    op_name: &str,
    op_func: &'static str,
    // TYPED row/width slots — the head-major caller states its own row structure through the doors
    // (head-major rows, mask rows) exactly as the plain assembler's callers do.
    rows: scratchy_subtile::sdsc_abstract::RowCount,
    cols: scratchy_subtile::sdsc_abstract::BlockCols,
    inputs: &[EwOperand<'_>],
    o: &Stk<FlatTag>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // Flat output ⇒ head-major (per-head attention) op; the builder derives head_major from the type.
    assemble_pointwise_broadcast_off(
        op_name,
        op_func,
        rows,
        cols,
        inputs,
        o,
        scratchy_subtile::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    )
}

/// The per-chunk OUTPUT column offset (in elements) the emitter must apply for a
/// COLUMN-BLOCK-SPLIT wide op. The tape splits wide ops (e.g. the granite MLP
/// intermediate 12800 → the col-blocks `[0,8192)` + `[8192,12800)`) into multiple
/// nodes that share one output tensor; each chunk's `region.cols.start` is where it
/// MUST write. Dropping it (writing every chunk at base 0) leaves the later blocks'
/// columns UNWRITTEN — MEASURED on-card: silu·up `t65[8192..12800] = 0`, ~27% of the
/// MLP energy zeroed → wrong token. LOCKED by `silumul_chunks_cover_output`.
pub fn pointwise_chunk_out_offset(cols_start: u32) -> u32 {
    cols_start
}

/// [`assemble_pointwise_broadcast`] that writes the output to a COLUMN-BLOCK OFFSET
/// (`out_offset` elements into the output tensor) — see [`pointwise_chunk_out_offset`].
#[allow(clippy::too_many_arguments)]
pub fn assemble_pointwise_broadcast_off<O: KindTag>(
    op_name: &str,
    op_func: &'static str,
    // TYPED row/width slots. Every caller states WHICH row quantity (`RowCount`'s doors: mask rows,
    // chunk rows, padded rows, pad remainder, token rows) and WHICH width (`BlockCols`'s doors: one
    // stick, the new block's `mq_pad`, the head dim, feature cols) fills the slot — the quantities
    // are numerically equal across whole rung families (64 == 64 at mq<=2, hd == stick at hd=64),
    // so a bare `u32` could carry any of them without anything noticing.
    rows: scratchy_subtile::sdsc_abstract::RowCount,
    cols: scratchy_subtile::sdsc_abstract::BlockCols,
    inputs: &[EwOperand<'_>],
    o: &Stk<O>,
    // TYPED, like the inputs' `In::sliced`. A write offset is a DEVICE address, so it must come from
    // a nest that knows the tensor's extents; the only constructor is `View::dev`. When this was a
    // bare `u32` a caller could pass a flat column index, and one did -- the split MLP intermediate
    // wrote its second column block at element 8192 instead of `(8192/64)*(rows*64)`, correct at one
    // row and wrong at every prefill. The reads were already typed and were already right.
    out_offset: scratchy_subtile::addr::DevOff,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let rows = rows.get();
    let cols = cols.get();
    let out_offset = out_offset.into_raw_elems();
    // The OUTPUT type drives the layout: a `Stk<FlatTag>` output IS a head-major (per-head attention)
    // op — rank-3 flat; a `Stk<RowBlockedTag>` output is the residual token stream (rank-2 stick-major).
    // So `head_major` is DERIVED from the output kind (no redundant bool that could disagree with it).
    let o_name = o.name();
    let head_major = O::kind() == StickKind::Flat;
    // BUILD-TIME GUARD (guard-every-crash): an SFP transcendental whose input is read
    // OUT-BROADCAST (a reduction `out` dim) at MULTI-STICK width (cols > 64 → the op is
    // split across cores) CRASHES the dxp compile with `map::at` (ddc drops the
    // per-core SFP-split / synthesized chunk state for this reduction+split geometry).
    // OBSERVED 2026-06-27: a full-width rsqrt over the rms `meps` (out-broadcast,
    // numCoresUsed_=9) → `dxp_standalone --bundle` map::at, a `cargo build` failure via
    // the bake's dxp-nonzero panic. The correct geometry for a transcendental on a
    // REDUCED scalar (rsqrt of the rms mean) is 1-STICK; the broadcast belongs in the
    // FOLLOWING multiply (non-transcendental → safe). This fn runs in the `#[forward]`
    // proc-macro (compile time), so the panic IS a cargo-build error — encoding the
    // dxp contract in the lowering, not leaving it as a raw on-card abort.
    const SFP_SPLIT_TRANSCENDENTALS: &[&str] = &[
        "reciprocal",
        "sqrt",
        "rsqrt",
        "gelu",
        "relu",
        "tanh",
        "layernormscale",
        "abs",
        "sigmoid",
        "silu",
        "exp",
    ];
    if SFP_SPLIT_TRANSCENDENTALS.contains(&op_func)
        && cols > Fp16::ELEMS_PER_STICK
        && inputs.iter().any(|i| i.out_broadcast)
    {
        panic!(
            "[superdsc sfp-transcendental-guard] {op_name}: SFP `{op_func}` emitted at \
             multi-stick width (cols={cols} > {}) with an OUT-BROADCAST (reduction) input — \
             the full-width-transcendental geometry that CRASHES the dxp compile (map::at in \
             ddc's per-core SFP split). A transcendental on a reduced scalar MUST be 1-stick \
             (cols={}); broadcast in the following multiply. (guard-every-crash-at-build-time)",
            Fp16::ELEMS_PER_STICK,
            Fp16::ELEMS_PER_STICK,
        );
    }
    let op = pointwise_broadcast_opspec(
        op_func_from_str(op_func),
        rows,
        cols,
        inputs,
        o_name,
        out_offset,
        head_major,
    )
    .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast {op_name}: {e}"))
}

/// fp32 elementwise (the torch-spyre RMSNorm island): every operand IEEE_FP32 (`df`), whole-tensor,
/// non-head-major. Callers must upcast ALL operands via `dl16tofp32` first (no mixed [fp16,fp32] op).
#[allow(clippy::too_many_arguments)]
pub fn assemble_pointwise_broadcast_df(
    op_name: &str,
    op_func: &'static str,
    rows: u32,
    cols: u32,
    inputs: &[EwOperand<'_>],
    o_name: &str,
    df: Df,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // head_major=TRUE ⇒ rank-3 flat (NOT rank-2 stick-major). The fp32 rmsnorm island runs FLAT so it
    // agrees with the fp16↔fp32 converts (which MUST be rank-3 — the narrowing convert crashes dxp in
    // rank-2). Consistent flat inside the island; only the x-read / out-write boundaries touch RowBlocked.
    let op = pointwise_broadcast_opspec_df(
        op_func_from_str(op_func),
        rows,
        cols,
        inputs,
        o_name,
        0,
        true,
        df,
    )
    .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast_df {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast_df {op_name}: {e}"))
}

/// The broadcast gate probe (`mul(x[full], v[out-broadcast])`) via the general
/// [`assemble_pointwise_broadcast`] — kept for the on-card de-risk dump.
pub fn assemble_broadcast_mul_gate(
    op_name: &str,
    // TYPED like every pointwise row/width slot: the probe's caller states what its tensor's rows
    // and columns are.
    rows: scratchy_subtile::sdsc_abstract::RowCount,
    cols: scratchy_subtile::sdsc_abstract::BlockCols,
    x_name: &str,
    v_name: &str,
    o_name: &str,
) -> EmittedOp {
    let mut sym = 0i64;
    // "multiply" is the INTERNAL op_func_from_str key (→ OpFunc::Multiply, whose
    // wire name() is "mul"); passing "mul" would miss the match and default to Add.
    assemble_pointwise_broadcast(
        op_name,
        "multiply",
        rows,
        cols,
        &[
            In::full(&rbo(x_name)).ew(),
            EwOperand {
                mb_broadcast: false,
                out_broadcast: true,
                ..In::full(&rbo(v_name)).ew()
            },
        ],
        &rbo(o_name),
        &mut sym,
        None, // gate probe (not in the walk) → no global layout
    )
}

/// GATE probe (RoPE slice-operand prerequisite): `add(x[:, 0:half], x[:, half:2·half])`
/// → `out[m, half]` — BOTH inputs slice the SAME dataspace `x` (`[m, 2·half]`) at
/// offsets 0 and `half`. Tests whether dxp accepts an operand whose AllocNode
/// `startAddr` is OFFSET into a tensor (the `.with_offset` feature). If it compiles,
/// RoPE composes as multiplies/adds over half-slices (no shuffle). NOT in the walk.
pub fn assemble_slice_add_gate(
    op_name: &str,
    rows: u32,
    half: u32,
    x_name: &str,
    o_name: &str,
) -> EmittedOp {
    let op = slice_add_gate_opspec(rows, half, x_name, o_name)
        .unwrap_or_else(|e| panic!("assemble_slice_add_gate {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let mut sym = 0i64;
    emit_sdsc_tiled(op_name, &op, &folds, &mut sym, None)
        .unwrap_or_else(|e| panic!("assemble_slice_add_gate {op_name}: {e}"))
}

fn slice_add_gate_opspec(rows: u32, half: u32, x: &str, o: &str) -> Result<OpSpec, String> {
    let half_ext = StickExtent::<Fp16>::new(half)?; // the op iterates over [m, half]
    let dims = vec![
        ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: half_ext.elems(),
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
    ];
    let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores)?;
    let dd = plan.iter_syms(["mb", "out", "y"]);
    let mk = |is_input: bool, name: &str| {
        TensorArg::<3>::new(
            is_input,
            name.to_string(),
            Role::Output,
            [Scale::Active, Scale::Active, Scale::Active],
            dd,
            ["mb", "out", "y"],
            "out",
            Allocation::Hbm,
        )
        .map_err(|e| e.0)
    };
    let x_lo = mk(true, x)?; // offset 0
    let x_hi = mk(true, x)?.with_offset(half); // reads x[:, half:]
    let out = mk(false, o)?;
    Ok(OpSpec {
        op: OpFunc::Add,
        is_reduction: false,
        iter: plan,
        args: vec![
            AnyTensorArg::R3(x_lo),
            AnyTensorArg::R3(x_hi),
            AnyTensorArg::R3(out),
        ],
        op_info: OpInfo::None,
        tiled_symbols: vec![],
        time_tile: None,
    })
}

/// The emitted SuperDSC op + the per-trip stride the CONCRETE-UNROLL needs. For a
/// time=1 op `affine_strides` is all-`{}` (one flat `sdsc_execute`, addresses baked
/// concrete in the json). For a TIME-TILED op (time=N): each tiled HBM tensor
/// carries a `{out: stride_bytes}` map, and [`concrete_trips`] expands the op into
/// N concrete SdscOps (trip `t`'s tiled startAddr = base + `t·stride_bytes`),
/// emitted as N flat `sdsc_execute` — NO symbols, NO `scf.for`. This sidesteps
/// dxp's always-on LoopUnroll (task #53).
/// WHICH BUFFER EACH OPERAND SLOT BINDS, in `args` order — the one fact the wire `SdscOp` does not carry.
///
/// ⛔⛔⛔ ITS ABSENCE MADE A GATE UNFIREABLE. The emitted JSON identifies an operand only by its SLOT:
/// `dsName_` is `Tensor0/1/2` and `ldsIdx_` is 0/1/2, both op-local. `emitted_attn_agreement`'s premise —
/// two ops that name the same block of the same buffer must address it at the same offset — therefore had
/// nothing to group by, and it silently keyed on the OP NAME instead, which is unique per op by
/// construction. Measured: 625 ops, 625 keys, uses-per-key `{1: 625}`. Both of its reports printed 0
/// disagreements for every shape ever tried, including the ones with known head_dim>64 bugs, and its gate
/// asserted that an empty set was empty.
///
/// The emitter knows this at lowering time (`OpSpec::args` carries the name and direction) and threw it
/// away one line later. Keeping it costs nothing and changes no emitted byte — it is not serialised into
/// the op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgBinding {
    pub buffer: String,
    pub is_input: bool,
    /// This operand's MATERIALIZED footprint in bytes — see [`materialized_bytes`], the ONE
    /// derivation. Carried on the binding so a consumer that only has the emitted descriptor (the
    /// attention producer/consumer lock) reads the same number the emitter sized the operand with,
    /// instead of re-deriving it from `layoutDimOrder_` and getting a BROADCAST axis wrong.
    pub footprint_bytes: u64,
    /// ⭐ WHICH bytes, as blocks — see [`TouchedBlocks`]. `footprint_bytes` answers "how many", a
    /// different question that stopped having the same answer once one op writes a strided region.
    pub touched: TouchedBlocks,
}

/// ⭐⭐⭐ AN OPERAND'S MATERIALIZED FOOTPRINT IN BYTES — the ONE derivation, for every caller.
///
/// A dim that is COLLAPSED for this operand (`scale != Active`: a reduction OUTPUT dim, or a
/// BROADCAST input dim — `out_broadcast`/`mb_broadcast`) does NOT occupy its full iteration extent:
/// the stick axis collapses to ONE stick, a non-stick collapsed dim to a single row. Using the raw
/// iteration extent over-counts these — an out-broadcast scalar `[nqh, 1 stick]` read across a
/// `cap`-wide output, or a per-head reduce accumulator.
///
/// ⛔ THIS WAS SPELLED TWICE. `emit_sdsc` had it inline (sizing a synth slot and its own footprint
/// guard) and the attention producer/consumer lock re-derived it from the EMITTED
/// `layoutDimOrder_` × per-core `el_`, which cannot see `scale` at all — so the lock counted a
/// col-broadcast `corr` as `rows × hd` instead of `rows × one stick` and refused a correct op for
/// reading one stick past its buffer. Two derivations of ONE quantity is the defect family this
/// codebase keeps paying for; there is now one function and both callers ask it.
pub fn materialized_bytes(v: &scratchy_subtile::superdsc_opspec::ArgView, iter: &WorkPlan) -> u64 {
    v.layout
        .iter()
        .zip(v.scale.iter())
        .map(|(&d, &sc)| match sc {
            Scale::Active => iter.extent(d) as u64,
            _ if d == v.stick => v.df.elems_per_stick() as u64, // fp16/bf16: 64; fp32: 32; fp8/int8: 128
            _ => 1u64,
        })
        .product::<u64>()
        * v.df.word_length() as u64 // fp16/bf16: 2 B; fp32: 4 B; fp8/int8: 1 B
}

/// ⭐⭐⭐ EACH DECLARED AXIS'S REAL DEVICE STRIDE IN BYTES, **DIFFERENCED FROM THE ONE LAW** — never
/// multiplied out here.
///
/// `stride(i) = off_view(corner with i = 1) - off_view(corner all 0)`, evaluated by
/// [`StickLayout::off_view`] over [`DeviceExtents::of_view`] — the same law the placement nests use
/// and the same one dxp reconstructs. A hand-written product cannot be right about a stick-blocked
/// tensor, whose device nest splits the stick axis into `(stick_groups, lanes)`.
pub fn declared_axis_block_steps(
    v: &scratchy_subtile::superdsc_opspec::ArgView,
    iter: &WorkPlan,
) -> Vec<u64> {
    let stick_idx = v
        .layout
        .iter()
        .position(|&d| d == v.stick)
        .unwrap_or(usize::MAX);
    let ext = scratchy_subtile::sdsc_abstract::DeviceExtents::of_view(
        &v.layout
            .iter()
            .map(|&d| iter.extent(d) as usize)
            .collect::<Vec<_>>(),
        &v.device_extent
            .iter()
            .map(|o| o.map(|p| p as usize))
            .collect::<Vec<_>>(),
        stick_idx,
        &(0..v.layout.len())
            .map(|li| v.scale.get(li).map(|s| s.to_i64()).unwrap_or(1) > 0)
            .collect::<Vec<_>>(),
        scratchy_subtile::sdsc_abstract::Span::Materialized,
    );
    let layout =
        scratchy_subtile::sdsc_abstract::StickLayout::for_view_df(ext.dims(), stick_idx, v.df);
    let zero = vec![0usize; v.layout.len()];
    let base = layout.off_view(&ext, &zero) as u64;
    (0..v.layout.len())
        .map(|i| {
            // ⛔⛔⛔ THE STICK AXIS IS DIFFERENCED BY `lanes`, NOT BY ONE, AND THAT IS NOT A DETAIL.
            // `dev_off_stk` is `(j/stk)*(rows*stk) + i*stk + (j%stk)`: a UNIT step on the stick axis
            // moves ONE ELEMENT inside the current stick (`j%stk`), while the distance a sweep of that
            // axis actually repeats by is the STICK-GROUP plane (`rows*stk`). Differencing by 1 reports
            // 2 bytes for an axis whose real block step is thousands, which credited a full-head-width
            // write with `one stick + 2 bytes` and refused it (`coverage 4098` where the buffer is 8192).
            let delta = if i == stick_idx {
                v.df.elems_per_stick() as usize
            } else {
                1
            };
            let mut c = zero.clone();
            c[i] = delta;
            (layout.off_view(&ext, &c) as u64).saturating_sub(base) * v.df.word_length() as u64
        })
        .collect()
}

/// ⭐⭐⭐ THE BYTES AN OPERAND TOUCHES, AS A **SET OF BLOCKS** — because a full-head-width write on a
/// stick-blocked buffer is NOT one interval. It is `nslab` blocks a whole `rows*stick` plane apart.
///
/// ⛔ THE FOURTH MODEL OF THIS QUANTITY, AND THE FIRST THREE WERE ALL INTERVALS: base-only,
/// whole-iteration-volume, per-core-fold (blind to `Scale`), and dense-product (blind to STRIDE).
/// Each earlier fix corrected the arithmetic while keeping "an operand's bytes are contiguous", which
/// the stick-group collapse breaks outright.
///
/// ⛔ AND NOT A BOUNDING SPAN: that OVERSTATES the write, so the producer/consumer lock would start
/// ACCEPTING real gaps. Overstating a write is the direction that hides bugs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TouchedBlocks {
    /// The contiguous run at each block, in bytes.
    pub run_bytes: u64,
    /// `(count, stride_bytes)` per outer axis this op sweeps more than once.
    pub outer: Vec<(u64, u64)>,
}

impl TouchedBlocks {
    /// Every block's byte offset from the operand's base.
    pub fn offsets(&self) -> Vec<u64> {
        let mut offs = vec![0u64];
        for &(count, stride) in &self.outer {
            let mut next = Vec::with_capacity(offs.len() * count.max(1) as usize);
            for &o in &offs {
                for i in 0..count.max(1) {
                    next.push(o + i * stride);
                }
            }
            offs = next;
        }
        offs
    }
}

/// [`TouchedBlocks`] for one operand view, with every distance asked of [`declared_axis_block_steps`].
pub fn touched_blocks(
    v: &scratchy_subtile::superdsc_opspec::ArgView,
    iter: &WorkPlan,
) -> TouchedBlocks {
    let word = v.df.word_length() as u64;
    let lanes = v.df.elems_per_stick() as u64;
    let stride = declared_axis_block_steps(v, iter);
    let swept: Vec<u64> = v
        .layout
        .iter()
        .enumerate()
        .map(|(i, &d)| match v.scale[i] {
            Scale::Active => iter.extent(d) as u64,
            _ if d == v.stick => lanes,
            _ => 1,
        })
        .collect();
    // The STICK axis contributes a contiguous run of one stick, and one BLOCK per further stick group
    // it sweeps — that group step is the axis's own differenced stride.
    let si = v
        .layout
        .iter()
        .position(|&d| d == v.stick)
        .unwrap_or(usize::MAX);
    let mut run = lanes.min(swept.get(si).copied().unwrap_or(lanes)) * word;
    let mut outer: Vec<(u64, u64)> = Vec::new();
    for i in 0..v.layout.len() {
        let count = if i == si {
            swept[i].div_ceil(lanes)
        } else {
            swept[i]
        };
        if count <= 1 {
            continue;
        }
        let st = stride[i];
        if st == run {
            run *= count;
        } else if st > 0 {
            outer.push((count, st));
        }
    }
    TouchedBlocks {
        run_bytes: run,
        outer,
    }
}

pub struct EmittedOp {
    pub op_name: String,
    pub op: SdscOp,
    /// Parallel to `op.args`: the buffer each operand slot binds. See [`ArgBinding`].
    pub arg_bindings: Vec<ArgBinding>,
    /// `time` trips (1 = not tiled). The pre-unroll factor for [`concrete_trips`].
    pub time: u32,
    /// Parallel to `op.args` (tensor index): `{out: stride_bytes}` for a tiled
    /// HBM tensor, `{}` for a non-tiled / LX / INPUT tensor. `stride_bytes` =
    /// `per_iter_out_elems * device_stride_out * 2` (the per-trip HBM advance the
    /// concrete unroll adds onto the base address).
    pub affine_strides: Vec<BTreeMap<&'static str, i64>>,
    /// Non-zero ONLY for KV cache-write ops (`op_name` "cachewr_*"): the per-step
    /// slot stride in BYTES (= head_dim·2). The runtime adds `seq_pos·slot_stride_bytes`
    /// to the op's KV-segment offset so step `p` writes the cache at slot `p`. 0 = the
    /// op uses the plain per-layer offset (every non-cache-write op).
    pub slot_stride_bytes: u32,
    /// SLAB WRITE (kill-restickify Stage 2, hd==64 only): TRUE for the slab-granular INCREMENTAL Kᵀ
    /// restickify — it transposes ONLY the current 64-slot slab (not the whole active K each step), and
    /// the shim shifts BOTH its resident input (natural K cache) AND output (resident kct) seg2 bases by
    /// `(seq_pos/64)·slab_stride_bytes` so step `p` re-transposes slab `p/64` in place; prior slabs
    /// persist. Classified [`GroupKind::Slab`] — a DISTINCT kind from `Slot` (its `slab_stride_bytes`
    /// = STICK_BYTES·stick = 8192 ≠ the cachewr's `slot_stride_bytes` 128, so fusing would trip the Slot
    /// uniform-stride assert). 0/`false` = the plain full-active_cap restickify. Default false/0.
    pub slab_write: bool,
    pub slab_stride_bytes: u32,
    /// PAGE FOLD: folds ONE page of resident prefix into the running online-softmax state. Such ops
    /// are grouped APART from the body and re-launched once per page the context spans, each launch
    /// rebased to that page. The fold is an accumulation, so repeating it IS the loop — there is no
    /// separate loop body and no state to thread, which is why unbounded context costs no
    /// restructuring.
    pub kv_page_fold: bool,
    /// THIS OP'S KERNEL STEPS ONE REQUEST PER UNIT OF ITS BATCH AXIS, so ONE launch computes every row
    /// of the batch and the fold needs a pass per PAGE rather than per (request, page).
    ///
    /// Travels to the runtime through the manifest because the runtime cannot see it: what axes an op
    /// was baked with is not visible from a session. Claiming it without the axis collapses the passes
    /// while the kernel still reads one request's K — every row but one attending the wrong history,
    /// fluently. See `fold_plan::reps`, which needs this AND a pool that admits a single stride.
    pub kv_batched_requests: bool,
    /// THE ROW REGIME this op's fold passes were baked under (meaningful with [`Self::kv_page_fold`]):
    /// whole-batch passes sweep every `nqh*mq` shared-buffer row and need NO per-pass rebase in the
    /// intermediate segment; per-request passes carry one request's `nqh` rows and do. Declared on the
    /// bundle ([`KV_FOLD_ROWS_PER_REQUEST_KEY`]) because the row count a pass was baked at is not
    /// visible from a session — the worker derives `set_int_stride` from what it loads
    /// ([`manifest_fold_row_regime`]) instead of restating the emitter's choice as its own constant.
    pub kv_fold_rows: scratchy_subtile::sdsc_abstract::FoldRowRegime,
    /// PAGED cache-write modulus: the write slot is the position MODULO this, because the page a
    /// position lives in is chosen by the request's block table, not by the position. Applying an
    /// absolute-position shift to a paged bundle writes past the page — into another request's KV
    /// once the pool is shared. Non-zero marks the bundle paged; a runtime that does not know the
    /// key must REFUSE it rather than fall back.
    pub kv_page_slots: u32,
    /// WHICH REQUEST of a batched decode this op's KV belongs to.
    ///
    /// A prefill chunk's rows are consecutive positions of one sequence, so its cache writes share a
    /// page and differ only in slot — which `slot_no_fuse` already covers. A decode batch's rows are
    /// separate requests, so they differ in PAGE as well, and the page comes from a block table the
    /// runtime holds one of per request. That index is the one thing the runtime cannot derive from
    /// the op itself.
    ///
    /// 0 for every op of an unbatched bundle and for the first request of a batched one, so the
    /// single-request path is untouched.
    pub kv_request: u32,
    /// TRUE for a cache-write copy that writes a DISTINCT per-entry slot (the mq>1 PREFILL
    /// per-slot cachewr copies, baked at `qh·cap·hd + s·hd` for distinct `s`). Such copies
    /// MUST NOT be fused into a `GroupKind::Slot` group: the shim applies ONE `slot_pos`
    /// base-shift to a fused Slot group (decode semantics — every copy writes the SAME slot,
    /// baked at slot 0). Fusing distinct-slot copies collapses them (on-card: the prefill KV
    /// cache landed only a partial subset of slots). So a distinct-slot copy is classified
    /// [`GroupKind::SlotSolo`] → its OWN singleton group, launched with its baked slot doff
    /// (no shift, since prefill `slot_pos == 0`). Decode cachewr (same slot) leaves this
    /// `false` and keeps the fused-`Slot` launch reduction. Default `false`.
    pub slot_no_fuse: bool,
    /// HOST KV-CACHE WRITE (Route B). Set on ONE op per `AttnDecode` (the first cachewr).
    /// The shim does NOT run this op's program (nor the next [`kv_n_skip`] on-card cachewr
    /// copies): instead it host-scatters this token's roped K into the resident K cache in the
    /// **Kᵀ device layout the score matmul reads** (the on-card cachewr wrote it NATURAL
    /// slot-major, which the cap-sticked `[hd,cap]` kernel reads scrambled — the multi-day
    /// attention bug), and the V natural, then marks seg2 for re-H2D. Mirrors the `host_rmsnorm`
    /// pattern (op carries metadata; on-card op skipped). Default `false`.
    pub host_kv_write: bool,
    /// `2·nqh − 1`: the remaining on-card cachewr copies the shim skips after the tagged op.
    pub kv_n_skip: u32,
    /// Source roped new K/V (per-step activations) and the resident K/V cache tensor ids; the
    /// shim resolves them via `places["t{id}"]`. `kv_nqh/kv_nkvh/kv_hd/kv_cap` give the layout
    /// (the shim derives `gqa = nqh/nkvh` and writes K at the Kᵀ offset, V at the natural offset).
    pub kv_new_k_tid: u32,
    pub kv_new_v_tid: u32,
    pub kv_kc_tid: u32,
    pub kv_vc_tid: u32,
    pub kv_nqh: u32,
    pub kv_nkvh: u32,
    pub kv_hd: u32,
    pub kv_cap: u32,
}

impl EmittedOp {
    /// A by-value copy, for building a trimmed op list (see `ops_for_grouping`). `EmittedOp` is not
    /// `Clone` by design — copies of a baked op are almost always a mistake — so this is explicit.
    pub fn shallow_copy(&self) -> EmittedOp {
        EmittedOp {
            op_name: self.op_name.clone(),
            op: self.op.clone(),
            arg_bindings: self.arg_bindings.clone(),
            time: self.time,
            affine_strides: self.affine_strides.clone(),
            slot_stride_bytes: self.slot_stride_bytes,
            kv_page_fold: self.kv_page_fold,
            kv_batched_requests: self.kv_batched_requests,
            kv_fold_rows: self.kv_fold_rows,
            kv_page_slots: self.kv_page_slots,
            kv_request: self.kv_request,
            slab_write: self.slab_write,
            slab_stride_bytes: self.slab_stride_bytes,
            slot_no_fuse: self.slot_no_fuse,
            host_kv_write: self.host_kv_write,
            kv_n_skip: self.kv_n_skip,
            kv_new_k_tid: self.kv_new_k_tid,
            kv_new_v_tid: self.kv_new_v_tid,
            kv_kc_tid: self.kv_kc_tid,
            kv_vc_tid: self.kv_vc_tid,
            kv_nqh: self.kv_nqh,
            kv_nkvh: self.kv_nkvh,
            kv_hd: self.kv_hd,
            kv_cap: self.kv_cap,
        }
    }

    /// The SOLE constructor. When the op is time-tiled (time>1) it ASSERTS the
    /// **TiledHasStrideAndNode** witness: every tensor with a non-empty stride map
    /// has a matching AllocNode (so [`concrete_trips`] can bump its base address
    /// per trip) — a tiled tensor with a stride but no node is a build-time panic
    /// (internal-consistency bug), not a silently-wrong on-card address.
    fn new(
        op_name: String,
        op: SdscOp,
        // The SPEC's operands, for the buffer binding the wire form drops. Taken here rather than at each
        // call site so an op cannot exist without it.
        spec_args: &[AnyTensorArg],
        // The op's own iteration extents — the other half of [`materialized_bytes`]. Passed with the
        // args so a binding cannot exist without the footprint that sizes it.
        iter: &WorkPlan,
        time: u32,
        affine_strides: Vec<BTreeMap<&'static str, i64>>,
    ) -> EmittedOp {
        if time > 1 {
            for dsc_map in &op.dscs_ {
                for dsc in dsc_map.values() {
                    for (ti, strides) in affine_strides.iter().enumerate() {
                        if strides.is_empty() {
                            continue; // non-tiled / LX / INPUT tensor — OK.
                        }
                        assert!(
                            dsc.scheduleTree_.iter().any(|n| n.ldsIdx_ as usize == ti),
                            "EmittedOp {op_name}: tiled tensor {ti} has a non-empty stride but no \
                             allocate node to bump per trip (TiledHasStrideAndNode witness)"
                        );
                    }
                }
            }
        }
        let arg_bindings = spec_args
            .iter()
            .map(|a| {
                let v = a.view();
                ArgBinding {
                    buffer: v.name.to_string(),
                    is_input: v.is_input,
                    footprint_bytes: materialized_bytes(&v, iter),
                    touched: touched_blocks(&v, iter),
                }
            })
            .collect();
        EmittedOp {
            op_name,
            op,
            arg_bindings,
            time,
            affine_strides,
            slot_stride_bytes: 0,
            kv_page_fold: false,
            kv_batched_requests: false,
            kv_fold_rows: scratchy_subtile::sdsc_abstract::FoldRowRegime::WholeBatch,
            kv_page_slots: 0,
            kv_request: 0,
            slab_write: false,
            slab_stride_bytes: 0,
            slot_no_fuse: false,
            host_kv_write: false,
            kv_n_skip: 0,
            kv_new_k_tid: 0,
            kv_new_v_tid: 0,
            kv_kc_tid: 0,
            kv_vc_tid: 0,
            kv_nqh: 0,
            kv_nkvh: 0,
            kv_hd: 0,
            kv_cap: 0,
        }
    }
}

/// The SINGLE total lowering: typed [`OpSpec`] → wire [`SdscOp`]. Total (no
/// fallible path) — the witnesses already rejected every invalid input upstream
/// (at OpSpec-construction time). `op_name` keys the `dscs_`/`coreIdToDsc_` map.
/// All fold factors come from the one `folds` ([`SdscFoldSet`], witness (c)); the
/// `exUnit` comes from `op.op.ex_unit()` (witness #6); the `memOrg_` is the
/// sealed hbm/lx-only struct (no register file → no DtException 1535).
///
/// `sym_id_base` is the running monotonic negative-symbol-id counter for the
/// whole bundle: this op's first tiled symbol is `-(sym_id_base+1)`, and the
/// returned `EmittedOp` consumed `next_sym_id - sym_id_base` ids. Threaded by the
/// walk so negative ids are globally unique across ops (design risk #1).
pub fn emit_sdsc_tiled(
    op_name: &str,
    op: &OpSpec,
    folds: &SdscFoldSet,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    let time = op.time();
    let inner = emit_sdsc(op_name, op, folds, layout)?;
    if time == 1 {
        // Concrete-address single-shot path: all-empty strides, no unroll.
        let affine_strides = vec![BTreeMap::new(); op.args.len()];
        return Ok(EmittedOp::new(
            op_name.to_string(),
            inner,
            &op.args,
            &op.iter,
            1,
            affine_strides,
        ));
    }
    rewrite_op_for_time_tile(op_name, op, folds, inner, time, sym_id_base)
}

/// Rewrite a freshly-emitted (time=1-shaped) [`SdscOp`] into its TIME-TILED form:
///   * divide the `out` extent by `time` in `N_`, `dataStageParam_.ss_/el_`, and
///     each tiled tensor's `build_coordinates` (the `out` `size`/`elem_arr_1`);
///   * flip every TILED HBM AllocNode to `isStartAddrSymbolic_: Some(1)` with
///     per-core NEGATIVE sym-id strings in its `startAddressCoreCorelet_.data_`;
///   * compute `affine_strides` ({out: per_iter_out_elems * device_stride_out *
///     2}) for the tiled W/O tensors, `{}` for INPUT/LX, and register the base
///     symbols. Returns the validated [`EmittedOp`] (SymbolicWhenTiled).
fn rewrite_op_for_time_tile(
    op_name: &str,
    op: &OpSpec,
    folds: &SdscFoldSet,
    mut inner: SdscOp,
    time: u32,
    sym_id_base: &mut i64,
) -> Result<EmittedOp, SuperDscError> {
    let tiled_dim = op
        .time_tile
        .as_ref()
        .expect("rewrite_op_for_time_tile called on a non-tiled op")
        .dim()
        .name();
    let views: Vec<ArgView<'_>> = op.args.iter().map(|a| a.view()).collect();
    let cores = inner.numCoresUsed_.max(1);
    let eps = Fp16::ELEMS_PER_STICK as i64;
    let per_core_out = op.iter.per_core_extent(tiled_dim) as i64;
    let out_per_time = per_core_out / time as i64;

    // ── 1. Divide the `out` extent by `time` in N_ / ss_ / el_ ──────────────
    // N_ carries the per-core×time iteration extent for the tiled dim (the
    // dataStageParam_ ss_/el_ then read per-core // split; we make BOTH reflect
    // the per-time slice, matching compute_ops.py reading iteration_space AFTER
    // coarse_tile divided it). The FULL `out` is `extent`; per-time full = full /
    // time so the per-core stage = (full/time)/split = out_per_time.
    let full_out = op.iter.extent(tiled_dim) as i64;
    let per_time_full_out = full_out / time as i64;
    for dsc_map in inner.dscs_.iter_mut() {
        for dsc in dsc_map.values_mut() {
            set_iter_dim_i(&mut dsc.N_, tiled_dim, per_time_full_out);
            set_iter_dim_i(
                &mut dsc.dataStageParam_.get_mut("0").unwrap().ss_,
                tiled_dim,
                out_per_time,
            );
            set_iter_dim_i(
                &mut dsc.dataStageParam_.get_mut("0").unwrap().el_,
                tiled_dim,
                out_per_time,
            );
        }
    }

    // ── 2. Per-tensor: device stride + divided coordInfo (CONCRETE-UNROLL) ──────
    // The tiled path is emitted by PRE-UNROLLING the time loop into `time` CONCRETE
    // sub-ops at WRITE time (`concrete_trips`), each a proven time=1-format SdscOp
    // with per-trip startAddr = base + t·stride_bytes. NO symbols, NO scf.for —
    // this sidesteps dxp's always-on LoopUnroll (which clones identical symbol_ids
    // → DtException "Symbol already reserved"; see task #53). We keep the per-tensor
    // `affine_strides` (the per-trip byte advance the unroll adds) and the divided
    // `out` coordInfo; the AllocNode stays CONCRETE (isStartAddrSymbolic_ = None).
    let _ = (cores, sym_id_base, folds); // (no longer mints symbols; kept for sig parity)
    let mut affine_strides: Vec<BTreeMap<&'static str, i64>> = vec![BTreeMap::new(); views.len()];
    // Which tensors are tiled: HBM tensors whose layout contains `tiled_dim` (the
    // out stick dim). INPUT (A) does NOT carry `out`, so it stays un-bumped.
    for (ti, v) in views.iter().enumerate() {
        let tiled = !v.allocation.is_lx() && v.layout.contains(&tiled_dim);
        if !tiled {
            continue;
        }
        // DEVICE-STRIDE (design risk #4 — get it RIGHT or it is silently wrong).
        // We only KNOW the stride when the tiled dim is the CONTIGUOUS innermost
        // (stick) dim: consecutive `out`-windows are then physically adjacent in
        // HBM, so the per-trip byte advance is exactly `out_per_time * 2`
        // (device_stride_out = 1 element). If the tiled dim is NOT the tensor's
        // stick dim we CANNOT derive the stick-major stride with confidence —
        // FAIL LOUD (a build Err) rather than emit a wrong advance.
        if v.stick != tiled_dim {
            return Err(SuperDscError(format!(
                "{op_name}: tiled dim '{tiled_dim}' is NOT the stick (contiguous innermost) dim \
                 of tensor {ti} ('{}', stick='{}') — the stick-major device stride for a \
                 non-contiguous tiled dim is not derivable here without the SpyreTensorLayout \
                 device_size, and a wrong per-trip HBM advance is silently-wrong output. \
                 Refusing to guess (design risk #4).",
                v.name, v.stick
            )));
        }
        // DEVICE-STRIDE per `out` element for the per-trip HBM advance (design risk #4).
        // The byte distance between consecutive `out` windows depends on the tensor's
        // ON-DEVICE layout — and a RE-TILED KERNEL is NOT laid out the same as a flat
        // activation, so a single hardcoded stride is wrong for one of them:
        //   • KERNEL weight — RE-TILED to `[out/STICK, in, STICK]` (the out-tile axis is
        //     OUTERMOST), so advancing ONE `out` element crosses the whole inner tile =
        //     the product of its NON-out host extents (`in`=K for `[in,out]`). This is
        //     EXACTLY `DeviceTileLayout::per_core_stride_elems(out)` — the SAME stride the
        //     per-core address path uses (the matmul cos→1.0 fix). The OLD hardcoded `1`
        //     under-strided the only TIME-TILED matmul (the lm_head) by K× ⇒ every trip
        //     after the first re-read trip-0's weight columns ⇒ wrong upper-vocab logits
        //     ⇒ a degenerate, prompt-independent argmax.
        //   • OUTPUT / activation — FLAT `[total/STICK, STICK]`, row-major, so `out`'s
        //     stride is the product of host extents to its RIGHT (the phantom `y`=1 for a
        //     decode logits row ⇒ 1). Unchanged from before for the m=1 decode path.
        // GUARD: a non-KERNEL tiled tensor with a >1 PER-CORE extent to the LEFT of `out`
        // would need per-row tiling the flat row-major advance can't express — refuse
        // (build error) rather than silently mis-stride. ⭐ 2026-07-08: use PER-CORE extents,
        // not the full iteration extent — time-tiling operates on the PER-CORE work, and with
        // the batch=1 matmul now SPLITTING mb across cores (per_core mb=1), each core's tiled
        // output is a SINGLE row ⇒ the flat per-trip advance is valid (exactly the proven m=1
        // lm_head case). The old full-extent `op.iter.extent` false-fired on a split-mb matmul
        // whose per-core mb is 1 (the mq=8 prefill: mb-split shrinks out-parallelism ⇒ some wide
        // matmuls now LX-overflow → time-tile with per_core mb=1, which is safe).
        let oi = v.layout.iter().position(|&d| d == tiled_dim).unwrap_or(0);
        // Per-dim TRUE physical device extent (torch-spyre `arg.device_size`, `superdsc.py:447-469`):
        // prefer the tensor's OWN declared `device_extent` override when present — this tensor is a
        // slice/reuse of a LARGER physical allocation along `d` — else fall back to this op's own
        // iteration-space view (`op.iter.extent`), which is correct whenever the tensor is allocated
        // exactly to this op's shape (the overwhelmingly common, and until now ONLY, case).
        let phys_extent = |d: &'static str| -> i64 {
            let idx = v.layout.iter().position(|&ld| ld == d);
            let ov = idx.and_then(|i| v.device_extent.get(i).copied().flatten());
            ov.map(|p| p as i64)
                .unwrap_or_else(|| op.iter.extent(d) as i64)
        };
        // Per-trip HBM advance in BYTES for time-tiled tensor `ti`. `v.df` word length handles fp32
        // merge-partials (4 B) vs fp16 (2 B).
        let stride_bytes: i64 = if v.role == Role::Kernel {
            // KERNEL weight re-tiled `[out/STICK, in, STICK]` (out-tile OUTERMOST): advancing one `out`
            // element crosses the whole inner tile = product of its non-`out` host extents (`in`=K). ==
            // `DeviceTileLayout::per_core_stride_elems(out)`, the stride the per-core address path uses.
            let device_stride_out: i64 = v
                .layout
                .iter()
                .filter(|&&d| d != tiled_dim)
                .map(|&d| phys_extent(d))
                .product();
            out_per_time * device_stride_out * v.df.word_length() as i64
        } else {
            // OUTPUT/activation is STICK-BLOCKED on `out` (dev_off rank-2/3 stick_idx=1), NOT flat
            // row-major (the pre-bug#1 assumption). Advancing `out` by `out_per_time` jumps
            // `out_per_time * (stick-block row count)`, NOT `out_per_time * 1` — the row count is the FULL
            // extent LEFT of `out` (the device layout uses full dims; per-core mb is WRONG — it aliased at
            // mb>1). DERIVE the advance from the SystolicIR `StickLayout::dev_off` (the ONE addressing
            // path) so it cannot diverge from per_core_addr and is stick-block-correct at ANY row count.
            // Kani: time_tile_sticklayout_stride_tiles_disjoint (GREEN: advance == trip footprint ⇒
            // disjoint) + time_tile_flat_stride_aliases_at_multirow (RED: the old flat form aliases at
            // rows>1). The `tiled_trips_alias` bake guard remains the backstop.
            let full_rows = v.layout[..oi]
                .iter()
                .map(|&d| phys_extent(d) as usize)
                .product::<usize>()
                .max(1);
            let out_full = phys_extent(tiled_dim) as usize;
            // `row_blocked_df` (NOT the fp16-hardcoded `row_blocked`): the stick width baked into
            // `dev_off`'s block-advance formula must be THIS tensor's own format (`v.df` — e.g. an
            // fp32 K-split merge-partial output is 32-wide, fp8 128-wide), the same `view.df`
            // `per_core_addr` (2747+) threads through for the base address. Using the fp16-only
            // constructor here silently assumed a 64-wide stick for every tiled OUTPUT tensor,
            // mis-sizing the per-trip block advance for any non-fp16 tiled output — a real, distinct
            // per-tensor stride mismatch, matching torch-spyre's own rule (superdsc.py:447-469) that
            // every stride is derived from the tensor's OWN device format/size, never a shared default.
            let advance_elems = scratchy_subtile::sdsc_abstract::StickLayout::row_blocked_df(
                full_rows, out_full, v.df,
            )
            .dev_off(0, out_per_time as usize) as i64;
            advance_elems * v.df.word_length() as i64
        };
        let mut sm = BTreeMap::new();
        sm.insert(tiled_dim, stride_bytes);
        affine_strides[ti] = sm;

        // Divide the `out` coordInfo size (the AllocNode stays concrete — the
        // per-trip address bump happens in `concrete_trips` at write time).
        for dsc_map in inner.dscs_.iter_mut() {
            for dsc in dsc_map.values_mut() {
                if let Some(node) = dsc
                    .scheduleTree_
                    .iter_mut()
                    .find(|n| n.ldsIdx_ as usize == ti)
                {
                    divide_out_coordinate(&mut node.coordinates_, tiled_dim, out_per_time, eps);
                }
            }
        }
    }

    Ok(EmittedOp::new(
        op_name.to_string(),
        inner,
        &op.args,
        &op.iter,
        time,
        affine_strides,
    ))
}

/// Divide the `out`-dim `coordInfo` of a tiled AllocNode by `time`: the
/// gen_coord_info_value `size` (Affine alpha_ + elem_arr_0/elem_arr_1) for the
/// tiled stick dim must reflect the PER-TIME extent `out_per_time`, not the full
/// per-core extent. Mirrors compute_ops.py reading `iteration_space[dim] //
/// work_slices[dim]` AFTER coarse_tile divided the iteration space.
fn divide_out_coordinate(
    coords: &mut serde_json::Value,
    tiled_dim: &str,
    out_per_time: i64,
    eps: i64,
) {
    let Some(ci) = coords.get_mut("coordInfo").and_then(|c| c.as_object_mut()) else {
        return;
    };
    let Some(entry) = ci.get_mut(tiled_dim) else {
        return;
    };
    // The stick-dim variant (elemArr==2): rebuild it with the new per-time size.
    // size==alpha_[0] and the elem_arr_1 factor == size/eps (non-reduction stick).
    let nsplits = entry
        .get("folds")
        .and_then(|f| f.get("dim_prop_attr"))
        .and_then(|a| a.get(0))
        .and_then(|c| c.get("factor_"))
        .and_then(|f| f.as_i64())
        .unwrap_or(1);
    *entry = gen_coord_info_value(
        out_per_time,
        nsplits,
        eps,
        /*is_stick_dim=*/ true,
        /*is_stick_reduction=*/ false,
        /*group_stride=*/
        eps, // time-tiled rewrite is the natural-stride path (not a RowBlocked reduce)
    );
}

/// Like [`set_iter_dim`] but takes an already-i64 value (for the divided
/// per-time extents in the tiled rewrite).
fn set_iter_dim_i(it: &mut IterSpace, name: &str, v: i64) {
    match name {
        "mb" => it.mb_ = v,
        "out" => it.out_ = v,
        "in" => it.in_ = v,
        "x" => it.x_ = v,
        "y" => it.y_ = v,
        "i" => it.i_ = v,
        "j" => it.j_ = v,
        "ij" => it.ij_ = v,
        _ => {}
    }
}

/// Resolve one arg's HBM segment base + intra-segment byte offset. With a
/// [`BundleLayout`] (the real full-model walk) the arg's GLOBAL placement is looked
/// up by its `t{id}` name → `(SEGMENT_OFFSETS[role_segment], packed_offset + slice)`,
/// so every op references a shared tensor at the SAME address (task #55). Without a
/// layout (unit tests / single-op fixtures) it falls back to the per-op
/// `segment_base(arg_index)` (arg 0→seg0, …) — correct for a one-op bundle.
fn resolve_seg_base(
    layout: Option<&BundleLayout>,
    name: &str,
    arg_index: usize,
    offset_elems: u32,
    arg_bytes: u64,
    df: Df,
) -> Result<(u64, u64), SuperDscError> {
    let slice = offset_elems as u64 * df.word_length() as u64; // 4 B for fp32, 1 B fp8/int8, else 2 B fp16
    if let Some(l) = layout {
        // 1. A real SubtileIR tensor → its global placement. The id comes from the layout's own
        //    mint table, NOT from picking the digits back out of the spelling.
        if let Some(bundle::PlaceId::Act(tid)) = l.id_of(name)
            && let Some(p) = l.placements.get(&tid)
        {
            // The twin of the synth footprint guard below, for REAL tensors. Without it an op
            // whose baked offset outgrows its tensor silently writes into whatever was placed
            // next — which is how the per-head Kᵀ re-transpose walked into the adjacent layer's
            // cache every layer of every step for weeks before a bus fence. Compared against the
            // 128-B GRANULE the device reads in, so a sub-granule broadcast const (the `[1,1]`
            // scale) is not a false positive.
            if slice + arg_bytes.max(1) > align128(p.size).max(128) {
                return Err(SuperDscError(format!(
                    "t{tid}: access offset {slice}B + {arg_bytes}B exceeds its placement \
                         footprint {}B (seg{}) — the op addresses PAST its own tensor and would \
                         alias whatever is placed next. The offset arithmetic and the placement must \
                         come from the SAME layout authority.",
                    p.size, p.segment
                )));
            }
            return Ok((segment_base(p.segment)?, p.offset + slice));
        }
        // 2. A SYNTHETIC intermediate (silu/rmsnorm decomposition: `t{id}_silu`,
        //    `t{id}_sq`, …) — no `t{id}` placement. Assign it a STABLE offset in the
        //    Intermediate segment (seg3) on first sight, reused thereafter, so a
        //    producer and consumer of the same name resolve to the SAME address and
        //    it never collides with weights/activations.
        let seg = SegRole::Intermediate.segment();
        let mut s = l.synth.borrow_mut();
        let off = if let Some(&o) = s.map.get(name) {
            o
        } else {
            // ⛔⛔⛔ THERE IS NO LAZY ALLOCATION. THIS IS A BUILD ERROR.
            //
            // scratchy is a procmacro compiler: every intermediate this lowering invents is known
            // when it is invented, so a tensor that reaches its first ACCESS undeclared is a
            // lowering that forgot to say what it was making — not a case to serve.
            //
            // What serving it cost. The bump did not advance `next`, so every undeclared
            // synthetic received `align128(s.next)` — THE SAME ADDRESS — and nothing recorded a
            // size, so each got a zero-byte placement that hid the collision from every guard
            // downstream. On granite that put `t452_{rot,xc,rs}`, `t453_{rot,xc,rs}` and
            // `t461_silu` on one buffer: rope is `rot = x·P`, `xc = x·cos`, `rs = rot·sin`,
            // `out = xc + rs`, so three of its four intermediates were the same memory. A
            // zero-length placement is also what the device reads as `1 << 27` flits = 16 GiB.
            //
            // Declaring is one line at the site that mints the name (`BundleLayout::synth`), and
            // it is what gives the tensor its true footprint instead of one access's extent.
            //
            // ⚠️ These are STILL NOT COLOURED. The slot-colouring pass runs over the SubtileIR
            // DAG, and these tensors do not exist until lowering decomposes an op — so they get a
            // bump allocator of their own instead of liveness-based reuse. Declaring them is the
            // precondition for fixing that; it is not the fix.
            panic!(
                "synthetic '{name}' is accessed but was never declared to the layout. Every \
                 intermediate is known at bake — declare it with `BundleLayout::synth(id, dims)` \
                 at the site that mints the name. Serving this silently gave every undeclared \
                 tensor ONE shared address and a zero-byte placement."
            );
        };
        if let Some(&fp) = s.sizes.get(name) {
            // DECLARED synth (owns its full shape via BundleLayout::synth): its true footprint
            // is already reserved up front. GUARD: no access may exceed it — an out-of-footprint
            // write is a SHAPE MISMATCH that would alias the next tensor → `cargo build` Err
            // (the typed-shape contract). This is the principled cure for the under-reservation
            // that aliased sp/spprod/mxp on-card → mxp=0 → softmax overflow → garbage.
            if slice + arg_bytes.max(1) > fp {
                return Err(SuperDscError(format!(
                    "synth '{name}': access offset {slice}B + {arg_bytes}B exceeds its declared \
                     footprint {fp}B — shape mismatch (would alias the next intermediate). Fix the \
                     declared shape (BundleLayout::synth) or the op's view/offset."
                )));
            }
        } else {
            // UNDECLARED synth (rmsnorm/silu decomposition, not yet shape-typed): grow-on-access
            // high-water so a piecewise multi-write still reserves its true footprint (correct,
            // but UNCHECKED — these should be migrated to declared shapes).
            let end = align128(off + slice + arg_bytes.max(1));
            if end > s.next {
                s.next = end;
            }
        }
        return Ok((segment_base(seg)?, off + slice));
    }
    // No layout (single-op fixtures / unit tests): per-op segment by arg position.
    Ok((segment_base(arg_index)?, slice))
}

pub fn emit_sdsc(
    op_name: &str,
    op: &OpSpec,
    folds: &SdscFoldSet,
    layout: Option<&BundleLayout>,
) -> Result<SdscOp, SuperDscError> {
    let cores = folds.core_fold();
    let views: Vec<ArgView<'_>> = op.args.iter().map(|a| a.view()).collect();
    let out_idx = views.len().saturating_sub(1);
    // fp8 W8A8 matmul: a matmul/batchmatmul with an fp8 (SEN143_FP8) NON-OUTPUT operand. The DDL
    // `batchmatmulfp8` kernel (bmm.ddl:55) + its dataflow constraint require a DISTINCT operand layout
    // vs the fp16 kernel (verified against IBM's l0_tethering/fp8_32core_nonmx fixture): the fp8 WEIGHT
    // is a 2-D PACKED stick `[in:2, out:64]` (2 fp8 K-values × 64 N per 128-byte stick — HALF-word packed
    // along K), and the compute `dataFormat_` is SEN143_FP8. The activation (INPUT) is a flat fp8 128-K
    // stick and the OUTPUT stays fp16 (64-N stick) — those fall out of each arg's own `df`. Only the
    // KERNEL's 2-D pack + the compute format are special-cased here.
    let fp8_matmul = matches!(op.op, OpFunc::Matmul | OpFunc::BatchMatmul)
        && views
            .iter()
            .any(|v| !matches!(v.role, Role::Output) && matches!(v.df, Df::Fp8));
    let all_dims: Vec<&'static str> = op.iter.dims().iter().map(|d| d.name).collect();
    // Per-core work-slice indices (core → {dim → slice_idx}), the same map fed to
    // `coreIdToWkSlice_` — REAL per-core HBM addressing (#50) reads core c's slice
    // index along each split dim from it.
    let wk_slice = core_to_wk_slice(&op.iter, &all_dims);

    // ── N_ : the full iteration space (dense slots, -1 = unused). ──
    // Set ONLY the dims the op ACTUALLY has (op.iter.dims()). The old code blindly
    // set mb/out/in/x/y from op.iter.extent(), which returns 1 for an ABSENT dim →
    // a pointwise op ([mb,out,y]) got PHANTOM `in_:1, x_:1`. torch-spyre's REAL
    // emitted rsqrt SDSC (run via its own codegen, 2026-06-27) has N_={mb,out} with
    // NO in/x — the phantom unit dims are the divergence that makes dxp's SFP NR
    // refinement collapse to the seed (rsqrt→1/x). Matmul is unaffected: `in` is a
    // real matmul dim (set here), and x/i/j/ij come from the special-case below.
    let mut n_space = IterSpace::empty();
    for d in op.iter.dims() {
        set_iter_dim(&mut n_space, d.name, op.iter.extent(d.name));
    }
    // matmul N_ = ONLY the real iteration dims {mb, out, in} — NO i_/j_/ij_/x_ — matching torch-spyre's
    // generate_sdsc exactly (dxp SYNTHESIZES the output-tile counts in its coarse_tile pass, which
    // maxDimSizes_=-1 now enables). Setting i_ explicitly (tried 1 and per_core_mb) had NO effect on-card
    // — dxp ignores/overrides it; the real lever was maxDimSizes_=-1 (see the scheduleTree node). Leave
    // i_/j_/ij_ at -1 ⇒ skip_serializing ⇒ absent.

    // ── dataStageParam_["0"] : per-core steady-state (= full // split). ──
    // EXACTLY ONE stage ("core"). dxp's SdscCoreletSplit.cpp:70 asserts the INPUT SDSC
    // has `data_stage_params.size() == 1` and SYNTHESIZES the 2nd ("chunk") stage +
    // the 2-corelet fold itself (the post-split form is what every dxp fixture shows).
    // Emitting a 2nd stage here is a BUILD-time DtException (guard-every-crash). The
    // SFP rsqrt→1/x bug is NOT the missing chunk stage — dxp adds it downstream.
    let mut ss = IterSpace::empty();
    ss.name_ = "core";
    for d in op.iter.dims() {
        let per_core = (d.size / op.iter.split_of(d.name).max(1)).max(1);
        set_iter_dim(&mut ss, d.name, per_core);
    }
    let stage = StageParam {
        ss_: ss.clone(),
        el_: ss,
    };

    // The fused-epilogue's extra operand (if any): its Tensor{i} still gets a labeledDs_/primaryDsInfo_
    // entry below (it is a real tensor the on-card walk must address), but it is NOT one of THIS op's own
    // inputLabeledDs/outputLabeledDs — it belongs to the SECOND computeOp_ entry built after the loop.
    let epilogue_arg_idx: Vec<usize> = match op.op_info {
        OpInfo::FusedEpilogue {
            arg_idx, second, ..
        } => core::iter::once(arg_idx)
            .chain(second.map(|s| s.arg_idx))
            .collect(),
        _ => Vec::new(),
    };

    // ── labeledDs_ + primaryDsInfo_ (one entry per distinct Role). ──
    let mut labeled = Vec::with_capacity(views.len());
    let mut primary: BTreeMap<&'static str, LayoutInfo> = BTreeMap::new();
    let mut input_refs: Vec<String> = Vec::new();
    let mut output_refs: Vec<String> = Vec::new();
    for (i, v) in views.iter().enumerate() {
        labeled.push(LabeledDs {
            ldsIdx_: i as u32,
            dsName_: format!("Tensor{i}"),
            dsType_: v.role.ds_type(),
            scale_: v.scale.iter().map(|s| s.to_i64()).collect(),
            wordLength: v.df.word_length(),
            // Per-tensor dtype from the arg's typed `Df` (`v.df`, the SINGLE source of truth) — NOT a name
            // suffix. `Df::Fp32` → IEEE_FP32 (4-byte, the KSPLIT fp32-SFP-merge partials); `Df::Fp8` →
            // SEN143_FP8 (1-byte, the packed fp8 W8A8 path); `Df::Bf16` → BF16E; else SEN169_FP16. INERT for
            // the working build (every fp16 arg keeps `Df::Fp16`, so the numbers are byte-identical).
            // `Df::Bf16` is DORMANT (no arg carries it; the score path stays SEN169_FP16 — bf16-output matmul
            // is dxp-rejected, see lower_attn_node). The matmul PSUM accumulates in SEN169_FP16 (DeepTools
            // `bmm.ddl`: `%ptsum_fp`/`%pesum` are both `%type_fp16 = SEN169_FP16` — RCUDD1A has NO fp32 psum;
            // only `bmm_sen1p5.ddl` binds IEEE_FP32). NOTE: SEN169_FP16 is 1-6-9 (6-bit exp, bias 31) ⇒ max
            // finite ≈ 4.3e9, NOT 65504 (that is IEEE-fp16's 5-bit-exp max). Kani-proven
            // (`sen169_holds_attention_score_no_overflow`): a ~1e5 score is a NORMAL SEN169 value.
            dataFormat_: v.df.dataformat(),
            memOrg_: if v.allocation.is_lx() {
                MemOrg::lx_only()
            } else {
                MemOrg::hbm_lx()
            },
        });
        // interslicetranspose_fp16: the OUTPUT role carries the TRANSPOSE — its
        // stick is the 8×8 inter-slice block `["out","mb"]`/[8,8] (vs the INPUT's
        // plain `["out"]`/[64]). Every other op keeps the single-stick layout.
        let is_transpose_out = op.op == OpFunc::Transpose && matches!(v.role, Role::Output);
        // The fp8 WEIGHT (KERNEL of an fp8 matmul) is a 2-D PACKED stick: `[in:2, out:64]` — 64 N-columns
        // × 2 K-rows per 128-byte stick (2 fp8 packed per fp16-width slot along K). This is what the DDL
        // `batchmatmulfp8` dataflow constraint requires (KERNEL in-stick 2, out-stick 64); a flat
        // `out:128` fp8 kernel is "no suitable op mapping". The activation/output are single-stick (their
        // own `df`), so ONLY the fp8-matmul KERNEL takes this branch.
        let is_fp8_kernel = fp8_matmul && matches!(v.role, Role::Kernel);
        primary.entry(v.role.ds_type()).or_insert_with(|| {
            if is_transpose_out {
                LayoutInfo {
                    layoutDimOrder_: v.layout.to_vec(),
                    stickDimOrder_: vec!["out", "mb"],
                    stickSize_: vec![8, 8],
                }
            } else if is_fp8_kernel {
                LayoutInfo {
                    layoutDimOrder_: v.layout.to_vec(),
                    stickDimOrder_: vec!["in", "out"],
                    stickSize_: vec![2, 64],
                }
            } else {
                LayoutInfo {
                    layoutDimOrder_: v.layout.to_vec(),
                    stickDimOrder_: vec![v.stick],
                    stickSize_: vec![v.df.elems_per_stick()], // fp16/bf16: 64; fp32: 32; fp8/int8: 128
                }
            }
        });
        // computeOp_ wiring: `Tensor{i}-idx{i}`; the LAST arg is the output. The epilogue's own
        // operand is excluded here — the SECOND computeOp_ entry references it explicitly below.
        if epilogue_arg_idx.contains(&i) {
            continue;
        }
        let r = format!("Tensor{i}-idx{i}");
        if i == out_idx {
            output_refs.push(r);
        } else {
            input_refs.push(r);
        }
    }
    // A reduction accumulator is BOTH an input AND the output (matches the
    // reduce fixture: accum is inputLabeledDs[1] and the sole outputLabeledDs).
    if op.is_reduction && views.len() == 2 {
        input_refs.push(format!("Tensor{out_idx}-idx{out_idx}"));
    }

    // ── scheduleTree_ : one AllocNode per distinct dataspace (deduped by name). ──
    // REAL per-core HBM addresses (#50): each arg lives in its OWN 16 GiB HBM
    // segment (SEGMENT_OFFSETS[arg_index]), and within a segment core c's tile
    // starts at its work-slice byte offset. Distinct args never alias (distinct
    // segments); cores of the same OUTPUT never alias (distinct work-slices) — we
    // ASSERT both below.
    let mut tree: Vec<AllocNode> = Vec::new();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Collision guard: collect every (output) byte address emitted; any duplicate
    // is two cores writing the same byte = silently-wrong → build `Err`.
    let mut out_addr_seen: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
    for (i, v) in views.iter().enumerate() {
        let comp = v.allocation.component();
        let name = format!("allocate-Tensor{i}_{comp}");
        if !seen.insert(name.clone()) {
            continue;
        }
        // THE ONE device layout for this tensor view, computed ONCE and handed to BOTH the on-card WALK
        // (build_coordinates) and the per-core START (per_core_addr). Neither derives its own, so the walk
        // and start cannot address the tensor differently — a divergence is not representable (task #11).
        let sl = view_stick_layout(v, &op.iter);
        let coordinates = build_coordinates(v, &sl, &op.iter, fp8_matmul, op.is_reduction);
        // SLICE read offset (intra-tensor): a sliced operand (RoPE's x[half:]) reads
        // from `seg_base + offset_elems·wordLength`. `per_core_addr` adds the per-core
        // work-slice offset on top of this intra-segment base.
        // Full byte size of this arg's tensor (product of its layout extents · 2),
        // used to size a synthetic intermediate's seg3 slot when it has no global
        // placement (resolve_seg_base).
        // PHYSICAL footprint of this arg = product over dims of its MATERIALIZED extent. A dim
        // that is COLLAPSED for this arg (`scale ≠ Active`: a reduction OUTPUT dim, or a
        // BROADCAST input dim — out_broadcast/mb_broadcast) does NOT occupy its full iteration
        // extent: the stick axis collapses to ONE 64-stick; a non-stick collapsed dim to a single
        // row. Using the raw iteration extent over-counts these (e.g. an out-broadcast scalar
        // [nqh,1-stick] read across cap=256, or a per-head reduce accum), mis-sizing the synth
        // slot and false-firing the footprint guard. (`out_broadcast → out=RedStick`,
        // `mb_broadcast → mb=RedNonStick`, reduce accum's reduced dim = RedStick — see EwOperand.)
        let arg_bytes = materialized_bytes(v, &op.iter);
        let (seg_base, intra_base) =
            resolve_seg_base(layout, v.name, i, v.offset_elems, arg_bytes, v.df)?;
        // ── arrangement authority ── record THIS view's device layout for its tensor. If an earlier op
        // addressed the same tensor NON-equivalently (a producer/consumer Dense-vs-stick disagreement, the
        // M>1 scramble), this is a build Err naming the tensor — not silent garbage on-card. Declare only
        // for a FULL-tensor HBM data access: KERNELS re-tile via their RetileDescriptor (own layout); LX is
        // core-local scratch; a SLICED (`offset_elems`) or BROADCAST/reduced (non-`Active` scale) operand
        // is partial. And a BLOCK write (a GQA-replicate `[gqa,hd]` block at offset 0, a col-chunk) covers
        // LESS than the whole tensor even at offset 0 — its `arg_bytes` is below the tensor's declared
        // footprint, so it must NOT declare the canonical layout (else its `[gqa,hd]` view conflicts with
        // the whole-tensor `[nqh,hd]` read, a false positive). Only an access covering the full footprint
        // speaks for the tensor.
        if let Some(l) = layout {
            let active = v.offset_elems == 0 && v.scale.iter().all(|s| matches!(s, Scale::Active));
            // The tensor's FULL footprint: a real `t{id}` from its placement, a synth from the synth
            // registry. An op whose access is smaller than this is a partial (block/col-chunk) write and
            // does NOT speak for the canonical layout (else a `[1,8192]` logits-chunk write conflicts with
            // the full `[1,49664]` read). Unknown footprint (neither) ⇒ declare (whole-tensor default).
            let full_bytes = match l.id_of(v.name) {
                Some(bundle::PlaceId::Act(tid)) => l.placements.get(&tid).map(|p| p.size),
                _ => None,
            }
            .or_else(|| l.synth.borrow().sizes.get(v.name).copied());
            let full = full_bytes.is_none_or(|fp| arg_bytes >= fp);
            if active && full && !matches!(v.role, Role::Kernel) && !v.allocation.is_lx() {
                l.declare_arrangement(v.name, view_stick_layout(v, &op.iter))?;
            }
        }
        let per_core = per_core_addr(seg_base, v, &sl, &op.iter, &wk_slice, intra_base, cores)?;
        // No two cores of an OUTPUT tensor may share a byte address (aliased write) — EXCEPT a K-split
        // matmul, where the `k` cores each contract a K-slice into a PARTIAL product that dxp
        // PSUM-accumulates into the SHARED output tile (torch-spyre's reduction-dim split). That case is
        // legitimate ONLY when a dim the op REDUCES — split, and absent from THIS output's layout (`in`
        // for a matmul, whose output is `[mb,out]`) — is the one split. An `mb`/`out` shared output is
        // still a real collision, so the guard stays for every non-reduction split.
        let reduction_split = op.iter.split_of("in") > 1 && !v.layout.contains(&"in");
        if matches!(v.role, Role::Output)
            && !v.is_input
            && !v.allocation.is_lx()
            && !reduction_split
        {
            for &a in &per_core {
                if !out_addr_seen.insert(a) {
                    return Err(SuperDscError(format!(
                        "op {op_name}: OUTPUT tensor {i} ('{}') emits a COLLIDING per-core HBM \
                         address {a:#x} — two cores would write the same byte (silently-wrong). \
                         The work-division split must partition the output disjointly (#50).",
                        v.name
                    )));
                }
            }
        }
        let addr = AddrFold::new(&per_core, folds);
        tree.push(AllocNode {
            nodeType_: "allocate",
            name_: name,
            prev_: String::new(),
            ldsIdx_: i as u32,
            component_: comp,
            isStartAddrSymbolic_: None,
            // torch-spyre emits the HOST dim order verbatim (`[mb, in]`, mb-outermost) even for a
            // stick-major tensor — dxp applies `get_generic_stick_layout` INTERNALLY when it reconstructs
            // the stick-blocked `device_size` from `-1`, so the mb-outermost label + `-1` already yields the
            // stick-major walk (`element_torchspyre_parity`). Reordering here DIVERGES from the golden.
            layoutDimOrder_: v.layout.to_vec(),
            // ── The ON-CARD WALK, DERIVED from the SAME `sl` the per-core START uses (`StickLayout::
            // max_dim_sizes`) so the two cannot disagree. Stick-blocked (`RowBlocked`/`Kernel`) ⇒ `-1` per
            // dim (dxp reconstructs the RowBlocked device_size from N_/layoutDimOrder_/stickSize_ and
            // coarse-tiles M — torch-spyre parity, _create_sdsc_tensors:491; the prior matmul→ACTUAL-extents
            // left mb>1 rows partial on-card). A `Flat`/head-major start is row-major, NOT stick-blocked, so
            // it pins the ACTUAL extents instead — forcing `-1` there told the card to walk RowBlocked over a
            // row-major buffer (the producer-Flat / card-RowBlocked scramble). Decode (M=1) tiles trivially.
            // ⛔⛔⛔ THE OPERAND'S OWN DEVICE EXTENTS, NOT THE OP'S ITERATION EXTENTS.
            //
            // `TensorArg::with_device_extent` means "this op ITERATES a window of a LARGER
            // allocation, so derive strides from the allocation". The arrangement
            // (`view_stick_layout`) honours it and the per-core START (`per_core_addr`) honours it —
            // both through `DeviceExtents::of_view` — but this line built its own extent vector from
            // `op.iter.extent`, so a declaration moved OUR addresses and NEVER REACHED THE CARD.
            //
            // For a pinned (`Flat`/row-major) walk `maxDimSizes_` IS the walk dxp reconstructs, so a
            // windowed operand was walked at its ITERATION pitch. MEASURED: a head-batched
            // accumulator op declaring `mb = mq*hd/lanes` still emitted `max=[nqh, mq, lanes]`, i.e.
            // it strode `y` by `mq*lanes` where the token stream needs `mq*hd` — short by exactly the
            // slab count, inert at `hd == one stick`, silently wrong above it.
            //
            // Same "declared extent read in one place and dropped in another" hole `DeviceExtents`
            // exists to close; this was its third reader and it was missed. All three now derive from
            // ONE call, so a declaration cannot be honoured for the address and ignored for the walk.
            //
            // `Span::Swept` with NO declaration reproduces `op.iter.extent(d)` exactly, and no
            // operand in any shipped bundle declares one — so every existing bundle is byte-identical.
            maxDimSizes_: sl.device_walk(
                scratchy_subtile::sdsc_abstract::DeviceExtents::of_view(
                    &v.layout
                        .iter()
                        .map(|&d| op.iter.extent(d) as usize)
                        .collect::<Vec<_>>(),
                    &v.device_extent
                        .iter()
                        .map(|o| o.map(|p| p as usize))
                        .collect::<Vec<_>>(),
                    v.layout
                        .iter()
                        .position(|&d| d == v.stick)
                        .unwrap_or(usize::MAX),
                    &vec![true; v.layout.len()],
                    scratchy_subtile::sdsc_abstract::Span::Swept,
                )
                .dims()
                .iter()
                .map(|&e| e as i64)
                .collect::<Vec<_>>()
                .as_slice(),
            ),
            indirectAllocType_: "no_indirection",
            relatedIndirectAccessAlloc_: None,
            indexTensorType_: None,
            startAddressCoreCorelet_: addr,
            backGapCore_: None,
            coordinates_: coordinates,
        });
    }

    let constant_info = match op.op_info {
        OpInfo::SfpConstTable => sfp_constant_table(),
        OpInfo::ReduceScaling(packed) => scaling_factor_const(packed),
        OpInfo::ReduceScalingFp32(bits) => scaling_factor_const_fp32(bits),
        OpInfo::None | OpInfo::FusedEpilogue { .. } => serde_json::json!({}),
    };
    // torch-spyre's `generate_constant_info` returns the JSON *string* "{}" for the
    // EMPTY case and a DICT only when populated. VERIFIED on the REAL frontend (compiled
    // `torch.rsqrt` + `F.rms_norm`): rsqrt/add/mul → `"constantInfo_": "{}"` (a str), the
    // `mean` op's scaling_factor → a dict. The empty OBJECT `{}` is FALSY in dxp's Python
    // (`if constantInfo_:`), so the SFP constant-table / Newton-Raphson-refine setup is
    // SKIPPED — transcendentals collapse to their seed (rsqrt→1/x, sqrt→identity): the
    // 35×-magnitude rmsnorm bug, with the matmul (no SFP, no NR) unaffected. The truthy
    // string "{}" makes dxp run the setup so the NR refinement executes on-card.
    let constant_info = if constant_info.as_object().is_some_and(|o| o.is_empty()) {
        serde_json::Value::String("{}".to_string())
    } else {
        constant_info
    };

    // The fused epilogue's SECOND computeOp_ entry (when present): mirrors `sdsc_bmm_lxopt.json`'s
    // `MatMul_122` golden — it reads THIS op's own output plus the extra operand, and writes back
    // to the SAME output in place (no new HBM tensor, no separate trip). Built before `dsc` so both
    // entries share one `computeOp_` vec.
    // ⛔ ONE ENTRY PER STAGE, and each stage reads the PRIOR stage's own output — which is the same
    // labeled-ds in every case, since every stage writes the output in place. So the chain composes
    // by construction: stage 2 sees stage 1's sum without either naming the other.
    let epilogue_stage_compute_op =
        |arg_idx: usize, op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc| {
            let out_ref = format!("Tensor{out_idx}-idx{out_idx}");
            let epi_ref = format!("Tensor{arg_idx}-idx{arg_idx}");
            ComputeOp {
                exUnit: "sfp",
                opFuncName: op_func.name().to_string(),
                // Matches the golden's biasadd entry exactly: plain fp16, regular fidelity — the
                // epilogue's own operand is always fp16 (mask/bias/residual), independent of whatever
                // dtype the matmul itself ran in.
                attributes_: ComputeAttrs {
                    dataFormat_: Fp16::NAME,
                    fidelity_: "regular",
                },
                // The golden's biasadd entry carries `"location": "invalid"` (vs the matmul's "Inner") —
                // mirrored verbatim rather than guessed.
                location: "invalid",
                auxLoopName: None,
                isAtMainLoop: None,
                isAtTop: None,
                level: None,
                coreExclude: vec![],
                coreClExclude: vec![],
                opConsts: None,
                inputLabeledDs: vec![out_ref.clone(), epi_ref],
                interimLabeledDs: vec![],
                outputLabeledDs: vec![out_ref],
                indirectAccessIndexLabeledDs: vec![],
            }
        };
    let epilogue_compute_ops: Vec<ComputeOp> = match op.op_info {
        OpInfo::FusedEpilogue {
            arg_idx,
            op_func,
            second,
        } => core::iter::once(epilogue_stage_compute_op(arg_idx, op_func))
            .chain(second.map(|s| epilogue_stage_compute_op(s.arg_idx, s.op_func)))
            .collect(),
        _ => Vec::new(),
    };

    let dsc = Dsc {
        numCoresUsed_: cores,
        numCoreletsUsed_: folds.corelet_fold(),
        coreIdsUsed_: (0..cores).collect(),
        N_: n_space,
        coordinateMasking_: BTreeMap::new(),
        maskingConstId_: -1,
        dataStageParam_: BTreeMap::from([("0".to_string(), stage)]),
        primaryDsInfo_: primary,
        scheduleTree_: tree,
        labeledDs_: labeled,
        constantInfo_: constant_info,
        computeOp_: {
            let mut ops = vec![ComputeOp {
                exUnit: op.ex_unit().as_str(),
                // Dtype-suffixed matmul opFuncName per DDL bmm.ddl (batchmatmulfp8/int8, matmulfp8/int8): a
                // quantized matmul's OPERANDS carry the dtype (fp8/int8 via each arg's typed `Df`; its OUTPUT is
                // fp16, so read an operand, not the output). Same-dtype kernels only (the DDL pairs operands
                // same-dtype). fp16 matmul + every non-matmul op keep the bare `OpFunc::name`.
                opFuncName: {
                    let base = op.op.name();
                    if matches!(op.op, OpFunc::Matmul | OpFunc::BatchMatmul) {
                        match views
                            .iter()
                            .find(|v| !matches!(v.role, Role::Output))
                            .map(|v| v.df)
                        {
                            Some(Df::Fp8) => format!("{base}fp8"),
                            Some(Df::SenInt8) => format!("{base}int8"),
                            _ => base.to_string(),
                        }
                    } else {
                        base.to_string()
                    }
                },
                // fidelity "regular" matches BOTH working fixtures (sdsc_silu.json + MatMul_49)
                // — "high" is rejected by dxp. So fidelity is NOT the SFP rsqrt→1/x lever.
                attributes_: ComputeAttrs {
                    // The compute-op dataFormat marker. For the fp8 matmul it is SEN143_FP8 — the OPERAND
                    // format, matching IBM's l0_tethering fp8 fixture (the PSUM still accumulates in fp16 via
                    // bmm.ddl `%ptsum_fp`, but the `batchmatmulfp8` kernel is selected by this SEN143_FP8
                    // marker + the fp8 operand layouts; SEN169_FP16 here is "no suitable op mapping"). Else the
                    // dormant bf16 score path is BF16E, and every other op is SEN169_FP16 (byte-identical).
                    dataFormat_: if fp8_matmul {
                        <Fp8 as DataFormat>::NAME
                    } else if views.get(out_idx).is_some_and(|o| matches!(o.df, Df::Bf16)) {
                        "BF16E"
                    } else if views.first().is_some_and(|v| matches!(v.df, Df::Fp32)) {
                        // torch-spyre sets the computeOp data_format from args[0]
                        // (superdsc.py:882 `data_format=args[0].data_format`). The fp32 rmsnorm
                        // island's args[0] is IEEE_FP32; leaving this SEN169_FP16 made the fp32
                        // reduce a MAC whose fp16→fp32 result conversion DD2 refuses at lowering
                        // ("Unsupported result precision conversion", ConstructFMAInstr /
                        // SentientToProgIR/Utils.cpp:34). fp16/fp8/bf16 ops are byte-identical.
                        Df::Fp32.dataformat()
                    } else {
                        Fp16::NAME
                    },
                    fidelity_: "regular",
                },
                location: "Inner",
                // OMITTED to match torch-spyre's real frontend (see ComputeOp doc): emitting
                // an explicit auxLoopName="" suppressed the SFP Newton-Raphson refine (rsqrt→1/x).
                auxLoopName: None,
                isAtMainLoop: None,
                isAtTop: None,
                level: None,
                coreExclude: vec![],
                coreClExclude: vec![],
                opConsts: None,
                inputLabeledDs: input_refs,
                interimLabeledDs: vec![],
                outputLabeledDs: output_refs,
                indirectAccessIndexLabeledDs: vec![],
            }];
            ops.extend(epilogue_compute_ops);
            ops
        },
    };

    Ok(SdscOp {
        sdscFoldProps_: vec![FoldProp {
            factor_: folds.time_fold() as i64,
            label_: "time",
        }],
        sdscFolds_: SdscFolds {
            dim_prop_func: vec![FoldFunc::Affine {
                alpha_: 1,
                beta_: 0,
            }],
            dim_prop_attr: vec![FoldProp {
                factor_: folds.time_fold() as i64,
                label_: "time",
            }],
            data_: BTreeMap::from([("[0]".to_string(), "0".to_string())]),
        },
        coreFoldProp_: FoldProp {
            factor_: folds.core_fold() as i64,
            label_: "core",
        },
        coreletFoldProp_: FoldProp {
            factor_: folds.corelet_fold() as i64,
            label_: "corelet",
        },
        numCoresUsed_: cores,
        // coreIdToDsc_ MUST span exactly the cores the op USES (`cores`), NOT a fixed
        // MAX_CORES — torch-spyre uses `range(num_cores)` (compute_ops.py:527). With
        // MAX_CORES, an op using <32 cores lists phantom cores 18..31 here while
        // coreIdToDscSchedule (below) covers only 0..cores, so the multi-op stitcher
        // derives units for cores absent from the schedule → DtException "expecting a
        // valid entry in core schedule" (ModuleStitcher.cpp:216). Keep them consistent.
        coreIdToDsc_: (0..cores).map(|c| (c.to_string(), 0u32)).collect(),
        numWkSlicesPerDim_: all_dims.iter().map(|&d| (d, op.iter.split_of(d))).collect(),
        coreIdToWkSlice_: wk_slice,
        coreIdToDscSchedule: core_dsc_schedule(cores),
        dscs_: vec![BTreeMap::from([(op_name.to_string(), dsc)])],
    })
}

/// Set one named iteration dim in the dense [`IterSpace`] by slot name.
fn set_iter_dim(it: &mut IterSpace, name: &str, v: u32) {
    let v = v as i64;
    match name {
        "mb" => it.mb_ = v,
        "out" => it.out_ = v,
        "in" => it.in_ = v,
        "x" => it.x_ = v,
        "y" => it.y_ = v,
        "i" => it.i_ = v,
        "j" => it.j_ = v,
        "ij" => it.ij_ = v,
        _ => {}
    }
}

/// Mint a `RowBlocked` activation/output handle `[rows, cols]` for a matmul operand — the ONE runtime
/// kind-boundary (the token stream IS RowBlocked, so this never fails; a genuinely non-RowBlocked tensor
/// cannot be minted here, which is the point). Named `rb_*` for terse call sites. df is annotation-only
/// (the emit reads it from the OpSpec), so fp16 is fine for the fp8 activation too.
pub fn rb(name: &str, rows: u32, cols: u32) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(name, StickLayout::row_blocked(rows as usize, cols as usize))
        .expect("token-stream matmul operand is RowBlocked by construction")
}

/// Mint a `Flat` (head-major, row-major) output handle `[rows, cols]` — the per-head attention
/// pointwise/reduce path (`head_major`). Its `Stk<FlatTag>` type is what the builder reads to know the
/// op is head-major (a `RowBlocked` output ⇒ residual token stream). df annotation-only.
pub(crate) fn fl(name: &str, rows: u32, cols: u32) -> Stk<FlatTag> {
    Stk::<FlatTag>::new(name, StickLayout::flat(rows as usize, cols as usize))
        .expect("head-major operand is Flat by construction")
}

/// Output-handle helpers for the pointwise/reduce builders, where the `Stk` layout is ANNOTATION-ONLY —
/// the builder reads only `o.name()` and `O::kind()`, never the dims — so the output is minted by NAME.
/// `rbo` = RowBlocked (residual token stream), `flo` = Flat (head-major / per-head attention). The KIND
/// is the whole point: it's what the builder can't be handed wrong, and what drives head_major.
pub(crate) fn rbo(name: &str) -> Stk<RowBlockedTag> {
    rb(name, 1, 64)
}

/// A typed pointwise INPUT operand: a handle whose layout KIND is in its type (`K`), plus its broadcast
/// semantics. This is what makes input addressing FLOW — the per-arity `pw*` builders take `In<K>`, so
/// when a producer's typed output handle is passed as an input, its kind is carried by the type. Passing
/// a handle into a consumer that requires a fixed kind (a matmul activation must be RowBlocked) is a
/// `cargo build` type error if the handle is the wrong kind — end-to-end, not self-declared.
pub struct In<'a, K: KindTag> {
    h: &'a Stk<K>,
    mb_broadcast: bool,
    out_broadcast: bool,
    col_offset: u32,
}
#[allow(dead_code)] // ctors are used as the pointwise builders migrate to handle-flow (rmsnorm first)
impl<'a, K: KindTag> In<'a, K> {
    pub fn full(h: &'a Stk<K>) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: false,
            col_offset: 0,
        }
    }
    pub fn scalar(h: &'a Stk<K>) -> Self {
        In {
            h,
            mb_broadcast: true,
            out_broadcast: true,
            col_offset: 0,
        }
    }
    pub fn col(h: &'a Stk<K>) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: true,
            col_offset: 0,
        }
    }
    pub fn sliced(h: &'a Stk<K>, col_offset: scratchy_subtile::addr::DevOff) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: false,
            col_offset: col_offset.into_raw_elems(),
        }
    }
    /// [`col`] (per-row scalar, out-broadcast over the stick) PLUS a base `col_offset` — the per-head-block
    /// variant: a `[mqu,stk]` reduced scalar at `h·mqu·stk`, broadcast over the `[mqu,mqp]` score block.
    pub fn col_at(h: &'a Stk<K>, col_offset: scratchy_subtile::addr::DevOff) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: true,
            col_offset: col_offset.into_raw_elems(),
        }
    }
    /// Broadcast a `[1,cols]` const/gain over rows (the rmsnorm gamma / a mb-broadcast operand).
    pub fn mb(h: &'a Stk<K>) -> Self {
        In {
            h,
            mb_broadcast: true,
            out_broadcast: false,
            col_offset: 0,
        }
    }
    /// [`mb`] PLUS a base `col_offset` — the per-block prefix-mask slice (`pmask[.., b*stick..]`,
    /// broadcast over the mq new rows, one stick-wide block at a time).
    pub fn mb_at(h: &'a Stk<K>, col_offset: scratchy_subtile::addr::DevOff) -> Self {
        In {
            h,
            mb_broadcast: true,
            out_broadcast: false,
            col_offset: col_offset.into_raw_elems(),
        }
    }
    pub fn ew(&self) -> EwOperand<'a> {
        EwOperand {
            name: self.h.name(),
            mb_broadcast: self.mb_broadcast,
            out_broadcast: self.out_broadcast,
            col_offset: self.col_offset,
        }
    }
}

/// One-input pointwise `o = f(a)` with TYPED handles — the handle-flow entry point. The input kind `A`
/// flows from whatever produced `a`; the output `o`'s kind drives head_major (see
/// [`assemble_pointwise_broadcast_off`]). Byte-identical to the slice builder (it IS the slice builder).
#[allow(clippy::too_many_arguments)]
pub fn pw1<A: KindTag, O: KindTag>(
    op_name: &str,
    op_func: &'static str,
    // TYPED row/width slots — the handle-flow entry point demands the same named-door quantities
    // as `assemble_pointwise_broadcast_off`, so a caller states its extent's meaning here too.
    rows: scratchy_subtile::sdsc_abstract::RowCount,
    cols: scratchy_subtile::sdsc_abstract::BlockCols,
    a: In<A>,
    o: &Stk<O>,
    sym: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    assemble_pointwise_broadcast_off(
        op_name,
        op_func,
        rows,
        cols,
        &[a.ew()],
        o,
        scratchy_subtile::addr::DevOff::ZERO,
        sym,
        layout,
    )
}

/// Two-input pointwise `o = f(a, b)` with TYPED handles — each input carries its own kind.
#[allow(clippy::too_many_arguments)]
pub fn pw2<A: KindTag, B: KindTag, O: KindTag>(
    op_name: &str,
    op_func: &'static str,
    // TYPED row/width slots — same contract as `pw1`.
    rows: scratchy_subtile::sdsc_abstract::RowCount,
    cols: scratchy_subtile::sdsc_abstract::BlockCols,
    a: In<A>,
    b: In<B>,
    o: &Stk<O>,
    sym: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    assemble_pointwise_broadcast_off(
        op_name,
        op_func,
        rows,
        cols,
        &[a.ew(), b.ew()],
        o,
        scratchy_subtile::addr::DevOff::ZERO,
        sym,
        layout,
    )
}

/// UNARY dtype-convert (`dl16tofp32` / `fp32todl16`) opspec. Unlike [`pointwise_opspec`] — which marks EVERY
/// operand `Role::Output` ⇒ ONE shared `primaryDsInfo["OUTPUT"]` (one stick) — a convert's input and output
/// have DIFFERENT sticks (f16 64-elem vs fp32 32-elem). So the input is `Role::Input` and the output
/// `Role::Output`, giving TWO `primaryDsInfo` entries (INPUT@64, OUTPUT@32), each with its own dtype's stick
/// (`stick_elems_for` on the LabeledDs). Matches torch-spyre's per-arg dtype-op layout (superdsc.py:748) — the
/// fix for the KSPLIT `cvt` op's dxp `lit_dim_size` (the shared primary forced the fp32 output onto 64-stick,
/// disagreeing with its 4-byte wordLength).
/// The (input, output) dtypes intrinsic to a dtype-CONVERT op — the convert's whole purpose is to
/// CHANGE dtype, so the pair is a property of the [`OpFunc`], not of the operand names. Each arg's
/// [`Df`] drives its own `stickSize_`/`wordLength`/`dataFormat_`, giving the two distinct
/// `primaryDsInfo_` entries (input vs output stick) a convert needs — the typed replacement for the
/// old `_fp32`/`_fp8` name-suffix sniffing. A non-convert op is `(Fp16, Fp16)` (unreached — only
/// [`convert_opspec`] calls this, and only for the convert OpFuncs).
fn convert_dtypes(op: OpFunc) -> (Df, Df) {
    match op {
        OpFunc::Dl16ToFp32 => (Df::Fp16, Df::Fp32),
        OpFunc::Fp32ToDl16 => (Df::Fp32, Df::Fp16),
        // qfp8ch: f16 activation → SEN143_FP8 (E4M3). The DDL `matmulfp8` then consumes the fp8 stick.
        OpFunc::Qfp8ch => (Df::Fp16, Df::Fp8),
        _ => (Df::Fp16, Df::Fp16),
    }
}

fn convert_opspec(
    op: OpFunc,
    rows: u32,
    cols: u32,
    in_name: &str,
    o_name: &str,
) -> Result<OpSpec, String> {
    let (in_df, out_df) = convert_dtypes(op);
    // A dtype-CONVERT touches BOTH its input and output dataspaces per core, so the stick-axis split
    // must keep each core's slab a whole stick of the WIDER of the two formats (max elems/stick):
    // qfp8ch is f16→fp8, so the fp8 output's 128-lane stick binds (NOT the f16 input's 64), else a
    // core gets a 64-wide slab = HALF an fp8 stick → dxp `L3DlOpsScheduler:1070`. This df drives the
    // work-division stick basis; fp16→fp32 keeps 64 (fp16 wider than fp32's 32) ⇒ byte-identical.
    let stick_df = if out_df.elems_per_stick() >= in_df.elems_per_stick() {
        out_df
    } else {
        in_df
    };
    // Validate `cols` against the WIDER stick (128 for the fp8 output), not just fp16's 64: an fp8
    // convert whose cols is a 64- but not 128-multiple is a typed `Err` here, never a sub-stick fault.
    assert_df_stick_multiple(cols, stick_df)?;
    let cols_ext = StickExtent::<Fp16>::new(cols)?;
    let dims = vec![
        ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: cols_ext.elems(),
            is_reduction: false,
            is_stick: true,
            df: stick_df,
        },
        ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
    ];
    let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores)?;
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    let device_dims_r2 = plan.iter_syms(["mb", "out"]);
    // ── STICK-MAJOR (SCRATCHY_SUPERDSC_STICKMAJOR) ── the SAME m>1 seam as the residual pointwise/matmul:
    // at rows>1 the qfp8ch INPUT (the clamp `cl`) is written STICK-MAJOR (residual pointwise, rank-2) and
    // the OUTPUT `afp8` is read STICK-MAJOR by matmulfp8 (rank-2 fp8 activation), so present the convert
    // rank-2 [mb,out] too — else flat vs stick-major diverge at cols>stick and the fp8 activation is
    // scrambled. Each operand keeps its OWN df stick width: the fp16 input tiles on 64, the fp8 output on
    // 128 (`with_df` is per-operand; `for_view_df` then addresses each at that width). A convert is
    // whole-tensor (no offset), so no block-alignment is needed. OFF path = rank-3 (byte-identical).
    // qfp8ch (fp16→fp8, output stick 128 WIDER) works rank-2 stick-major. But dl16tofp32 (output stick 32
    // NARROWER than the 64 input) CRASHES dxp's per-dim buffer walk in rank-2 (gdb: getBufferCapacityForNode
    // PerDim indexes a size-3 dim list at -1) — the narrowing convert's rank-2 collapse drops the `y` dim dxp
    // needs. torch-spyre keeps dtype-converts RANK-3 (superdsc.py:756 "a type-conversion op requires an outer
    // spatial dim beyond the stick"). So keep qfp8ch stick-major; force the fp16↔fp32 converts rank-3.
    let is_fp32_convert = matches!(op, OpFunc::Dl16ToFp32 | OpFunc::Fp32ToDl16);
    let stickmajor = rows > 1 && !is_fp32_convert;
    let time_tile = plan
        .time_tile_for_lx(|p, opt| pointwise_lx_resident(p, 2, opt, Df::Fp16), "out")
        .map_err(|e| e.0)?;
    // The fp32 convert (rank-3) must ADDRESS RowBlocked so it reads/writes the RowBlocked residual byte-for-
    // byte (a rank-3 flat read row-mixes). Force the per-tensor ElementArrangement on BOTH operands (the fp16
    // side reads RowBlocked x; the fp32 side writes RowBlocked x32). Only for multi-row fp32 converts.
    let rb = is_fp32_convert && rows > 1;
    let mk = |is_input: bool, name: &str, role: Role, df: Df| -> Result<AnyTensorArg, String> {
        if stickmajor {
            Ok(AnyTensorArg::R2(
                TensorArg::<2>::new(
                    is_input,
                    name.to_string(),
                    role,
                    [Scale::Active, Scale::Active],
                    device_dims_r2,
                    ["mb", "out"],
                    "out",
                    Allocation::Hbm,
                )
                .map_err(|e| e.0)?
                .with_df(df)
                .with_row_blocked(rb),
            ))
        } else {
            Ok(AnyTensorArg::R3(
                TensorArg::<3>::new(
                    is_input,
                    name.to_string(),
                    role,
                    [Scale::Active, Scale::Active, Scale::Active],
                    device_dims,
                    ["mb", "out", "y"],
                    "out",
                    Allocation::Hbm,
                )
                .map_err(|e| e.0)?
                .with_df(df)
                .with_row_blocked(rb),
            ))
        }
    };
    let input = mk(true, in_name, Role::Input, in_df)?;
    let output = mk(false, o_name, Role::Output, out_df)?;
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    Ok(OpSpec {
        op,
        is_reduction: false,
        iter: plan,
        args: vec![input, output],
        op_info: OpInfo::None,
        tiled_symbols,
        time_tile,
    })
}

/// Assemble a dtype-convert op (see [`convert_opspec`]). Emit-time panic on a malformed spec = a `cargo build`
/// error (guard-every-crash), same as the other `assemble_*`.
#[allow(clippy::too_many_arguments)]
pub fn assemble_convert(
    op_name: &str,
    op_func: &str,
    rows: u32,
    cols: u32,
    in_: &Stk<RowBlockedTag>,
    o: &Stk<RowBlockedTag>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // TYPED: a dtype-convert's input and output are both the RowBlocked token stream (only the df
    // changes — fp16→fp32 / fp16→fp8); the handle carries only `.name()`, so the emit is byte-identical.
    let op = convert_opspec(op_func_from_str(op_func), rows, cols, in_.name(), o.name())
        .unwrap_or_else(|e| panic!("assemble_convert {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_convert {op_name}: {e}"))
}

/// Assemble a complete SuperDSC for ONE shape-preserving ELEMENTWISE op over a
/// `[rows, cols]` tile: `op_func` ∈ {"add","multiply","silu",…} (sfp unit), via
/// the typed [`pointwise_opspec`] builder + [`emit_sdsc`].
pub fn assemble_pointwise(
    op_name: &str,
    op_func: &'static str,
    rows: u32,
    cols: u32,
    in_names: &[&str],
    o_name: &str,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let mut sym_id_base: i64 = 0;
    assemble_pointwise_seeded(
        op_name,
        op_func,
        rows,
        cols,
        in_names,
        o_name,
        &mut sym_id_base,
        layout,
    )
}

/// [`assemble_pointwise`] with an explicit running symbol-id counter (risk #1).
#[allow(clippy::too_many_arguments)]
pub fn assemble_pointwise_seeded(
    op_name: &str,
    op_func: &'static str,
    rows: u32,
    cols: u32,
    in_names: &[&str],
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // head_major=false: every `assemble_pointwise_seeded` caller is a whole-tensor residual op
    // (MLP / fp8 quant / elementwise), so it is eligible for the stick-major m>1 seam fix.
    let op = pointwise_opspec(
        op_func_from_str(op_func),
        rows,
        cols,
        in_names,
        o_name,
        false,
    )
    .unwrap_or_else(|e| panic!("assemble_pointwise {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_pointwise {op_name}: {e}"))
}

/// Build the typed [`OpSpec`] for an `interslicetranspose_fp16` — a single-input,
/// single-output 8×8 inter-slice BLOCK TRANSPOSE on the PT unit (mirrors the golden
/// `sdsc_interslicetranspose.json`). The op iterates `[mb, out, y]` (= the INPUT
/// shape); `emit_sdsc` overrides the OUTPUT role's `stickDimOrder_`/`stickSize_` to
/// `["out","mb"]`/`[8,8]` (the transpose itself — the input keeps `["out"]`/`[64]`).
/// INPUT is `Role::Input`, OUTPUT is `Role::Output`, so two distinct
/// `primaryDsInfo_` entries exist (else the single `or_insert` would skip the
/// output override). Used to transpose the prefill chunk's new-K `[mq,hd]→[hd,mq]`
/// for the new-token `Q·Kᵀ` score block (decode mq=1 skips it — a free reinterpret).
fn transpose_opspec(mb: u32, out: u32, in_name: &str, o_name: &str) -> Result<OpSpec, String> {
    let out_ext = StickExtent::<Fp16>::new(out)?;
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
            size: out_ext.elems(),
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
    ];
    // LOWER to TileIR: same pattern as pointwise_opspec/reduce_opspec_df/the matmul builders.
    // LX-fit guard (#1535): a transpose's resident set is the 1 input + 1 output tile
    // (n_operands = 2), tiled over the `out` stick dim if it overflows.
    let tile_op = crate::tile_op::TileOp {
        kind: crate::tile_op::TileOpKind::PointwiseOrReduce { n_operands: 2 },
        dims: dims.clone(),
        df: Df::Fp16,
    };
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, distribute_cores, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    let input = TensorArg::<3>::new(
        true,
        in_name.to_string(),
        Role::Input,
        [Scale::Active, Scale::Active, Scale::Active],
        device_dims,
        ["mb", "out", "y"],
        "out",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?;
    let output = TensorArg::<3>::new(
        false,
        o_name.to_string(),
        Role::Output,
        [Scale::Active, Scale::Active, Scale::Active],
        device_dims,
        ["mb", "out", "y"],
        "out",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?;
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    Ok(OpSpec {
        op: OpFunc::Transpose,
        is_reduction: false,
        iter: plan,
        args: vec![AnyTensorArg::R3(input), AnyTensorArg::R3(output)],
        op_info: OpInfo::None,
        tiled_symbols,
        time_tile,
    })
}

/// Assemble a complete SuperDSC `interslicetranspose_fp16` transposing `in_name`
/// `[mb,out]` → `o_name` `[out,mb]`.
pub fn assemble_transpose(
    op_name: &str,
    mb: u32,
    out: u32,
    in_name: &str,
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op = transpose_opspec(mb, out, in_name, o_name)
        .unwrap_or_else(|e| panic!("assemble_transpose {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_transpose {op_name}: {e}"))
}

/// Build the typed [`OpSpec`] for a `restickify` — a DEPLOYED device dim-reorder
/// (vs the UNDEPLOYED `interslicetranspose`). Reorders `in_name` `[mb,out]`
/// (sticked on `out`) → `o_name` `[out,mb]` (sticked on `mb`): the SAME logical
/// data re-tiled so a downstream matmul reads it in the KERNEL `[in,out]` layout
/// WITHOUT a fused bmm-transpose (which SIGABRTs in dxp_standalone — IBM #1731).
/// torch-spyre inserts exactly this for a transposed matmul input
/// (test_padding.py: "restickify reorders x's device dims").
///
/// Used by in-bundle attention to turn the natural-written K cache `[cap,hd]`
/// (hd-sticked, the contiguous per-slot write layout) into the prefix-score
/// kernel layout `[hd,cap]` (cap-sticked) on-card, resident — no host, no
/// re-upload. The OUTPUT carries the reorder via its `layoutDimOrder_`
/// (`["out","mb"]`) + `stickDimOrder_` (`["mb"]`), distinct from the INPUT's
/// (`["mb","out"]` / `["out"]`); the normal emit branch handles distinct sticks
/// (no Transpose-style 8×8 override).
///
/// ⚠️ POD-CONFIRM (SDK `dscdefn.cpp opFuncsToString` + a restickify golden): the
/// exact `opFuncName`, `exUnit`, and the precise output layoutDimOrder/stickDimOrder
/// the deployed template expects. `mb` (the output stick) MUST be 64-aligned.
fn restickify_opspec(
    batch: u32,
    cap: u32,
    hd: u32,
    in_name: &str,
    o_name: &str,
) -> Result<OpSpec, String> {
    // Dim names + stick roles match the dxp-validated golden sdsc_restickify.json
    // EXACTLY: `out` = INPUT stick (carries hd), `y` = OUTPUT stick (carries cap),
    // `mb` = the preserved batch (nqh heads — golden had mb=1; we ride nqh here).
    // BOTH the input stick (hd→out) and output stick (cap→y) must be whole 64-sticks.
    let _out_ext = StickExtent::<Fp16>::new(hd)?;
    let _y_ext = StickExtent::<Fp16>::new(cap)?;
    // BOTH out (input stick) and y (output stick) are 64-granular split dims — the dual
    // stick. The golden splits both (out:2 × y:9 = 18 cores); marking only one leaves the
    // other with no stick-multiple candidate → scheduler "no valid candidate". (matmul
    // likewise has two stick dims: out + in.) mb (nqh) is the non-stick preserved batch.
    let dims = vec![
        ItDim {
            name: "mb",
            size: batch.max(1),
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: hd,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "y",
            size: cap,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
    ];
    let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores)?;
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    // INPUT: layout `[mb,out,y]` sticked on `out` (hd-sticked = the natural cache write
    // layout). OUTPUT: layout KEEPS `[mb,out,y]` but sticks on `y` (cap-sticked = the
    // transposed-kernel layout the score bmm reads). Same dim ORDER (batch `mb`=nqh
    // OUTERMOST ⇒ each head's [hd,cap] block is CONTIGUOUS), only the stick axis swaps
    // out→y — THE restickify. Batch-outermost is REQUIRED: the per-batch score bmm
    // ([`matmul_opspec_batched`]) reads kc_t as KERNEL `[y,in,out]`=[nqh,hd,cap] and
    // advances one `hd·cap` block per head; the OLD `[out,y,mb]` output put heads
    // INNERMOST (interleaved) → per-head bytes wrong (the 5-attempt failure root).
    let input = TensorArg::<3>::new(
        true,
        in_name.to_string(),
        Role::Input,
        [Scale::Active, Scale::Active, Scale::Active],
        device_dims,
        ["mb", "out", "y"],
        "out",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?;
    let output = TensorArg::<3>::new(
        false,
        o_name.to_string(),
        Role::Output,
        [Scale::Active, Scale::Active, Scale::Active],
        device_dims,
        ["mb", "out", "y"],
        "y",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?;
    Ok(OpSpec {
        op: OpFunc::Restickify,
        is_reduction: false,
        iter: plan,
        args: vec![AnyTensorArg::R3(input), AnyTensorArg::R3(output)],
        op_info: OpInfo::None,
        tiled_symbols: vec![],
        time_tile: None,
    })
}

/// Assemble a complete SuperDSC `restickify` reordering `in_name` `[mb,out]`
/// (sticked on `out`) → `o_name` `[out,mb]` (sticked on `mb`). See
/// [`restickify_opspec`]. The deployed on-card transpose primitive.
#[allow(clippy::too_many_arguments)]
pub fn assemble_restickify(
    op_name: &str,
    batch: u32,
    cap: u32,
    hd: u32,
    in_name: &str,
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op = restickify_opspec(batch, cap, hd, in_name, o_name)
        .unwrap_or_else(|e| panic!("assemble_restickify {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_restickify {op_name}: {e}"))
}

/// [`restickify_opspec`] with per-operand ELEMENT offsets — for the PER-HEAD restickify
/// (batch=1, head h's `[cap,hd]→[hd,cap]` block sliced via `in_off=h·cap·hd`,
/// `out_off=h·hd·cap`). The batched (mb=nqh) restickify mishandles per-head data on-card
/// (same class as the 3-D-kernel batched-matmul bug — PROVEN via the attn host oracle:
/// cos 0.72-0.84), so emit nqh single-head restickifies, mirroring the per-head matmul fix.
#[allow(clippy::too_many_arguments)]
fn restickify_opspec_off(
    batch: u32,
    cap: u32,
    hd: u32,
    in_name: &str,
    o_name: &str,
    in_off: u32,
    out_off: u32,
) -> Result<OpSpec, String> {
    let _out_ext = StickExtent::<Fp16>::new(hd)?;
    let _y_ext = StickExtent::<Fp16>::new(cap)?;
    let dims = vec![
        ItDim {
            name: "mb",
            size: batch.max(1),
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: hd,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "y",
            size: cap,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
    ];
    let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores)?;
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    let input = TensorArg::<3>::new(
        true,
        in_name.to_string(),
        Role::Input,
        [Scale::Active, Scale::Active, Scale::Active],
        device_dims,
        ["mb", "out", "y"],
        "out",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(in_off);
    let output = TensorArg::<3>::new(
        false,
        o_name.to_string(),
        Role::Output,
        [Scale::Active, Scale::Active, Scale::Active],
        device_dims,
        ["mb", "out", "y"],
        "y",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(out_off);
    Ok(OpSpec {
        op: OpFunc::Restickify,
        is_reduction: false,
        iter: plan,
        args: vec![AnyTensorArg::R3(input), AnyTensorArg::R3(output)],
        op_info: OpInfo::None,
        tiled_symbols: vec![],
        time_tile: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn assemble_restickify_off(
    op_name: &str,
    batch: u32,
    cap: u32,
    hd: u32,
    in_name: &str,
    in_off: u32,
    o_name: &str,
    out_off: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op = restickify_opspec_off(batch, cap, hd, in_name, o_name, in_off, out_off)
        .unwrap_or_else(|e| panic!("assemble_restickify {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_restickify {op_name}: {e}"))
}

/// PATH B (host_kv_write nuke): a genuine RANK-2 per-head restickify that TRANSPOSES the natural
/// K cache `[cap,hd]` (flat row-major, the on-card `[1,hd]`-per-slot dense write) into the `[hd,cap]`
/// cap-sticked Kᵀ KERNEL the score matmul reads — mirroring torch-spyre `key.transpose(-2,-1).
/// contiguous()`. The rank-3 `restickify_opspec_off` CANNOT do this: `DeviceTileLayout::device_size`
/// yields the sticked-kernel tiling `[b/64,a,64]` ONLY for `len==2 && stick_idx==1`, so a rank-3
/// tensor is FLAT for BOTH operands ⇒ no transpose (the historic cos≈0). Here:
///   • INPUT  logical `[cap,hd]`, stick = `hd` (dim 1 = LAST, dxp-friendly) ⇒ device_size `[hd/64,cap,64]`
///     ⇒ reads K[slot][d] at `dev_off([cap,hd],1,·)` — EXACTLY the SLAB-MAJOR store the on-card cachewr
///     writes (`[hd/64,cap,64]`, one `[1,64]` slab-copy per token-slot; at hd==64 = the flat `slot*hd+d`).
///   • OUTPUT logical `[hd,cap]`, stick = `cap` (dim 1 = LAST) ⇒ device_size `[cap/64,hd,64]` ⇒ writes
///     K[slot][d] at `kcache_kt_write_offset(slot,d)` = `dev_off([hd,cap],1,[d,slot])`, the score read.
/// Proven by `natural_store_plus_restickify_produces_kt_layout` (interp) + the `pathb_k_*` Kani chain.
/// `datastageBasedElemOff` (true ONLY for ReStickify, ddcv1.cpp:3630) drives the element offsets from
/// these per-operand datastage tilings, so the transpose is realized on-card. Emitted PER HEAD (offsets).
fn restickify_kt_opspec_2d(
    cap: u32,
    hd: u32,
    in_name: &str,
    o_name: &str,
    in_off: u32,
    out_off: u32,
) -> Result<OpSpec, String> {
    // Both extents must be whole 64-sticks: cap is the OUTPUT stick, hd is the (padded) free dim.
    let _cap_ext = StickExtent::<Fp16>::new(cap)?;
    let _hd_ext = StickExtent::<Fp16>::new(hd)?;
    // INPUT sticks on its LAST dim `hd` (dxp requires the stick on the last dim — a stick-on-dim0
    // crashed dxp with std::out_of_range map::at, on-card 2026-07-05). The stick-last input reads
    // `dev_off([cap,hd],1,·)`, which is EXACTLY the SLAB-MAJOR store the on-card cachewr now writes
    // (`[hd/stick,cap,stick]`, one `[1,stick]` slab-copy per token-slot) — so it reads the resident
    // cache correctly at ANY head_dim. At hd==stick the slab-major store degenerates to the flat
    // `slot*hd+d`, so granite-3.3-2b/llama-1b stay byte-identical; hd>stick (granite-3.3-8b/llama-3.2-3b
    // hd=128, gemma hd=256/512) reads the multi-slab store. Kani: restickify_opspec_realizes_pathb_contract.
    // Iteration vocabulary = the two logical axes, named with CANONICAL dim names so the frontend
    // serializes their extents into N_/dataStageParam_'s fixed `out_`/`y_` slots. `cap` -> "y",
    // `hd` -> "out" (the 3D restickify's mb/out/y convention: out=hd, y=cap). Non-canonical names
    // ("cap"/"hd") left N_ EMPTY ({name_:"n"}) ⇒ dxp's SuperDsc::importJsonStr `.at()` aborted with
    // std::out_of_range map::at at JSON import (on-card 2026-07-05). Both marked stick so the splitter
    // finds a valid stick-multiple candidate on each operand.
    let dims = vec![
        ItDim {
            name: "y",
            size: cap,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: hd,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
    ];
    let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores)?;
    // INPUT: resident cache view `[y=cap, out=hd]` sticked on `out` (=hd, dim 1 = LAST, dxp-friendly) ⇒
    // device_size `[hd/64,cap,64]` = the SLAB-MAJOR store the cachewr writes (`dev_off([cap,hd],1,·)`) ⇒
    // reads K[slot][d] correctly at ANY head_dim (at hd==64 the single slab IS the flat `slot*hd+d`).
    // `dev_off` keys on stick POSITION + sizes (not names), so this is identical addressing to pre-rename.
    let input = TensorArg::<2>::new(
        true,
        in_name.to_string(),
        Role::Input,
        [Scale::Active, Scale::Active],
        plan.iter_syms(["y", "out"]),
        ["y", "out"],
        "out",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(in_off);
    // OUTPUT: Kᵀ kernel view `[out=hd, y=cap]` sticked on `y` (=cap, dim 1 = LAST) ⇒ device_size
    // `[cap/64,hd,64]` = the exact cap-sticked layout the score matmul reads (kcache_kt_write_offset).
    let output = TensorArg::<2>::new(
        false,
        o_name.to_string(),
        Role::Output,
        [Scale::Active, Scale::Active],
        plan.iter_syms(["out", "y"]),
        ["out", "y"],
        "y",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(out_off);
    Ok(OpSpec {
        op: OpFunc::Restickify,
        is_reduction: false,
        iter: plan,
        args: vec![AnyTensorArg::R2(input), AnyTensorArg::R2(output)],
        op_info: OpInfo::None,
        tiled_symbols: vec![],
        time_tile: None,
    })
}

/// Assemble ONE per-head Kᵀ restickify (natural `[slots,feats]` → slot-sticked `[feats,slots]`).
/// See [`restickify_kt_opspec_2d`]. The tile's two extents arrive as
/// [`KtTileSlots`](scratchy_subtile::sdsc_abstract::KtTileSlots)/[`KtTileFeats`](scratchy_subtile::sdsc_abstract::KtTileFeats) —
/// each door names which quantity fills it, so the pair cannot be handed over swapped.
/// `in_off`/`out_off` are the per-head element offsets into the natural cache (`head·cap·hd`) and
/// the Kᵀ scratch (`head·hd·cap`).
#[allow(clippy::too_many_arguments)]
pub fn assemble_restickify_kt_2d(
    op_name: &str,
    slots: scratchy_subtile::sdsc_abstract::KtTileSlots,
    feats: scratchy_subtile::sdsc_abstract::KtTileFeats,
    in_name: &str,
    in_off: scratchy_subtile::addr::DevOff,
    o_name: &str,
    out_off: scratchy_subtile::addr::DevOff,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let in_off = in_off.into_raw_elems();
    let out_off = out_off.into_raw_elems();

    let op = restickify_kt_opspec_2d(
        slots.extent(),
        feats.extent(),
        in_name,
        o_name,
        in_off,
        out_off,
    )
    .unwrap_or_else(|e| panic!("assemble_restickify_kt_2d {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_restickify_kt_2d {op_name}: {e}"))
}

/// PATH B (V side): a RANK-2 per-head restickify that RE-STICKS the natural V cache `[cap,hd]` (flat
/// row-major, the on-card `[1,hd]`-per-slot write) into the hd-sticked `[cap,hd]` KERNEL the value bmm
/// reads. UNLIKE K this is NOT a transpose — same dim order `[cap,hd]`, only the stick axis moves
/// flat→hd (dim0→dim1). The value bmm `out_pre = probs[1,cap]·V[cap,hd] → [1,hd]` reads V as kernel
/// `[k=cap, n=hd]` sticked on n=hd ⇒ device `[hd/64,cap,64]` = `vcache_write_offset`. Here:
///   • INPUT  logical `[cap,hd]`, stick = `cap` (dim 0, NOT last) ⇒ device_size FLAT ⇒ reads V[slot][d]
///     at `slot*hd+d` — EXACTLY the natural write (`dev_off([cap,hd],0,·)`).
///   • OUTPUT logical `[cap,hd]`, stick = `hd` (dim 1 = LAST) ⇒ device_size `[hd/64,cap,64]` ⇒ writes
///     V[slot][d] at `vcache_write_offset(slot,d)` = `dev_off([cap,hd],1,[slot,d])`, the value-bmm read.
/// Proven by `restickify_v_opspec_realizes_contract{,_hd128}` (Kani) + the `pathb_v_*` chain.
fn restickify_v_opspec_2d(
    cap: u32,
    hd: u32,
    in_name: &str,
    o_name: &str,
    in_off: u32,
    out_off: u32,
) -> Result<OpSpec, String> {
    let _cap_ext = StickExtent::<Fp16>::new(cap)?;
    let _hd_ext = StickExtent::<Fp16>::new(hd)?;
    let dims = vec![
        ItDim {
            name: "cap",
            size: cap,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
        ItDim {
            name: "hd",
            size: hd,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
    ];
    let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores)?;
    // INPUT: natural V view `[cap,hd]` sticked on `cap` (dim 0) ⇒ NOT (len==2 && stick_idx==1) ⇒ FLAT
    // row-major `slot*hd+d` = the natural per-slot dense write.
    let input = TensorArg::<2>::new(
        true,
        in_name.to_string(),
        Role::Input,
        [Scale::Active, Scale::Active],
        plan.iter_syms(["cap", "hd"]),
        ["cap", "hd"],
        "cap",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(in_off);
    // OUTPUT: value-bmm kernel view `[cap,hd]` sticked on `hd` (dim 1 = LAST) ⇒ device_size
    // `[hd/64,cap,64]` = the exact hd-sticked layout the value bmm reads (vcache_write_offset).
    let output = TensorArg::<2>::new(
        false,
        o_name.to_string(),
        Role::Output,
        [Scale::Active, Scale::Active],
        plan.iter_syms(["cap", "hd"]),
        ["cap", "hd"],
        "hd",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(out_off);
    Ok(OpSpec {
        op: OpFunc::Restickify,
        is_reduction: false,
        iter: plan,
        args: vec![AnyTensorArg::R2(input), AnyTensorArg::R2(output)],
        op_info: OpInfo::None,
        tiled_symbols: vec![],
        time_tile: None,
    })
}

/// Assemble ONE per-head V re-stickify (natural `[cap,hd]` → hd-sticked `[cap,hd]`). See
/// [`restickify_v_opspec_2d`].
#[allow(clippy::too_many_arguments)]
pub fn assemble_restickify_v_2d(
    op_name: &str,
    cap: u32,
    hd: u32,
    in_name: &str,
    in_off: u32,
    o_name: &str,
    out_off: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op = restickify_v_opspec_2d(cap, hd, in_name, o_name, in_off, out_off)
        .unwrap_or_else(|e| panic!("assemble_restickify_v_2d {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_restickify_v_2d {op_name}: {e}"))
}

/// Map a `&str` op-func (the existing `assemble_*` call convention) to the sealed
/// [`OpFunc`]. Unknown names default to `Add` (the SubtileIR walk only ever
/// passes the handled set; this keeps the legacy string API).
/// Internal op-name → [`OpFunc`]. PANICS on an unknown key (a build-time failure
/// at proc-macro/test time) rather than silently defaulting to `Add` — a typo'd
/// op name is a wrong-op (silently-wrong output), so it must NOT be swallowed.
/// `"add"`/`"mul"` are accepted (`"mul"` is an alias for `"multiply"`).
pub(crate) fn op_func_from_str(s: &str) -> OpFunc {
    match s {
        "matmul" => OpFunc::Matmul,
        "batchmatmul" => OpFunc::BatchMatmul,
        "add" => OpFunc::Add,
        "sub" | "subtract" => OpFunc::Subtract,
        "multiply" | "mul" => OpFunc::Multiply,
        "realdiv" => OpFunc::RealDiv,
        "abs" => OpFunc::Abs,
        "silu" => OpFunc::Silu,
        "exp" => OpFunc::Exp,
        "reciprocal" => OpFunc::Reciprocal,
        "sqrt" => OpFunc::Sqrt,
        "rsqrt" => OpFunc::Rsqrt,
        "sigmoid" => OpFunc::Sigmoid,
        "dl16tofp32" => OpFunc::Dl16ToFp32,
        "fp32todl16" => OpFunc::Fp32ToDl16,
        "gelu" => OpFunc::Gelu,
        "mish" => OpFunc::Mish,
        "tanh" => OpFunc::Tanh,
        "sum" => OpFunc::Sum,
        "max" => OpFunc::Max,
        "mean" => OpFunc::Mean,
        "identity" => OpFunc::Identity,
        "maximum" => OpFunc::Maximum,
        "minimum" => OpFunc::Minimum,
        "qfp8ch" => OpFunc::Qfp8ch,
        other => panic!(
            "op_func_from_str: unknown op name {other:?} — would have silently become `add` \
             (wrong op). Add an explicit arm or fix the caller."
        ),
    }
}

/// Op kind for MEDIUM-GRAIN fusion grouping (torch-spyre's Inductor-kernel
/// granularity, mirrored SDSC-native): a run of `Pure` ops (rmsnorm/matmul/
/// elementwise — no per-entry shim interception) FUSES into ONE concrete dxp
/// bundle of up to `g` trips; ops the shim intercepts per-manifest-entry must
/// stay SINGLETON so the interception lands on the right entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupKind {
    /// Fusable on-card trip (no per-entry shim handling).
    Pure,
    /// `host_kv_write`: the shim host-scatters KV then does `oi += skip` to skip
    /// the on-card cachewr copies it replaced. This entry AND those `skip`
    /// entries MUST be singletons so `oi += skip` skips exactly them.
    HostKv { skip: usize },
    /// `slot_write` DECODE cachewr: every fused copy writes the SAME slot (baked at slot 0,
    /// shim shifts the group base by `slot_pos·stride` ONCE). Consecutive `Slot` trips FUSE
    /// ≤ g (the launch reduction). MUST all share one slot — enforce via [`GroupKind::SlotSolo`]
    /// for the distinct-slot case, so a fused `Slot` group is uniform-slot by construction.
    ///
    /// CARRIES ITS REQUEST, and a run BREAKS where the request changes. A group is one launch and a
    /// launch resolves ONE request's page table and write cursor, so two requests cannot share a
    /// group — but a request's own kv-head copies still can, and they must: a batched decode step's
    /// cache writes are `nkvh · requests · 2` per layer, and making each its own singleton (which is
    /// what treating them as distinct-slot did) cost ~5,300 extra launches per forward and ran a
    /// batch of 8 twenty times slower than one request. 0 for an unbatched bundle, so its grouping
    /// is exactly what it was.
    Slot { req: u32 },
    /// `slot_write` PREFILL cachewr: writes a DISTINCT per-entry slot (baked at `s·hd`). The
    /// shim's single per-group slot-shift CANNOT express distinct slots, so each is its OWN
    /// singleton group (launched with its baked slot doff; prefill `slot_pos==0` ⇒ no shift).
    /// This makes "distinct-slot copy fused into a uniform-shift group" UNCONSTRUCTABLE.
    SlotSolo,
    /// PAGE FOLD (`kv_page_fold`): folds one page of resident prefix. Consecutive fold trips FUSE
    /// (they share a page base) but must never fuse with anything else — the shim re-launches this
    /// group per page, and a stray op swept in would re-run per page too, re-seeding the new-token
    /// block or re-adding the residual once per page.
    PageFold,
    /// `slab_write` DECODE incremental Kᵀ restickify (kill-restickify Stage 2, hd==64): re-transposes
    /// ONLY the current 64-slot slab; the shim shifts the group's seg2 base by
    /// `(slot_pos/64)·slab_stride_bytes` ONCE (each op baked at slab 0). Consecutive `Slab` trips FUSE
    /// ≤ g (the nkvh per-kv-head restickifies, all one `slab_stride_bytes`). DISTINCT kind from `Slot`:
    /// its stride (STICK_BYTES·stick = 8192) ≠ the cachewr's `slot_stride_bytes` (128), so fusing the two
    /// would trip the per-group uniform-stride assert (mixed-stride group → wrong base shift).
    Slab,
}

/// WHICH REQUEST A TRIP'S ADDRESSING BELONGS TO.
///
/// Wraps the op's `kv_request` so the group walk cannot silently compare it against something else
/// (a slot index, a page, a `rep`) — every one of those is also a small integer, and this file's
/// history is a list of exactly that mistake. `0` is both "request 0" and "untagged", deliberately:
/// an unbatched bundle must stay byte-identical, so the two are the same value. That conflation is
/// SAFE HERE and nowhere else — for a group BOUNDARY, treating an untagged trip as request 0 can
/// only break a run that would otherwise have fused, and over-breaking costs launches while
/// under-breaking costs correctness. (Where the conflation is NOT safe is the page base and write
/// cursor; those live behind `fold_plan::RequestSlot`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TripRequest(pub u32);

/// A body trip: WHAT it is plus WHOSE it is. The pair travels together so the fusion walk cannot
/// consult one without the other.
///
/// Before this type the walk saw only `GroupKind`, and only the `Slot` variant happened to carry a
/// request. So every OTHER kind fused across requests — and a group is ONE launch, which resolves
/// ONE request's page table, so a `Pure` run spanning two requests gave both of them the FIRST
/// request's KV page. That is the same class of bug as the six before it: correctness depended on
/// each variant separately remembering to check the request, and a new variant (or an op newly
/// tagged with a request, like the per-page Kᵀ re-transpose) got it wrong by default.
/// Now the request is checked ONCE, for all kinds, in [`Trip::fusable_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Trip {
    pub kind: GroupKind,
    pub req: TripRequest,
}

impl GroupKind {
    /// Whether two trips of these kinds may share one concrete bundle, IGNORING the request.
    ///
    /// Note `SlotSolo` and `HostKv` are not fusable even with THEMSELVES — that is how "this trip is
    /// a singleton by construction" is expressed, and it is what makes the walk's advance-by-one
    /// fallback unnecessary: a run is `[start, start+1)` unless the kind opts in.
    fn fusable_kind_with(self, other: GroupKind) -> bool {
        match (self, other) {
            (GroupKind::Pure, GroupKind::Pure) => true,
            (GroupKind::Slab, GroupKind::Slab) => true,
            (GroupKind::PageFold, GroupKind::PageFold) => true,
            (GroupKind::Slot { req: a }, GroupKind::Slot { req: b }) => a == b,
            // SlotSolo: distinct baked slot per entry — one per-group shift cannot express two.
            // HostKv: the shim's `oi += skip` must land on this exact entry.
            _ => false,
        }
    }
}

impl Trip {
    pub fn new(kind: GroupKind, req: TripRequest) -> Trip {
        Trip { kind, req }
    }

    /// THE ONE RULE THAT DECIDES EVERY GROUP BOUNDARY: same request, and kinds that fuse.
    ///
    /// A group is one launch; a launch resolves one request's page table and one write cursor and
    /// shifts the segment base ONCE. Two requests therefore cannot share a group whatever their
    /// kinds — so the request test lives HERE, above the kind test, and applies to kinds that do not
    /// exist yet.
    fn fusable_with(&self, other: &Trip) -> bool {
        self.req == other.req && self.kind.fusable_kind_with(other.kind)
    }
}

/// Partition trip indices `0..kinds.len()` into CONTIGUOUS groups for medium-grain
/// fusion. Contiguous ⇒ emission (dataflow) order preserved ⇒ every producer
/// precedes its consumer, so cross-group edges thread through the shared global
/// HBM placement (addresses are grouping-invariant — baked at lowering, not here).
/// A run of `Pure` trips is chunked ≤ `g`; each `HostKv` (with the `skip` entries
/// it replaces) and each `Slot` is its OWN singleton. `g==1` reproduces the per-op
/// partition EXACTLY. PURE (no env/IO) so Kani verifies the covering-partition
/// invariant (no trip dropped or double-emitted → no dropped/duplicated compute).
pub fn group_ranges(trips: &[Trip], g: usize) -> Vec<core::ops::Range<usize>> {
    let g = if g == 0 { 1 } else { g };
    let n = trips.len();
    let mut groups: Vec<core::ops::Range<usize>> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if let GroupKind::HostKv { skip } = trips[i].kind {
            // this entry + the `skip` on-card copies it replaced: each its own singleton, so the
            // shim's per-entry `oi += skip` advances over exactly them.
            let end = i.saturating_add(1).saturating_add(skip).min(n);
            let mut k = i;
            while k < end {
                groups.push(k..k + 1);
                k += 1;
            }
            i = end;
            continue;
        }
        // EVERY other kind takes the same walk. `end` starts one past `start`, so the walk always
        // advances (termination is structural, not a special case) and a kind that fuses with
        // nothing — SlotSolo — comes out a singleton without an arm of its own.
        let start = i;
        let mut end = start + 1;
        let mut cnt = 1usize;
        while end < n && cnt < g && trips[start].fusable_with(&trips[end]) {
            end += 1;
            cnt += 1;
        }
        groups.push(start..end);
        i = end;
    }
    groups
}

/// CBMC-tractable SCALAR twin of [`group_ranges`]'s partition walk (a `Vec<Range>`
/// is CBMC-hostile — dynamic alloc over symbolic control flow blows up). Runs the
/// IDENTICAL i-advancing control flow but tracks only scalars: the covered-prefix
/// end and whether every emitted group is contiguous + correctly sized (Pure ≤ g,
/// any non-Pure group == singleton). Returns `(covered_end, ok)`. Because it shares
/// [`group_ranges`]'s exact walk, proving `(covered_end==n && ok)` proves the Vec
/// version is an EXACT covering partition (no trip dropped/double-emitted) with the
/// size/singleton invariant. Keep byte-identical to `group_ranges`'s match arms.
pub fn group_ranges_cover_ok(trips: &[Trip], g: usize) -> (usize, bool) {
    let g = if g == 0 { 1 } else { g };
    let n = trips.len();
    let mut i = 0usize;
    let mut expect = 0usize; // start of the next group must equal the covered-prefix end
    let mut ok = true;
    while i < n {
        if let GroupKind::HostKv { skip } = trips[i].kind {
            let end = i.saturating_add(1).saturating_add(skip).min(n);
            let mut k = i;
            while k < end {
                ok &= k == expect; // contiguous; each is a [k,k+1) singleton (len 1 ≤ g)
                expect = k + 1;
                k += 1;
            }
            i = end;
            continue;
        }
        let start = i;
        let mut end = start + 1;
        let mut cnt = 1usize;
        while end < n && cnt < g && trips[start].fusable_with(&trips[end]) {
            end += 1;
            cnt += 1;
        }
        ok &= start == expect;
        ok &= (end - start) <= g; // group ≤ g
        ok &= end > start; // non-empty ⇒ the walk advances ⇒ terminates
        expect = end;
        i = end;
    }
    (expect, ok)
}

/// Whether ANY group of this partition spans two requests — the bug-#7 predicate.
///
/// A group is one launch, and a launch shifts the KV base by ONE request's page/cursor. So a group
/// holding trips of two requests gives the second request the first's KV, silently: fluent output,
/// wrong tokens. Scalar (Vec-free) so CBMC can prove it never happens for ANY trip sequence.
pub fn group_spans_two_requests(trips: &[Trip], g: usize) -> bool {
    let g = if g == 0 { 1 } else { g };
    let n = trips.len();
    let mut i = 0usize;
    while i < n {
        if let GroupKind::HostKv { skip } = trips[i].kind {
            i = i.saturating_add(1).saturating_add(skip).min(n);
            continue;
        }
        let start = i;
        let mut end = start + 1;
        let mut cnt = 1usize;
        while end < n && cnt < g && trips[start].fusable_with(&trips[end]) {
            end += 1;
            cnt += 1;
        }
        let mut k = start;
        while k < end {
            if trips[k].req != trips[start].req {
                return true;
            }
            k += 1;
        }
        i = end;
    }
    false
}

/// Medium-grain fusion group size (trips per concrete dxp bundle). `1` = the
/// historical per-op path (each trip its own program). Env `SCRATCHY_SUPERDSC_GROUP_SIZE`.
/// Hoisted OUT of the pure `group_ranges` (env reads unbound CBMC — see `plan_capped`).
pub fn group_size() -> usize {
    // DEFAULT = 512. Decode is ~100% launch-count bound (measured ~3640 launches/tok @ ~41µs); fusing
    // consecutive ops into ≤g groups collapses that. 2048 was too aggressive — a fp8-dynamic granite
    // prefill body fused 512-trip groups fine but a dxp_standalone `vector::_M_range_check` crash surfaced
    // once fusion pushed past that; 512 is the real ceiling, not 2048. `SCRATCHY_SUPERDSC_GROUP_SIZE=1` is
    // the explicit DEBUG opt-out (per-op fault isolation). Read at EMIT time + forwarded by each arch
    // build.rs (rerun-if-env-changed + rustc-env) so a change recompiles.
    std::env::var("SCRATCHY_SUPERDSC_GROUP_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&g| g >= 1)
        .unwrap_or(512)
}

/// The FLAT (no-loop) `bundle.mlir` body for the all-time=1 case: one
/// `sdscbundle.sdsc_execute` per op, addresses baked concrete in the json. Kept
/// BYTE-IDENTICAL to the historical emission so the green single-shot tests do
/// not change. `emit_bundle_mlir` delegates here when no op has time>1.
pub fn bundle_mlir(sdsc_filenames: &[String]) -> String {
    let mut body = String::new();
    for f in sdsc_filenames {
        body.push_str(&format!(
            "    sdscbundle.sdsc_execute () {{sdsc_filename=\"{f}\"}}\n"
        ));
    }
    format!("module {{\n  func.func @sdsc_bundle() {{\n{body}    return\n  }}\n}}\n")
}

/// Expand one [`EmittedOp`] into its CONCRETE per-trip [`SdscOp`]s. A time=1 op
/// yields `[op.clone()]`. A time=N op yields N copies: in trip `t`, every tiled
/// tensor's AllocNode start addresses are bumped by `t · stride_bytes` (the
/// `affine_strides[ti]["out"]` advance); addresses stay CONCRETE. This PRE-UNROLL
/// replaces a symbolic `scf.for` — dxp's always-on `LoopUnroll` would otherwise
/// clone an in-loop `sdsc_execute` with IDENTICAL `symbol_ids` and double-reserve
/// them (DtException "Symbol already reserved", VariableDefinition.cpp:629; #53).
pub fn concrete_trips(e: &EmittedOp) -> Vec<SdscOp> {
    if e.time <= 1 {
        return vec![e.op.clone()];
    }
    (0..e.time)
        .map(|t| {
            let mut op = e.op.clone();
            for dsc_map in op.dscs_.iter_mut() {
                for dsc in dsc_map.values_mut() {
                    for node in dsc.scheduleTree_.iter_mut() {
                        let ti = node.ldsIdx_ as usize;
                        let Some(stride) = e.affine_strides.get(ti).and_then(|m| m.get("out"))
                        else {
                            continue; // non-tiled tensor — base address unchanged.
                        };
                        let bump = t as i64 * *stride;
                        for v in node.startAddressCoreCorelet_.data_.values_mut() {
                            let base: i64 = v.parse().unwrap_or(0);
                            *v = (base + bump).to_string();
                        }
                    }
                }
            }
            op
        })
        .collect()
}

/// The `bundle.mlir` orchestration `dxp_standalone --bundle -d <dir>` consumes:
/// one FLAT `sdscbundle.sdsc_execute () {sdsc_filename="sdsc_{i}.json"}` per
/// EXPANDED trip — concrete addresses baked into each json, NO `scf.for`, NO
/// symbols. A tiled op (time=N) contributes N consecutive flat executes (its
/// pre-unrolled trips from [`concrete_trips`]); a time=1 op contributes one. The
/// trip ordering is `ops` in order, trips `0..time` — IDENTICAL to [`write_bundle`]
/// so `sdsc_{i}.json` names line up. Delegates to [`bundle_mlir`].
pub fn emit_bundle_mlir(ops: &[EmittedOp]) -> String {
    let total: usize = ops.iter().map(|e| e.time.max(1) as usize).sum();
    let filenames: Vec<String> = (0..total).map(|i| format!("sdsc_{i}.json")).collect();
    bundle_mlir(&filenames)
}

/// Write a SuperDSC bundle directory: one `sdsc_{idx}.json` per op (serialized
/// `{ "{op_name}": SdscOp }`) + `bundle.mlir`. This is the artifact the
/// in-scratchy AoT bake feeds to `dxp_standalone --bundle -d <dir>` (driven by
/// `scr chat`, never a hand CLI — see the design memory). `ops` = [`EmittedOp`]
/// in topological order. Every op's HBM start addresses are SYMBOLIZED (see
/// [`symbolize_op`]) and the segment-base VALUES passed via `bundle.mlir`
/// symbol_ids so dxp's ModuleStitcher wires the multi-op dataflow.
/// Build the per-op manifest entries (idx/subdir/op_name/slot_stride_bytes + the `host_kv_write`
/// metadata) in the SAME idx/trip order [`write_bundle`] lays out the `op_{idx}/` subdirs. Factored
/// out so [`write_bundle_cached`] can refresh `op_manifest.json` UNCONDITIONALLY (even on a
/// device-artifact cache hit) — the manifest is a runtime-read artifact (like `bundle_layout.json`)
/// and MUST always reflect the current emitter, else a host-routing change (e.g. `host_kv_write`) is
/// silently dropped when the `sdsc_*.json` device ops are byte-identical (the cache reuses the old
/// dir, so the old manifest survives). Pure: no file IO; mirrors `write_bundle`'s trip iteration.
/// Per-trip [`GroupKind`] + owning-op index for the body's flat trip sequence.
/// `write_bundle` and `build_manifest_ops` BOTH call this so their grouping is
/// IDENTICAL (same partition, same subdir names). `host_kv_write`'s first trip
/// carries `HostKv{skip=kv_n_skip}` (kv_n_skip counts the on-card cachewr TRIPS
/// it replaces — the same unit the shim's `oi += kv_n_skip` advances over);
/// `slot_stride_bytes>0` → `Slot`; everything else `Pure` (fusable).
fn trip_kinds_and_owner(ops: &[EmittedOp]) -> (Vec<Trip>, Vec<usize>) {
    let mut kinds = Vec::new();
    let mut owner = Vec::new();
    for (oi, e) in ops.iter().enumerate() {
        let nt = (e.time.max(1)) as usize; // == concrete_trips(e).len()
        for ti in 0..nt {
            let k = if e.host_kv_write && ti == 0 {
                GroupKind::HostKv {
                    skip: e.kv_n_skip as usize,
                }
            } else if e.kv_page_fold {
                GroupKind::PageFold
            } else if e.slab_write {
                // Incremental Kᵀ restickify (kill-restickify Stage 2) → Slab (fusable, one
                // `slab_stride_bytes`). A DISTINCT kind from Slot: its 8192-byte slab stride ≠ the
                // cachewr's 128, so fusing them would trip the per-group uniform-stride assert.
                GroupKind::Slab
            } else if e.slot_stride_bytes > 0 {
                // Distinct per-slot cachewr (mq>1 prefill) → SlotSolo (singleton); uniform-slot
                // cachewr (decode) → Slot (fusable). A distinct-slot copy MUST NOT be fused into
                // a uniform-shift Slot group (would collapse the per-slot writes).
                if e.slot_no_fuse {
                    GroupKind::SlotSolo
                } else {
                    GroupKind::Slot { req: e.kv_request }
                }
            } else {
                GroupKind::Pure
            };
            kinds.push(Trip::new(k, TripRequest(e.kv_request)));
            owner.push(oi);
        }
    }
    (kinds, owner)
}

/// Whether the per-page fold gets a group of its own.
///
/// Splitting it costs ONE EXTRA LAUNCH PER LAYER, and this path is launch-bound: measured, the split
/// body is 5 groups where the baseline is 4, which is 40 more launches per token. That is only worth
/// paying when the fold is actually re-launched — i.e. when the context spans more than one page.
/// So the body is emitted BOTH ways and the runtime picks: `Fused` is the baseline's launch count
/// for any context that fits a page, `Split` is what makes longer contexts expressible at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FoldGrouping {
    /// Fold fused into the surrounding body — baseline launch count, single-page contexts only.
    Fused,
    /// Fold in its own re-launchable group — one extra launch per layer, any context length.
    Split,
}

// The FOLD-ROW REGIME a bundle declares travels as `bundle::FoldRows` on every `bundle::OpEntry`, and
// `bundle::BundleCode::fold_rows()` is its one reader — including the "fold groups disagree" refusal,
// since a session carries ONE intermediate stride.

/// Trip kinds under a fold grouping: `Fused` folds the per-page fold back into the ordinary work so
/// it merges with the neighbouring run. THE one place the two variants differ.
/// The ops a bundle variant actually contains. `Fused` serves single-page contexts, which have no
/// FULL pages at all, so their zero-masked fold is dropped outright — keeping it would add back the
/// very launches this variant exists to avoid.
pub fn ops_for_grouping(ops: &[EmittedOp], _fold: FoldGrouping) -> Vec<&EmittedOp> {
    ops.iter().collect()
}

fn trip_kinds_for(ops: &[EmittedOp], fold: FoldGrouping) -> (Vec<Trip>, Vec<usize>) {
    let (mut kinds, owner) = trip_kinds_and_owner(ops);
    if fold == FoldGrouping::Fused {
        for t in kinds.iter_mut() {
            if matches!(t.kind, GroupKind::PageFold) {
                // Reclassified, but it KEEPS ITS REQUEST — so it still cannot fuse across requests.
                t.kind = GroupKind::Pure;
            }
        }
    }
    (kinds, owner)
}

/// This op's own single `Dsc` (every `EmittedOp` wraps exactly one `dscs_` entry — `emit_sdsc`/
/// `emit_sdsc_tiled` always produce ONE `Dsc` keyed by `op_name`) and its primary `ComputeOp` (index
/// 0 — the matmul/pointwise itself; a fused epilogue's own 2nd entry is looked at separately by
/// `print_fusion_candidates`, which is exactly the case a scan for NEW candidates must not double
/// count as still-unfused).
fn primary_compute_op(e: &EmittedOp) -> Option<(&Dsc, &ComputeOp)> {
    let dsc = e.op.dscs_.first()?.values().next()?;
    let cop = dsc.computeOp_.first()?;
    Some((dsc, cop))
}

/// Every `opFuncName` the DDL's matmul family answers to (fp16/fp8/int8 × plain/batched — see the
/// dtype-suffix match in this file's own `computeOp_` construction). A closed list, not a substring
/// guess, so a future op named e.g. `"batchnormfwd"` is never mistaken for a matmul.
const MATMUL_OP_FUNCS: &[&str] = &[
    "matmul",
    "matmulfp8",
    "matmulint8",
    "batchmatmul",
    "batchmatmulfp8",
    "batchmatmulint8",
];

/// FUSION CANDIDATES, measured from the real emitted tape (`SCRATCHY_SDSC_FUSION_SCAN=1`).
/// For every adjacent `(ops[i], ops[i+1])` where `ops[i]` is a matmul,
/// reports whether `ops[i+1]` reads `ops[i]`'s own output buffer (a real producer/consumer link, via
/// `arg_bindings` — the SAME buffer-identity mechanism the runtime uses, not a name-string guess) and
/// whether it writes BACK to that same buffer (the in-place "epilogue" shape this codebase's fusion
/// mechanism can already express). For each of `ops[i+1]`'s OTHER inputs (the epilogue's would-be
/// extra operand), prints that buffer's own `scale_` from `labeledDs_` — a `-1`/non-1 entry is this
/// codebase's existing broadcast marker (see `Scale`), the signal that decides admissibility,
/// not an assumption made without looking.
fn print_fusion_candidates(ops: &[EmittedOp]) {
    let mut n = 0usize;
    for i in 0..ops.len().saturating_sub(1) {
        let (a, b) = (&ops[i], &ops[i + 1]);
        let Some((_, a_cop)) = primary_compute_op(a) else {
            continue;
        };
        if !MATMUL_OP_FUNCS.contains(&a_cop.opFuncName.as_str()) {
            continue;
        }
        let Some((b_dsc, b_cop)) = primary_compute_op(b) else {
            continue;
        };
        let a_outputs: Vec<&str> = a
            .arg_bindings
            .iter()
            .filter(|ab| !ab.is_input)
            .map(|ab| ab.buffer.as_str())
            .collect();
        let b_inputs: Vec<&str> = b
            .arg_bindings
            .iter()
            .filter(|ab| ab.is_input)
            .map(|ab| ab.buffer.as_str())
            .collect();
        let b_outputs: Vec<&str> = b
            .arg_bindings
            .iter()
            .filter(|ab| !ab.is_input)
            .map(|ab| ab.buffer.as_str())
            .collect();
        let Some(&shared) = a_outputs.iter().find(|o| b_inputs.contains(o)) else {
            continue;
        };
        let in_place = b_outputs.contains(&shared);
        n += 1;
        eprintln!(
            "[fusion-scan] {:>3}. {} (matmul:{}) -> {} ({}:{}) shares '{shared}', in_place={in_place}",
            n, a.op_name, a_cop.opFuncName, b.op_name, b_cop.exUnit, b_cop.opFuncName,
        );
        for extra in b_inputs.iter().filter(|&&x| x != shared) {
            // `labeledDs_`/`arg_bindings` are BOTH parallel to `op.args` (ldsIdx_ == position ==
            // arg_bindings index — see `emit_sdsc`'s `labeled.push` loop), so the extra operand's
            // OWN position in `b.arg_bindings` is its index into `b_dsc.labeledDs_` too. `dsName_`
            // there is the generic `"Tensor{i}"` placeholder, never the real buffer name, so
            // matching by NAME (as opposed to by shared position) would silently find nothing.
            let idx = b.arg_bindings.iter().position(|ab| ab.buffer == *extra);
            let scale = idx
                .and_then(|i| b_dsc.labeledDs_.get(i))
                .map(|l| l.scale_.clone());
            eprintln!("[fusion-scan]      extra operand '{extra}' scale_={scale:?}");
        }
    }
    eprintln!("[fusion-scan] {n} candidate(s) found");
}

/// THE PER-GROUP ADDRESS SHIFTS: one [`bundle::KvShifts`] per launch group, in launch order.
///
/// The partition here MUST match [`render_dxp_input`]'s exactly — shifts partitioned one way against
/// programs partitioned another shift the wrong program. Both call `trip_kinds_for` + `group_ranges`
/// with the same `group_size()`, and `group_ranges` is Kani-proven an exact covering partition; the two
/// are then ZIPPED into one [`bundle::LaunchGroup`] each, so the pairing exists at one site instead of
/// being an index the reader has to trust.
pub fn launch_index(ops: &[EmittedOp], fold: FoldGrouping) -> Vec<bundle::KvShifts> {
    // MEDIUM-GRAIN: one manifest entry per fusion GROUP (a contiguous run of ≤ g Pure
    // trips = one concrete dxp bundle; every host_kv_write/slot trip a singleton). g==1
    // reproduces the historical per-trip manifest exactly. The shim reads one entry →
    // one program; a pure group needs no per-entry handling, a host_kv/slot singleton
    // carries its metadata (the shim's `oi += kv_n_skip` then skips exactly the singleton
    // groups that hold the replaced copies).
    let (kinds, owner) = trip_kinds_for(ops, fold);
    let groups = group_ranges(&kinds, group_size());
    // LAUNCHES, not ops. A group IS a launch, and fusion means the two numbers are far apart — the
    // decode body is ~271 ops in 5 groups. An op-count gate therefore cannot see a fusion regression:
    // marking each batched cache write as its own singleton left the op count untouched (+16 per
    // request, as designed) while multiplying launches ~26x, which ran a batch of 8 twenty times
    // slower than a single request. Print it so the emit fingerprint can carry it.
    //
    // ⛔⛔⛔ AND YET AT bs=1 A LAUNCH IS ESSENTIALLY FREE — MEASURED 2026-08-16, granite-8b fp8, by
    // baking the SAME tree at two group sizes and interleaving them run-for-run:
    //   GROUP_SIZE=128 → 5 groups/layer = 200 launches/token → slow-mode ITL 82.2-83.0 ms
    //   GROUP_SIZE=512 → 3 groups/layer = 120 launches/token → slow-mode ITL 82.3-83.1 ms
    // +80 launches a token moved the slow mode by NOTHING, so the per-launch overhead is <= ~0.02 ms
    // and the 0.134 ms/launch that `SCRATCHY_SDSC_GROUP_TIME` reports at GROUP_SIZE=1 is almost all the
    // per-op DRAIN that timer needs, not launch cost.
    //
    // ⚖️ BOTH FACTS ARE TRUE AND THEY ARE ABOUT DIFFERENT THINGS: the bs=8 regression above was
    // launches that each carry a SLOT/PAGE SHIFT and serialize the fold, while these are Pure groups
    // over one row. So do not price a bs=1 launch reduction (merging the Slot/PageFold kinds, i.e. the
    // host-routing project) as an ITL win — at bs=1 the decode critical path is the WEIGHT STREAM
    // (8.477 GB/token at ~103 GB/s against IBM's own 150 GB/s measured peak), and op count and launch
    // count have now BOTH been measured not to move it.
    eprintln!(
        "[spyre-superdsc] GROUPS — {} launch group(s) for {} trip(s)",
        groups.len(),
        kinds.len(),
    );
    // WHAT THE OPS ACTUALLY ARE. A transformer layer is ~7 matmuls, 2 norms, RoPE and attention; the
    // decode body emits 271 at ONE row. Counting them by family says where that number comes from,
    // which no other diagnostic here reports — op_manifest.json has the names but is only written on
    // a forge run, so it cannot be read from a local bake.
    if std::env::var_os("SCRATCHY_SDSC_OPHIST").is_some() {
        let mut hist: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
        for e in ops.iter() {
            // Collapse the indices a family varies over (head, block, request, tensor id).
            let fam: String = e
                .op_name
                .chars()
                .map(|c| if c.is_ascii_digit() { '#' } else { c })
                .collect();
            let fam = fam.replace("##", "#").replace("##", "#").replace("##", "#");
            *hist.entry(fam).or_default() += 1;
        }
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by_key(|e| std::cmp::Reverse(e.1));
        eprintln!("[sdsc-ophist] {} ops:", ops.len());
        for (k, n) in v.iter().take(24) {
            eprintln!("[sdsc-ophist]   {n:5}  {k}");
        }
        // ⭐⭐⭐ TRIPS, NOT OPS — and the two are FAR apart, which is why every op-count reading of this
        // backend's cost has been wrong. An op with `time = t` streams its weight tile `t` times, so
        // TRIPS are what the weight stream is billed in: the decode body's 635 trips at granite-8b are
        // 4 launch groups, and one group holds 512 of them. MEASURED against the card:
        //   2b  248 trips/layer x 40 layers = 9,920/token, wstride 60.9 MB/layer => 246 KB/trip, ITL 26.0 ms
        //   8b  635 trips/layer x 40 layers = 25,400/token, wstride 211.9 MB/layer => 389 KB/trip, ITL 82.5 ms
        // i.e. 93.6 vs 102.7 GB/s of achieved weight bandwidth — the two models sit on ONE line, and
        // 8b is NOT over-split relative to 2b. Fitting the pair as `fixed + bytes/bw` gives ~227 GB/s
        // marginal and ~1.5 us FIXED per trip, so roughly half of a trip's time is overhead that a
        // BIGGER TILE would amortize. This histogram is where that per-family tile size is read off.
        let mut trips: std::collections::BTreeMap<String, (usize, u32)> = Default::default();
        for e in ops.iter() {
            let fam: String = e
                .op_name
                .chars()
                .map(|c| if c.is_ascii_digit() { '#' } else { c })
                .collect();
            let fam = fam.replace("##", "#").replace("##", "#").replace("##", "#");
            let slot = trips.entry(fam).or_default();
            slot.0 += 1;
            slot.1 += e.time;
        }
        // ⭐⭐⭐ THE WEIGHT STREAM'S OWN GEOMETRY — cores, splits and trips per matmul, because at bs=1 the
        // weight stream IS the decode step (op count and launch count are both measured inert) and NOTHING
        // else here reports how the stream is divided. A `time > 1` op streams its weight tile that many
        // times, so `bytes / (cores * time)` is the per-core burst a trip actually issues; comparing it
        // against `USABLE_LX_BYTES` says whether a LONGER burst was available.
        for e in ops.iter().filter(|e| e.time > 0) {
            let splits = &e.op.numWkSlicesPerDim_;
            if e.time == 1 && splits.values().all(|&v| v <= 1) {
                continue; // an unsplit, untiled op has no stream geometry to report
            }
            eprintln!(
                "[sdsc-stream]   {:<24} time={:<3} cores={:<3} splits={:?}",
                e.op_name, e.time, e.op.numCoresUsed_, splits,
            );
        }
        let mut tv: Vec<_> = trips.into_iter().collect();
        tv.sort_by_key(|e| std::cmp::Reverse(e.1.1));
        let total: u32 = tv.iter().map(|(_, (_, t))| *t).sum();
        eprintln!(
            "[sdsc-trips] {total} trips over {} ops (trips = ops x time):",
            ops.len()
        );
        for (k, (n, t)) in tv.iter().take(24) {
            eprintln!(
                "[sdsc-trips]   {t:6} trips  {n:5} ops  time={:<4} {k}",
                *t as usize / n.max(&1)
            );
        }
    }
    // FUSION CANDIDATES, MEASURED, not guessed from reading `attn.rs`. Every prior candidate list
    // came from reading model-arch source; this walks the ACTUAL emitted
    // tape for whatever model/config is baking, generic across architectures (no attention-specific or
    // granite-specific logic below) — a `(matmul, next-op)` pair is a candidate iff they share a buffer
    // name via `arg_bindings` (the SAME mechanism the runtime uses to know which buffer is which), with
    // no assumption about which op families exist upstream.
    if std::env::var_os("SCRATCHY_SDSC_FUSION_SCAN").is_some() {
        print_fusion_candidates(ops);
    }
    // WHICH GROUP IS WHICH. The shim's per-group timer (`SCRATCHY_SDSC_GROUP_TIME`) reports by group
    // INDEX, because a launched group has no name on the runtime side — so its histogram is a list of
    // anonymous numbers unless the emitter says what index `gi` holds. It is the same index here and
    // there (both walk `group_ranges` output in order), so printing owner+extent+kind here is what
    // turns "op 19 is 24% of compute" into a statement about a specific matmul.
    if std::env::var_os("SCRATCHY_SDSC_OPHIST").is_some() {
        eprintln!(
            "[sdsc-groups] {} groups (index = the shim's `op N`):",
            groups.len()
        );
        for (gi, r) in groups.iter().enumerate() {
            let e = &ops[owner[r.start]];
            eprintln!(
                "[sdsc-groups]   op {gi:<3} {:3} trip(s)  req {}  {:?}  {}",
                r.end - r.start,
                kinds[r.start].req.0,
                kinds[r.start].kind,
                e.op_name,
            );
        }
    }
    let mut shifts: Vec<bundle::KvShifts> = Vec::new();
    for (gi, r) in groups.iter().enumerate() {
        // The group's first trip's owning op supplies name + any per-entry metadata.
        let e = &ops[owner[r.start]];
        // SLOT-group stride: the owner (first trip) carries it; a Slot group's shim slot-shift is
        // applied ONCE to the whole group, so EVERY op in the group MUST share one stride (else some
        // head lands at the wrong row). Guard it at build (an un-catchable `cargo build` panic — the
        // re-roll path swallows Err). Pure groups have stride 0 (their owner is a Pure op). HostKv
        // singleton stride is 0 (it carries kv_* metadata instead).
        let slot = if matches!(
            kinds[r.start].kind,
            GroupKind::Slot { .. } | GroupKind::SlotSolo
        ) {
            let stride = e.slot_stride_bytes;
            // A SlotSolo group is a singleton (r.len()==1) by construction, so the uniform-stride
            // check is trivially true; a fused Slot group must share one stride (uniform shift).
            for t in r.clone() {
                let o = &ops[owner[t]];
                assert!(
                    o.slot_stride_bytes == stride,
                    "fused Slot group {gi}: op '{}' stride {} != group stride {stride} — a fused \
                     Slot group's shim slot-shift is uniform, so all cachewr copies must share one \
                     stride (else a head writes the wrong cache row)",
                    o.op_name,
                    o.slot_stride_bytes
                );
                // AND one REQUEST. The shim resolves a group's page table and write cursor ONCE,
                // from the owner's `kv_request`, so two requests in one group would put one
                // request's token into the other's history — the one cache-write error no later
                // step can detect. `group_ranges` breaks a Slot run where the request changes, so
                // this cannot fire; it is here because the consequence is undetectable downstream.
                assert!(
                    o.kv_request == e.kv_request,
                    "fused Slot group {gi}: op '{}' names request {} but the group resolves request \
                     {} — a group is ONE launch and one page table, so a fused cachewr group must \
                     be single-request (else a request's token is written into another's history)",
                    o.op_name,
                    o.kv_request,
                    e.kv_request
                );
            }
            stride
        } else {
            0
        };
        // SLAB-group stride (kill-restickify Stage 2): mirror the Slot guard — the shim shifts the fused
        // Slab group's seg2 base by `(slot_pos/64)·slab_stride` ONCE, so every op in the group MUST share
        // one slab_stride (else a kv-head restickifies the wrong slab). Non-Slab groups: 0.
        let slab = if matches!(kinds[r.start].kind, GroupKind::Slab) {
            let stride = e.slab_stride_bytes;
            for t in r.clone() {
                let o = &ops[owner[t]];
                assert!(
                    o.slab_stride_bytes == stride,
                    "fused Slab group {gi}: op '{}' slab_stride {} != group stride {stride} — a fused \
                     Slab group's shim shift is uniform, so all restickifies must share one slab_stride \
                     (else a kv-head re-transposes the wrong slab)",
                    o.op_name,
                    o.slab_stride_bytes
                );
            }
            stride
        } else {
            0
        };
        let fold_group = matches!(kinds[r.start].kind, GroupKind::PageFold);
        let _ = gi; // launch order is the slice position now, not a field
        shifts.push(bundle::KvShifts {
            slot_stride_bytes: slot,
            slab_stride_bytes: slab,
            // PAGED: `page_slots` is the write-slot modulus AND the declaration that this bundle is
            // paged — a runtime that does not know the key must refuse rather than apply an
            // absolute-position shift and land past the page.
            page_slots: e.kv_page_slots,
            request: e.kv_request,
            // `page_fold` marks the group the runtime re-launches once per page.
            page_fold: fold_group,
            // ONE PASS PER PAGE INSTEAD OF PER (REQUEST, PAGE) — only when the fold's kernels actually
            // carry the request axis.
            batched_requests: fold_group && e.kv_batched_requests,
            fold_rows: match e.kv_fold_rows {
                scratchy_subtile::sdsc_abstract::FoldRowRegime::PerRequest => {
                    bundle::FoldRows::PerRequest
                }
                scratchy_subtile::sdsc_abstract::FoldRowRegime::WholeBatch => {
                    bundle::FoldRows::WholeBatch
                }
            },
        });
    }
    shifts
}

/// A KV HEAD INDEX, TYPED, from a loop counter — the one place a bare `usize` becomes a [`KvHead`].
///
/// A kv head, a query head, a slot and a feature are all small integers, and nothing but spelling kept
/// them apart at a call site. `?` here means a head past the pool's count is a build error rather than
/// an address in the next plane.
///
/// [`KvHead`]: scratchy_subtile::sdsc_abstract::KvHead
fn kv_head_of(
    kvh: u32,
    nkvh: u32,
) -> Result<scratchy_subtile::sdsc_abstract::KvHead, SuperDscError> {
    let n = std::num::NonZeroU32::new(nkvh)
        .ok_or_else(|| SuperDscError("a paged pool needs at least one kv head".into()))?;
    scratchy_subtile::sdsc_abstract::KvHead::new(kvh, n)
        .ok_or_else(|| SuperDscError(format!("kv head {kvh} is past the pool's {nkvh} head(s)")))
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  dxp's INPUT — the only files this emitter still writes, and they outlive one compile by nothing
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// One launch group's dxp input, RENDERED but not yet anywhere.
///
/// dxp takes a DIRECTORY and requires one json per device op (merging a group's ops into one file is
/// `DtException: Expected empty FoldManager when importing from json`), so these files must exist —
/// but only between being written and being compiled. Rendering is separate from writing so ONE
/// renderer serves both consumers: the bake queue's staging dir, and [`write_dxp_input`] for feeding
/// dxp by hand.
pub struct GroupInput {
    /// Launch order — the `group` of the [`bundle::OpEntry`] that names this program.
    pub group: u32,
    /// `(file name, contents)`: one `sdsc_{i}.json` per trip, then `bundle.mlir` LAST.
    ///
    /// Last is not cosmetic — `bundle.mlir` completes the group, so writing it last is what makes the
    /// directory safe to compile at exactly one point.
    pub files: Vec<(String, String)>,
    /// Total bytes, so the staging budget's reservation is exact rather than an estimate.
    pub bytes: usize,
    /// ⭐ CONTENT KEY over exactly what dxp will read, in read order — the bake queue's memo key.
    /// Two groups with the same key compile to the same bytes, which 42% of them do.
    pub key: u64,
}

/// Render every launch group's dxp input.
///
/// MEDIUM-GRAIN FUSION (the torch-spyre per-kernel model, mirrored SDSC-native): the flat trip
/// sequence is partitioned into contiguous GROUPS (≤ `group_size()` Pure trips; each
/// host_kv_write/slot trip a singleton) and each group is ONE flat multi-op bundle — the same shape
/// dxp compiles for the whole model, just smaller, so it compiles to ONE program. dxp wires the
/// intra-group dataflow by shared byte address (the addresses are the SAME global placements,
/// grouping-invariant — producer-out == consumer-in by construction).
///
/// The partition here MUST match [`build_manifest_ops_grouped`]'s exactly: a manifest partitioned one
/// way against programs partitioned another launches a different set of groups than the index
/// describes, silently dropping whole groups of every layer. Both call `trip_kinds_for` +
/// `group_ranges` with the same `group_size()`, and `group_ranges` is Kani-proven an exact covering
/// partition (`group_ranges_is_exact_covering_partition`).
pub fn render_dxp_input(ops: &[EmittedOp], fold: FoldGrouping) -> std::io::Result<Vec<GroupInput>> {
    // Pre-unroll every op into its concrete trips and render one json per trip, in `ops`-order then
    // trip-order.
    let mut trip_json: Vec<String> = Vec::new();
    for (idx, (e, trip)) in ops
        .iter()
        .flat_map(|e| concrete_trips(e).into_iter().map(move |t| (e, t)))
        .enumerate()
    {
        // Top-level key carries the GLOBAL program-step index `{idx}_{op_name}` (mirroring
        // torch-spyre's `{idx}_{opfunc}`) so dxp's ModuleStitcher can order it — without it:
        // `DtException: expecting a valid entry in core schedule` (ModuleStitcher.cpp:216).
        let key = format!("{idx}_{}", e.op_name);
        let one: BTreeMap<&str, &SdscOp> = BTreeMap::from([(key.as_str(), &trip)]);
        // COMPACT, not pretty. Nothing reads these by eye — dxp parses them — and a 12B model emits
        // ~230k of them across the rung ladder. MEASURED on a real gemma-4 op: 32073 B pretty vs
        // ~9.7 KB compact, so indentation was ~2/3 of every byte written and ~2/3 of the per-file time.
        trip_json.push(
            serde_json::to_string(&one)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        );
    }
    let (kinds, _owner) = trip_kinds_for(ops, fold);
    let mut out = Vec::new();
    for (gi, r) in group_ranges(&kinds, group_size()).iter().enumerate() {
        let mut files: Vec<(String, String)> = Vec::with_capacity(r.end - r.start + 1);
        let mut names: Vec<String> = Vec::with_capacity(r.end - r.start);
        for (li, ti) in (r.start..r.end).enumerate() {
            let name = format!("sdsc_{li}.json");
            files.push((name.clone(), trip_json[ti].clone()));
            names.push(name);
        }
        // ⭐ THE KEY HASHES THE COMPILER'S INPUT, in the order `bundle.mlir` lists it — hashing
        // anything else would let two different inputs collide, and hashing the directory afterwards
        // would re-read what we just wrote.
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for j in &trip_json[r.start..r.end] {
                j.hash(&mut h);
            }
            h.finish()
        };
        files.push(("bundle.mlir".to_string(), bundle_mlir(&names)));
        out.push(GroupInput {
            group: gi as u32,
            bytes: files.iter().map(|(_, c)| c.len()).sum(),
            key,
            files,
        });
    }
    Ok(out)
}

/// Write the rendered dxp input into `dir/group_{gi}/` — the ARTIFACT DUMP.
///
/// Not part of the build (the bake queue stages its own copy and reclaims it): this is for handing a
/// bundle to `dxp_standalone --bundle -d <dir>` by hand, which is how a scheduler refusal gets
/// diagnosed and how the on-card gate bundles are produced.
pub fn write_dxp_input(
    dir: &std::path::Path,
    ops: &[EmittedOp],
    fold: FoldGrouping,
) -> std::io::Result<()> {
    for g in render_dxp_input(ops, fold)? {
        let gdir = dir.join(format!("group_{}", g.group));
        std::fs::create_dir_all(&gdir)?;
        for (name, contents) in &g.files {
            std::fs::write(gdir.join(name), contents)?;
        }
    }
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Emitting a bundle
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// ⭐ THE MEMORY PLAN, PROJECTED INTO THE BAKED ARTIFACT.
///
/// ⛔ THE EXHAUSTIVE DESTRUCTURE IS THE GUARD, and it is the whole reason this is one function. There
/// is no `..` below: adding a field to [`BundleLayout`] fails to compile HERE until it is either baked
/// or explicitly named as emit-only.
fn bake_layout(l: &BundleLayout) -> bundle::BundleLayout<'static> {
    let BundleLayout {
        ids,
        placements,
        segment_bytes,
        kernel_weights,
        scalarmul_scales,
        synth,
        // EMIT-ONLY, deliberately not baked: the arrangement authority is the build-time check that a
        // tensor has ONE device layout (`declare_arrangement` returns a build `Err` naming the tensor
        // otherwise). Its verdict is that the bundle compiled; there is nothing for the runtime to do
        // with the table.
        arrangements: _,
        kv_request_stride_bytes,
    } = l;

    let mut places: Vec<bundle::Placement> = placements
        .iter()
        .map(|(tid, p)| bundle::Placement {
            id: bundle::PlaceId::Act(*tid),
            segment: p.segment as u32,
            offset: p.offset,
            size: p.size,
            is_logits: p.role == SegRole::Logits,
        })
        .collect();
    // SYNTHETIC intermediates are not SubtileIR tensors, so they have no `t{id}` — they are named
    // directly and always live in the Intermediate segment.
    let synth = synth.borrow();
    places.extend(synth.map.iter().map(|(name, off)| {
        bundle::Placement {
            // ⛔ REFUSES rather than defaulting. Every synthetic is minted through `PlaceId::synth`,
            // which records the id; a name in `map` with no id means something allocated an
            // intermediate by SPELLING it, which is the round-trip this replaced. A fabricated id
            // here would place a real tensor under a wrong identity and the host would bind past it.
            id: *ids
                .borrow()
                .get(name)
                .unwrap_or_else(|| panic!("synthetic '{name}' was allocated without an identity")),
            segment: SegRole::Intermediate as u32,
            offset: *off,
            // ⛔ NO `unwrap_or(0)`. A zero-length field is read by the device as `1 << 27` flits —
            // 16 GiB — in both `handleHostDMA` and `handleXLATentry`. Every synthetic now has a size,
            // declared or reserved at first reference, so a missing one is a bug in the allocator and
            // says so.
            size: *synth
                .sizes
                .get(name)
                .unwrap_or_else(|| panic!("synthetic '{name}' has an offset but no reserved size")),
            is_logits: false,
        }
    }));

    // ⭐ EVERY ADDRESS THIS BUNDLE WILL USE IS NOW DECIDED. Prove them before emitting.
    audit_layout_addresses(&places, segment_bytes, "<layout>");

    bundle::BundleLayout {
        segment_bytes: *segment_bytes,
        places: std::borrow::Cow::Owned(places),
        kernel_weights: std::borrow::Cow::Owned(
            kernel_weights
                .iter()
                .map(|(tid, k)| bundle::KernelWeight {
                    id: bundle::PlaceId::Act(*tid),
                    device_size: std::borrow::Cow::Owned(k.device_size.clone()),
                    stride_map: std::borrow::Cow::Owned(k.stride_map.clone()),
                    stick_size: k.stick_size,
                    word_length: k.word_length,
                })
                .collect(),
        ),
        scalarmul_scales: std::borrow::Cow::Owned(scalarmul_scales.clone()),
        kv_request_stride_bytes: *kv_request_stride_bytes,
    }
}

/// One emitted bundle, awaiting the device code its groups are still being compiled into.
///
/// ⛔ WHY THE TWO HALVES ARE SEPARATE. The bake queue is deliberately ASYNCHRONOUS — group N of this
/// bundle compiles while the NEXT bundle is being written, which is what keeps `dxp_standalone` running
/// `COMPILE_WIDTH` wide across bundle boundaries instead of winding down to one at each of them. So the
/// metadata is finished here and the programs are collected by [`Self::into_code`], after the ONE drain
/// at the end of the emit.
pub struct EmittedBundle {
    /// Content fingerprint — this bundle's identity, and how a sibling names it.
    pub fp: String,
    /// The memory plan. Empty (`Default`) when the caller supplied no layout.
    pub layout: bundle::BundleLayout<'static>,
    /// Every launch, in launch order: its address shifts paired with the id its compiled program will
    /// be reported under. PAIRED HERE, at the one point where both come off the same `group_ranges`
    /// walk, so nothing downstream re-derives the correspondence.
    pub groups: Vec<(bundle::KvShifts, crate::superdsc_bake::GroupId)>,
}

/// ⭐ EMIT ONE BUNDLE: build its launch index and memory plan, and hand every launch group to the
/// bounded builder queue.
///
/// ⛔ NOTHING PERSISTS. The only files this writes are dxp's input, staged and reclaimed by the bake
/// queue; the bundle itself is collected in memory for the macro to bake. "Don't do this work twice" is
/// the bake queue's CONTENT-KEY memo, which keys on the exact bytes handed to the compiler and so
/// dedupes ACROSS fingerprints — catching the ~31% of groups that recur between a bundle and its fused
/// twin, or between ladder rungs.
///
/// Returns the bundle's FINGERPRINT, which is its identity everywhere downstream: a sibling names it,
/// [`attach_reroll`] addresses it, and [`drain_emitted_bundles`] hands it to the macro. Emitting the same
/// fingerprint twice (the ladder's ceiling rung IS the body) is a no-op the second time.
///
/// `Err` is a build failure. A CARDLESS build (no `dxp_standalone`) stages nothing and writes NO file
/// at all; the bundle's metadata is still collected, and `bundle::have_device_code()` reports the
/// absence of the programs.
/// ⛔⛔⛔ THE WORK AUDIT — A PROGRAM THAT COULD WAIT FOREVER MAY NOT COMPILE.
///
/// Every loop count in an emitted op is a COMPILE-TIME CONSTANT: the fold factors, the core count,
/// the trip count are all decided by the `#[forward]` procmacro before a single byte reaches the
/// card. So "will this program terminate, and will it do work" is a question the bake can answer,
/// and the card should never be the thing that discovers the answer is no — it discovers it by
/// waiting, with no host stack and no statement of which op is spinning.
///
/// ⛔ ZERO MEANS MAXIMUM ON THIS HARDWARE, NOT NOTHING. That is not a guess: the device's own DMA
/// and XLAT paths both read `if (length == 0) length = 1 << 27` — a zero length field transfers
/// 16 GiB. A zero fold factor is the same shape of field in the same family of descriptors, and
/// this bake has already shipped two zero-valued constants that nothing rejected (a synthetic with
/// no reserved size, and an allocator handing out one address repeatedly). A zero here is either a
/// loop that runs never — silently wrong — or one that runs to a maximum nobody intended.
///
/// Checked per op:
///   * every `FoldProp.factor_` (the sdsc folds, the core fold, the corelet fold) is `> 0`
///   * `numCoresUsed_` is in `1..=MAX_CORES` — zero cores is an op that no unit will ever run,
///     and more than the machine has is a schedule the hardware cannot satisfy
///   * `time` (the pre-unroll trip count) is `> 0`
fn audit_op_work(ops: &[EmittedOp], fp: &str) {
    let mut fail: Vec<String> = Vec::new();

    for op in ops {
        let name = &op.op_name;
        if op.time == 0 {
            fail.push(format!("{name}: time (trip count) is 0"));
        }
        if op.op.numCoresUsed_ == 0 {
            fail.push(format!(
                "{name}: numCoresUsed_ is 0 — no unit will ever run this op"
            ));
        } else if op.op.numCoresUsed_ > MAX_CORES {
            fail.push(format!(
                "{name}: numCoresUsed_ is {} but the machine has {MAX_CORES}",
                op.op.numCoresUsed_
            ));
        }
        let folds = op
            .op
            .sdscFoldProps_
            .iter()
            .map(|f| ("sdsc", f))
            .chain(std::iter::once(("core", &op.op.coreFoldProp_)))
            .chain(std::iter::once(("corelet", &op.op.coreletFoldProp_)));
        for (which, f) in folds {
            if f.factor_ <= 0 {
                fail.push(format!(
                    "{name}: {which} fold '{}' has factor {} — a loop count must be positive, and \
                     a zero-valued count field on this device reads as its MAXIMUM",
                    f.label_, f.factor_
                ));
            }
        }
    }

    if !fail.is_empty() {
        let shown = fail.len().min(12);
        panic!(
            "\n⛔ [superdsc-work] bundle {fp}: {} OP(S) WITH NO PROVABLE WORK. Refusing to bake.\n\n{}\n{}\n             Every count here is a compile-time constant. A program whose loop bound is zero does \
             not fail on the card — it runs never, or it runs to a maximum, and either way the \
             host waits for a completion that is not coming.\n",
            fail.len(),
            fail[..shown]
                .iter()
                .map(|f| format!("  • {f}"))
                .collect::<Vec<_>>()
                .join("\n"),
            if fail.len() > shown {
                format!("  … and {} more\n", fail.len() - shown)
            } else {
                String::new()
            },
        );
    }
}

pub fn emit_bundle(
    ops: &[EmittedOp],
    layout: Option<&BundleLayout>,
    fold: FoldGrouping,
) -> std::io::Result<String> {
    // ⭐ EVERY LOOP COUNT IN THESE OPS IS DECIDED. Prove they terminate and do work before emitting.
    audit_op_work(ops, "<bundle>");
    // Content-addressed: the same ops at the same grouping ARE the same bundle, so the second emit has
    // nothing to do. The ladder's ceiling rung IS the body, so this fires on every rolled model.
    let fp = bundle_fp(ops, fold);
    if emitted().lock().is_ok_and(|c| c.contains_key(&fp)) {
        return Ok(fp);
    }
    let b = emit_bundle_inner(ops, layout, fold)?;
    debug_assert_eq!(
        b.fp, fp,
        "bundle_fp and emit_bundle_inner disagree on the fingerprint"
    );
    if let Ok(mut c) = emitted().lock() {
        c.entry(fp.clone()).or_insert((b, None));
    }
    Ok(fp)
}

/// This bundle's fingerprint — the SPLIT hash, with an `f` suffix marking the fused variant.
///
/// ⛔ THE FUSED TWIN'S NAME KEYS OFF THE UNFILTERED OPS. The runtime finds a body's fused twin as
/// `<split fp>f`, so the base hash must be the split body's, computed before `ops_for_grouping` drops
/// anything. Hashing the trimmed list is what made the twin unfindable.
fn bundle_fp(ops: &[EmittedOp], fold: FoldGrouping) -> String {
    let base = bundle_fingerprint_grouped(ops, FoldGrouping::Split);
    match fold {
        FoldGrouping::Split => base,
        FoldGrouping::Fused => format!("{base}f"),
    }
}

/// Attach the layer-loop parameters to the re-rolled BODY named by `fp`.
///
/// Separate from [`emit_bundle`] because a body's meta names its prefix, suffix, fused twin and every
/// ladder rung — none of which exist yet when the body itself is emitted.
pub fn attach_reroll(fp: &str, meta: bundle::RerollMeta<'static>) {
    if let Ok(mut c) = emitted().lock()
        && let Some(slot) = c.get_mut(fp)
    {
        slot.1 = Some(meta);
    }
}

/// EVERY BUNDLE THIS EMIT PRODUCED, by fingerprint — the channel the macro drains.
///
/// ⛔ A REGISTRATION, NOT A SEARCH. A bundle is in here because [`emit_bundle`] put it here, so
/// [`drain_emitted_bundles`] cannot silently return fewer bundles than were emitted.
#[allow(clippy::type_complexity)]
fn emitted()
-> &'static std::sync::Mutex<BTreeMap<String, (EmittedBundle, Option<bundle::RerollMeta<'static>>)>>
{
    static C: std::sync::OnceLock<
        std::sync::Mutex<BTreeMap<String, (EmittedBundle, Option<bundle::RerollMeta<'static>>)>>,
    > = std::sync::OnceLock::new();
    C.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()))
}

/// Take every emitted bundle, complete with its compiled device code — what the macro bakes.
///
/// Call ONCE, after `superdsc_bake::finish_global()` has drained the compilers.
pub fn drain_emitted_bundles() -> Result<Vec<bundle::BundleCode<'static>>, String> {
    let taken = match emitted().lock() {
        Ok(mut c) => std::mem::take(&mut *c),
        Err(_) => return Err("the emit collector was poisoned by a panic".to_string()),
    };
    taken
        .into_values()
        .map(|(b, reroll)| b.into_code(reroll))
        .collect()
}

fn emit_bundle_inner(
    ops: &[EmittedOp],
    layout: Option<&BundleLayout>,
    fold: FoldGrouping,
) -> std::io::Result<EmittedBundle> {
    // ── GUARD #14 (a VERIFICATION since task #50, not a refusal of time-tiling): across an op's
    //    trips, no two writes to the SAME OUTPUT tensor may land on the same byte address. With real
    //    per-core HBM segment addressing each tiled tensor's trips advance from a real per-core base,
    //    so they cannot alias by construction — a residual stride bug would be silently wrong, so it
    //    is checked rather than assumed. ──
    if let Some(e) = ops.iter().find(|e| e.time > 1 && tiled_trips_alias(e)) {
        return Err(std::io::Error::other(format!(
            "[spyre-superdsc] time-tiled op '{}' (time={}) emits ALIASING per-trip OUTPUT addresses \
             — two trips would write the same HBM byte (silently-wrong). The #50 per-core stride must \
             advance each trip past the previous; this is an internal stride/segment bug. Refusing to \
             bake.",
            e.op_name, e.time
        )));
    }

    let fp = bundle_fp(ops, fold);
    let kept: Vec<&EmittedOp> = ops_for_grouping(ops, fold);
    let filtered: Vec<EmittedOp>;
    let ops: &[EmittedOp] = if kept.len() == ops.len() {
        ops
    } else {
        // Rebuild the trimmed list by value only when something was actually dropped.
        filtered = kept.into_iter().map(EmittedOp::shallow_copy).collect();
        &filtered
    };

    let shifts = launch_index(ops, fold);
    let mut groups: Vec<(bundle::KvShifts, crate::superdsc_bake::GroupId)> = Vec::new();
    // No dxp (a Mac / cardless `cargo check`) ⇒ nothing to stage, and nothing to stage it FOR. This
    // is a CAPABILITY probe, not a behaviour flag: there is one code path and it is taken whenever the
    // tool exists.
    if let Some(bake) = crate::superdsc_bake::global() {
        for g in render_dxp_input(ops, fold)? {
            let id = crate::superdsc_bake::GroupId {
                fp: fp.clone(),
                group: g.group,
            };
            let gdir = bake.stage().group_dir(&fp, g.group as usize);
            // ⭐ CLAIM THE DISK BEFORE USING IT. The size is exact — the json is already rendered —
            // and `reserve` BLOCKS until it fits under `MAX_STAGED_BYTES`. This is what bounds
            // staging: without it the emitter runs ahead of dxp and stages the whole ladder.
            bake.reserve(g.bytes);
            std::fs::create_dir_all(&gdir)?;
            for (name, contents) in &g.files {
                std::fs::write(gdir.join(name), contents)?;
            }
            // SEALED HERE and nowhere else: `bundle.mlir` is rendered LAST in `files`, so this is the
            // one point at which the group is complete and safe to compile. `submit` returns the first
            // dxp refusal, so a ladder that cannot compile stops on its first group.
            bake.submit(crate::superdsc_bake::SealedGroup::sealed(
                gdir,
                id.clone(),
                g.bytes,
                g.key,
            ))
            .map_err(std::io::Error::other)?;
            // ⛔ THE SHIFTS FOR *THIS* GROUP, by the same index that produced it. `render_dxp_input`
            // and `launch_index` walk one partition, so a missing entry here is a partition bug and a
            // build error rather than a program launched with another group's shifts.
            let kv = *shifts.get(g.group as usize).ok_or_else(|| {
                std::io::Error::other(format!(
                    "{fp}: group {} has dxp input but no entry in the launch index ({} entries) — \
                     `render_dxp_input` and `launch_index` disagree about the group partition",
                    g.group,
                    shifts.len(),
                ))
            })?;
            groups.push((kv, id));
        }
    }

    Ok(EmittedBundle {
        fp,
        layout: layout.map(bake_layout).unwrap_or_default(),
        groups,
    })
}

impl EmittedBundle {
    /// Collect this bundle's compiled programs and produce the value the macro bakes.
    ///
    /// Call AFTER `superdsc_bake::finish_global()`. A group the queue has no result for is an `Err`
    /// naming it: after a successful drain that can only mean it was never submitted, and a bundle with
    /// a hole in its launch sequence must not reach the binary.
    pub fn into_code(
        self,
        reroll: Option<bundle::RerollMeta<'static>>,
    ) -> Result<bundle::BundleCode<'static>, String> {
        let bake = crate::superdsc_bake::global();
        let mut groups = Vec::with_capacity(self.groups.len());
        for (kv, id) in &self.groups {
            let compiled = bake.and_then(|b| b.compiled_group(id)).ok_or_else(|| {
                format!(
                    "{id}: submitted for compilation but no compiled program came back — the bake \
                     queue drained without an error, so this group was never submitted"
                )
            })?;
            groups.push(bundle::LaunchGroup {
                kv: *kv,
                init_binary: std::borrow::Cow::Owned(compiled.init_binary.clone()),
                job_bin_ptr: compiled.job_bin_ptr,
                correction: std::borrow::Cow::Owned(compiled.correction.clone()),
            });
        }
        Ok(bundle::BundleCode {
            fp: std::borrow::Cow::Owned(self.fp),
            layout: self.layout,
            groups: std::borrow::Cow::Owned(groups),
            reroll,
        })
    }
}

/// Deterministic content fingerprint of a SuperDSC bundle — its identity everywhere downstream, and
/// the bake queue's cue that a bundle it has already emitted is this one again.
///
/// Stable across builds (`DefaultHasher` has a fixed seed). Hashes the rendered `bundle.mlir` as well
/// as the per-op json, because the loop structure varies with `time`/strides while the byte-faithful
/// per-op json does not.
pub fn bundle_fingerprint(ops: &[EmittedOp]) -> String {
    bundle_fingerprint_grouped(ops, FoldGrouping::Split)
}

/// As [`bundle_fingerprint`], keyed by the fold grouping as well — the two variants share their ops and
/// differ only in group boundaries.
pub fn bundle_fingerprint_grouped(ops: &[EmittedOp], fold: FoldGrouping) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // MEDIUM-GRAIN: fold the group size in, since flipping G repartitions the body into different
    // launch groups and therefore different dxp programs.
    group_size().hash(&mut h);
    // ⛔ AND THE ACTUAL PARTITION, NOT JUST G. A `GroupKind` change (e.g. distinct-slot cachewr
    // Slot→SlotSolo) repartitions the body into a DIFFERENT set of groups even when every per-op json
    // and G are unchanged, so two bundles that launch differently would otherwise share a fingerprint —
    // and one would silently answer for the other.
    {
        let (kinds, _owner) = trip_kinds_for(ops, fold);
        for r in group_ranges(&kinds, group_size()) {
            (r.start, r.end).hash(&mut h);
        }
    }
    for e in ops {
        for trip in concrete_trips(e) {
            e.op_name.hash(&mut h);
            serde_json::to_string(&trip)
                .unwrap_or_default()
                .hash(&mut h);
        }
    }
    emit_bundle_mlir(ops).hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Build-time aliasing check for a TIME-TILED op (#50 verification, GUARD #14):
/// expand the op's concrete trips and assert that no two trips write the SAME
/// HBM byte address for the SAME OUTPUT tensor (ldsIdx). Returns `true` if any
/// alias is found (→ a build `Err` in [`write_bundle_cached`]). A time=1 op never
/// aliases (one trip). Only OUTPUT/non-input HBM allocate nodes are checked —
/// inputs/LX legitimately repeat addresses (they are read, never written).
fn tiled_trips_alias(e: &EmittedOp) -> bool {
    if e.time <= 1 {
        return false;
    }
    // ldsIdx of every OUTPUT (write) tensor: the computeOp_ outputLabeledDs names
    // are `Tensor{i}-idx{i}`, so the output ldsIdx set is derivable, but the simpler
    // robust signal is the LabeledDs dsType_ == "OUTPUT". Collect those ldsIdx.
    //
    // A K-SPLIT (reduction/`in`-dim) matmul legitimately has MULTIPLE cores accumulate a partial
    // product into the SAME shared output address WITHIN one trip (dxp PSUM-accumulates them) —
    // the exact case the #50 disjoint-output guard already relaxes elsewhere (`reduction_split` at
    // line ~4082: `op.iter.split_of("in") > 1 && !v.layout.contains(&"in")`). Comparing addresses
    // GLOBALLY across every trip (the old `seen` set spanning all trips) flagged those intra-trip,
    // same-address accumulation writes as if they were a cross-TRIP collision — a false positive:
    // matmul_o325-class ops (m=31, K-split, time=2) legitimately repeat an address several times
    // within trip 0 (one per K-split core) and AGAIN within trip 1 (bumped by exactly one trip's
    // stride) — never actually reusing a byte a DIFFERENT trip already wrote. The real invariant is
    // per-trip: dedupe addresses WITHIN each trip first (collapsing the expected K-split repeats),
    // THEN check that address does not ALSO appear in a DIFFERENT trip's deduped set — that is the
    // only condition GUARD #14 is meant to catch (trip N re-writing a byte trip M already wrote).
    let trips = concrete_trips(e);
    let mut seen_by_trip: Vec<std::collections::BTreeSet<(u32, u64)>> =
        Vec::with_capacity(trips.len());
    for trip in &trips {
        let mut this_trip: std::collections::BTreeSet<(u32, u64)> =
            std::collections::BTreeSet::new();
        for dsc_map in &trip.dscs_ {
            for dsc in dsc_map.values() {
                let out_lds: std::collections::BTreeSet<u32> = dsc
                    .labeledDs_
                    .iter()
                    .filter(|l| l.dsType_ == "OUTPUT")
                    .map(|l| l.ldsIdx_)
                    .collect();
                for node in &dsc.scheduleTree_ {
                    if node.component_ != "hbm" || !out_lds.contains(&node.ldsIdx_) {
                        continue;
                    }
                    for v in node.startAddressCoreCorelet_.data_.values() {
                        if let Ok(a) = v.parse::<u64>() {
                            this_trip.insert((node.ldsIdx_, a));
                        }
                    }
                }
            }
        }
        seen_by_trip.push(this_trip);
    }
    for i in 0..seen_by_trip.len() {
        for j in (i + 1)..seen_by_trip.len() {
            if !seen_by_trip[i].is_disjoint(&seen_by_trip[j]) {
                return true; // trip j re-writes a byte trip i already wrote.
            }
        }
    }
    false
}

// ───────────────────────────────────────────────────────────────────────────
// The SubtileIR walk — the integration that turns the model's fused tape into a
// SuperDSC bundle. Peer of the sengraph lowering off the SAME `SubtileIR`, but
// tile→tile (SubtileIR is ALREADY tile-level) into the work-divided IR. Driven
// from `scr chat` via the AoT bake — never a hand CLI.
// ───────────────────────────────────────────────────────────────────────────

// `SuperDscError` now lives in the shared leaf module [`scratchy_subtile::superdsc_error`]
// so the typed core (`superdsc_opspec::WorkPlan::time_tile_for_lx`) and this wire
// module raise the SAME build-time error without a module cycle. Re-exported here
// so every existing `superdsc::SuperDscError` path keeps resolving.
pub use scratchy_subtile::superdsc_error::SuperDscError;

/// Stable per-tensor SuperDSC dataspace name (SSA tensor id → `t{id}`).
fn ds_name(tr: &scratchy_subtile::subtile_ir::TensorRegion) -> String {
    crate::wiring::act_name(tr.tensor.index() as u32)
}

/// Lower one [`SubOp::MatmulTile`] node to a complete matmul [`SdscOp`].
///
/// Geometry comes straight off the node's regions (NOT the sengraph emitter):
/// `inputs[0]` = A `[m, k]`, `inputs[1]` = W `[k, n]` (row-major `[K, N]` per the
/// FUF `gemm(x:[..,K], w:[K,N])` convention — so `transpose_b` is always false),
/// `output` = `[m, n]`. The Spyre tape does NOT K-chunk (each `MatmulTile` is a
/// whole GEMM; SuperDSC owns the K-split via the cost model), so `k` is full K.
/// `batch = 1` — prefill/decode rows are folded into M.
// Returns a Vec so the fp32-SFP-merge split-K path (SCRATCHY_KSPLIT) can emit B block matmuls + fp32 merge
// adds; the default (single f16 matmul) returns a 1-element Vec — behavior-identical to the prior single-op
// return. All fp32 addressing is proof-gated (fp32_partial_addressing_4byte_single_row, mixed_dtype_add_...,
// ksplit_partial_merge_equals_full_matmul, ksplit_kslice_read_needs_full_weight_stick_stride).
fn lower_matmul_node<F: RopeForm>(
    node: &SubtileNode<F>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    // fp8 W8A8: the per-token activation quantize (square→amax→scale→clamp→qfp8ch) is a PURE function of the
    // activation tensor, independent of the weight — so q/k/v (all reading rms₁) and gate/up (both reading
    // rms₂) can quantize ONCE and share the fp8 activation. This set records the activation `t{id}`s already
    // quantized in this bundle; a repeat matmul on the same activation emits only `matmulfp8` + dequant.
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 && node.inputs.len() != 3 {
        return Err(SuperDscError(format!(
            "MatmulTile t{} expects 2 inputs (A, W) or 3 (A, W_fp8, w_scale for fp8 W8A8), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let a = &node.inputs[0];
    let w = &node.inputs[1];
    // Route the ENTRY through TileIR (node_to_single_tile_op), same as SiluMul/RmsNorm/RopeRotate/
    // AttnDecode/Elementwise/SumReduce/ScalarMul: m/k/n come FROM the TileOp's mb/in/out dims (built
    // directly from a.region.rows.len / a.region.cols.len / node.output.region.cols.len — see
    // subtile_tape_to_tile_ir.rs's MatmulTile arm — zero transformation, so byte-identical). Every
    // downstream check (w.region.rows.len != k, node.output.region.rows.len != m, etc.) still validates
    // OTHER tensors' regions against these values, unaffected by where m/k/n originate. The entire fp8
    // quant-chain decomposition below is COMPLETELY UNCHANGED.
    let matmul_tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    let dim = |name: &str| {
        matmul_tile_op
            .dims
            .iter()
            .find(|d| d.name == name)
            .map(|d| d.size)
    };
    let m = dim("mb").unwrap_or(a.region.rows.len);
    let k = dim("in").unwrap_or(a.region.cols.len);
    let n = dim("out").unwrap_or(node.output.region.cols.len);
    // TEMP diagnostic (emit-time): fires for EVERY matmul. If [MM] lines appear in the build log, the emit
    // IS re-running; arity=3 ⇒ the fp8 amax branch is live, arity=2 ⇒ fp8 rides the weight tensor's dtype.
    // If NO [MM] lines appear at all, the emit is CACHED (not re-running) — the real inert-toggle cause.
    eprintln!(
        "[MM] out=t{} arity={} m={m} k={k} n={n} act=t{} W=t{}{}",
        node.output.tensor.index() as u32,
        node.inputs.len(),
        a.tensor.index() as u32,
        w.tensor.index() as u32,
        if node.inputs.len() == 3 {
            format!(" wscale=t{}", node.inputs[2].tensor.index() as u32)
        } else {
            String::new()
        }
    );

    // ── fp8 W8A8 (arity-3 [act(fp16 [m,k]), W(fp8 [k,n]), w_scale(f16 [n,1] ≡ [1,n])]): per-token quantize
    //    act → fp8, `matmulfp8` (fp8×fp8→fp16), dequant by `a_scale[m]·w_scale[n]`. DECODE (m=1) only — the
    //    per-token amax uses reduce-MAX, which #33 PROVED returns seed 0 for rows>1; prefill (m>1) fp8 needs
    //    the sum-based per-row amax (deferred, build-guarded below). fp8-ness is TYPED: the quantized act
    //    (`Df::Fp8` via `synth_df` + the qfp8ch convert's output) and the weight/act matmul operands
    //    (`set_df(Df::Fp8)`) carry SEN143_FP8 residency (½ f16) + drive the `matmulfp8` opFunc — no name suffix.
    if node.inputs.len() == 3 {
        let ws = &node.inputs[2];
        // fp8 W8A8 PREFILL (m>1). torch-spyre's `quantize_fp8_with_scale`
        // (decompositions.py:966) takes `scale` as an input computed by the CALLER via native `torch.abs` +
        // `torch.ops.aten.amax` (confirmed via grep — no L2-norm substitute anywhere in torch-spyre); the
        // `#33` belief that reduce-MAX "seeds 0 for rows>1" was disproved by the stable-softmax `block_max`
        // (attn.rs), itself a multi-row MAX reduce. ONE formula for any row count (m==1 decode or m>1
        // prefill), matching torch-spyre exactly — mirrors rmsnorm's unification (no scratchy-only approx,
        // no diagnostic-sweep scaffolding left in the live path).
        let stk = Fp16::ELEMS_PER_STICK; // 64
        // The quant chain's typed extents: the activation's `m` is the matmul's token rows, `k`/`n`
        // its in/out feature columns, and the per-row scale tensors are ONE STICK wide.
        let m_rows = scratchy_subtile::sdsc_abstract::RowCount::of_token_rows(m);
        let k_cols = scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(k);
        let n_cols = scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(n);
        let scale_cols = scratchy_subtile::sdsc_abstract::BlockCols::of_one_stick(
            scratchy_subtile::sdsc_abstract::Lanes::FP16,
        );
        let a_name = ds_name(a);
        let w_name = ds_name(w); // fp8 weight `t{id}` — declared fp8 on the matmul operand (set_df), not the name
        let ws_name = ds_name(ws); // w_scale, read [1,n] (byte-identical to the loaded [n,1])
        let out = ds_name(&node.output);
        use bundle::SynthRole as R;
        // SHARED quant tensors are a pure function of the ACTIVATION (abs→amax→scale→clamp→fp8) — derive
        // them from the ACTIVATION's id so a second matmul on the same activation resolves the IDENTICAL
        // fp8 tensor + a_scale (the layout `synth` is idempotent per id → one reservation). PER-MATMUL
        // tensors (the matmul output `raw` and its dequant `dqa`) derive from the OUTPUT's id.
        let a_id = bundle::PlaceId::Act(a.tensor.index() as u32);
        let out_id = bundle::PlaceId::Act(node.output.tensor.index() as u32);
        // The operand SPELLING is a rendering of the id the layout is keyed by, so the op's name and
        // the allocator's key cannot drift — they have one preimage.
        let qn = |r: R| syn(layout, a_id.synth(r));
        let nm = |r: R| syn(layout, out_id.synth(r));
        // OP names — a DIFFERENT namespace from tensor names (a program's ops are not placed), so
        // they stay strings and take no `PlaceId`.
        let opn = |s: &str| format!("{out}_{s}");
        let (absx, amax, amaxfl, ascale, invs, sc, chi, cl) = (
            qn(R::FqAbsX),
            qn(R::FqAmax),
            qn(R::FqAmaxFl),
            qn(R::FqAscale),
            qn(R::FqInvS),
            qn(R::FqSc),
            qn(R::FqChi),
            qn(R::FqCl),
        );
        let (raw, dqa) = (nm(R::FqRaw), nm(R::FqDqA));
        let afp8 = qn(R::FqAfp8); // quantized fp8 activation — sized/typed fp8 via synth_df + the convert
        if let Some(l) = layout {
            // Footprints are `m` query rows (decode m=1 ⇒ byte-identical to the old `[1,·]`; prefill m>1
            // reserves the real multi-row extent so a per-row write can't clobber a neighbor row).
            for r in [R::FqAbsX, R::FqSc, R::FqChi, R::FqCl] {
                l.synth(a_id.synth(r), &[m, k]);
            }
            l.synth_df(a_id.synth(R::FqAfp8), &[m, k], Df::Fp8); // 1-byte / 128-stick residency (½ fp16)
            for r in [R::FqAmax, R::FqAmaxFl, R::FqAscale, R::FqInvS] {
                l.synth(a_id.synth(r), &[m, stk]);
            }
            for r in [R::FqRaw, R::FqDqA] {
                l.synth(out_id.synth(r), &[m, n]);
            }
        }
        let pos448 = rbo(&crate::wiring::act_name(FP8_POS448_TID));
        let neg448 = rbo(&crate::wiring::act_name(FP8_NEG448_TID));
        let inv448 = rbo(&crate::wiring::act_name(FP8_INV448_TID));
        let mut ops: Vec<EmittedOp> = Vec::new();
        // SHARE the activation quantize across matmuls that read the SAME activation. `insert` returns false
        // when this activation (`a_name`) was already quantized earlier in the bundle (e.g. K/V after Q, or
        // Up after Gate) — in that case skip re-emitting the whole abs→amax→scale→clamp→qfp8ch chain and
        // reuse its `afp8`/`ascale` (both named by `a_name`); only `matmulfp8` + dequant stay per-matmul.
        // (The chain is a launch/op cost paid ONCE per distinct activation instead of once per matmul.)
        let already_quantized = !quantized.insert(a_name.clone());
        if !already_quantized {
            // `absx`'s ONLY consumer is the MAX-reduce right below (never a matmul) — same bug class as
            // rmsnorm's `sq16` (see rmsnorm.rs): a shape-driven RowBlocked handle would make it stickmajor-
            // eligible purely from shape (rows>1 && cols>64, true here for any real hidden/intermediate `k`),
            // and the native reduce cannot correctly walk a stick-major activation along `cols` at rows>1 (it
            // sums/maxes across the wrong rows). Emitted via the TYPED `pw1`/`Stk<FlatTag>` path (not a raw
            // `&str` one) so the producer and this reduce's consumer are the SAME compile-time-checked
            // handle — `fl()` forces both ends flat; a kind mismatch would be a `cargo build` type error.
            // `absx` is RowBlocked, NOT Flat (2026-07-28) -- same fix, same reason, as rmsnorm's
            // `sq16` (see the long note in ir/bridge/tiled_op_sdsc_op/rmsnorm.rs). A Flat OUTPUT
            // handle forces head_major=true for the WHOLE op, which per pointwise.rs "stays rank-3"
            // and therefore also reads the INPUT activation rank-3 flat. But `a_name` is the
            // stick-major residual stream, so at m=mq=31 with k=2048 that read is (pointwise.rs's own
            // words) "scrambled at rows>1 AND cols>64": the per-row amax was taken over a scrambled
            // element set, giving every row the wrong fp8 activation scale -- on EVERY fp8 matmul
            // (qkv, o_proj, gate/up, down), every layer, for batched prefill only. RowBlocked puts
            // both this op and the amax reduce on the stick-major rank-2 path, matching how the
            // activation was actually written. At m==1 (decode) `stickmajor` is false in both, so the
            // emission is byte-identical to the previous Flat form.
            let absx_h = rb(&absx, m, k);
            ops.push(pw1(
                &opn("fq_absx_op"),
                "abs",
                m_rows,
                k_cols,
                In::full(&rb(&a_name, m, k)),
                &absx_h,
                sym_id_base,
                layout,
            )); // absx = |act|
            // REVERTED (2026-07-28): a multi-stage blocked-reduce experiment lived here (three
            // iterations, all confirmed on real hardware to make ZERO difference to the actual bug —
            // the K-cache inf this was meant to fix turned out to be caused by cachewr's matmul
            // mechanism instead, see the cachewr plain-copy fix). Unlike the reduce-MAX-at-rows>1
            // theory (which was checked against the OLD proven code and definitively disproven — that
            // code batches rows=nqh fine), this width-blocking theory was never confirmed against any
            // proven reference at all, was pure speculation, and never demonstrated any real effect.
            // Keeping ~150 lines of never-validated-on-hardware blocking logic around after its own
            // premise (fixing K-cache inf) is already fixed elsewhere is unjustified risk with no
            // upside. Reverted to the simple, single unblocked reduce — decode already proves this form
            // works (m=1 always takes this path); prefill (m>1) now takes it too.
            let amax_h = rb(&amax, m, stk);
            ops.push(assemble_reduce_seeded(
                &opn("fq_amax_op"),
                "max",
                m,
                k,
                &absx_h,
                &amax_h,
                sym_id_base,
                layout,
            ));
            // floor: amax = max(amax, 1/448) — a zero-amax (all-zero padded query row) would make
            // invs=recip(0)=inf → 0·inf=NaN downstream; every real (non-padded) row's amax is always >>
            // 1/448, so this leaves them byte-identical. A scratchy padding guard, not part of torch-spyre's
            // decomposition (which never sees a zero-amax row).
            let amaxfl_h = rb(&amaxfl, m, stk);
            ops.push(pw2(
                &opn("fq_amaxfl_op"),
                "maximum",
                m_rows,
                scale_cols,
                In::full(&amax_h),
                In::scalar(&inv448),
                &amaxfl_h,
                sym_id_base,
                layout,
            ));
            // a_scale = amax·(1/448); invs = reciprocal(a_scale) — torch-spyre's `quantize_fp8_with_scale`
            // takes `scale` as input and computes `reciprocal(scale)` itself.
            let ascale_h = rb(&ascale, m, stk);
            ops.push(pw2(
                &opn("fq_ascale_op"),
                "mul",
                m_rows,
                scale_cols,
                In::full(&amaxfl_h),
                In::scalar(&inv448),
                &ascale_h,
                sym_id_base,
                layout,
            ));
            let invs_h = rb(&invs, m, stk);
            ops.push(pw1(
                &opn("fq_invs_op"),
                "reciprocal",
                m_rows,
                scale_cols,
                In::full(&ascale_h),
                &invs_h,
                sym_id_base,
                layout,
            ));
            // scaled = act·invs; clamp to [-448,448] (torch-spyre's `quantize_fp8_with_scale` ALWAYS clamps).
            let sc_h = rb(&sc, m, k);
            ops.push(pw2(
                &opn("fq_sc_op"),
                "mul",
                m_rows,
                k_cols,
                In::full(&rb(&a_name, m, k)),
                In::col(&invs_h),
                &sc_h,
                sym_id_base,
                layout,
            ));
            let chi_h = rb(&chi, m, k);
            ops.push(pw2(
                &opn("fq_chi_op"),
                "minimum",
                m_rows,
                k_cols,
                In::full(&sc_h),
                In::scalar(&pos448),
                &chi_h,
                sym_id_base,
                layout,
            ));
            let cl_h = rb(&cl, m, k);
            ops.push(pw2(
                &opn("fq_cl_op"),
                "maximum",
                m_rows,
                k_cols,
                In::full(&chi_h),
                In::scalar(&neg448),
                &cl_h,
                sym_id_base,
                layout,
            ));
            // qfp8ch: f16 clamped act → SEN143_FP8. A dtype CONVERT (input 64-stick, output 128-stick), so it
            // goes through `assemble_convert` (two `primaryDsInfo_`), NOT a same-stick pointwise. The output's
            // `Df::Fp8` is intrinsic to the op (`convert_dtypes`).
            ops.push(assemble_convert(
                &opn("fq_afp8_op"),
                "qfp8ch",
                m,
                k,
                &rb(&cl, m, k),
                &rb(&afp8, m, k),
                sym_id_base,
                layout,
            ));
        } // end `if !already_quantized` — the shared activation-quantize chain (per distinct activation).
        // matmulfp8: raw = act_fp8 @ W_fp8 → fp16. Build the plain matmul opspec, then declare both operands
        // (A = quantized act, W = kernel) fp8 via `set_df` — that typed marker is what selects the `matmulfp8`
        // opFunc + the 128-stick / 1-byte operand residency. The output `raw` stays fp16 (the DDL `%ptsum_fp`).
        ops.push({
            // fp8 matmul GEOMETRY = fp16 (`::<Fp16>`): the OUTPUT is fp16 and the KERNEL's N-stick is 64,
            // so the `out` (N) work-division splits on the 64-stick — fp8-ness does NOT change the N-split.
            // (128 is ONLY the fp8 ACTIVATION's K-stick, which is the reduction axis — never split.) Then
            // stamp the activation + weight operands `Df::Fp8` (the ½-HBM 1-byte residency); `emit_sdsc`
            // detects the fp8 matmul and emits the 2-D PACKED KERNEL stick `[in:2, out:64]` + the
            // SEN143_FP8 compute marker (the DDL `batchmatmulfp8` operand contract). The OUTPUT stays fp16.
            // `Df::Fp8` twice, deliberately: `operand_df` is what A/W COST in LX and is the only one
            // available early enough to reach the work-division and time-tiling decision (both run
            // inside the opspec builder); the `set_df` loop below is what they ARE on the wire.
            let mut spec = crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_off_operands::<Fp16>(
                scratchy_subtile::sdsc_abstract::MatM::of_token_rows(m),
                scratchy_subtile::sdsc_abstract::MatN::of_out_features(n),
                scratchy_subtile::sdsc_abstract::MatK::of_in_features(k),
                scratchy_subtile::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &afp8,
                &w_name,
                &raw,
                0,
                0,
                0,
                Df::Fp8,
            )
            .map_err(SuperDscError)?;
            for arg in &mut spec.args {
                if !matches!(arg.view().role, Role::Output) {
                    arg.set_df(Df::Fp8);
                }
            }
            let folds = SdscFoldSet::new(spec.iter.cores_used());
            emit_sdsc_tiled(&opn("fq_mm"), &spec, &folds, sym_id_base, layout)?
        });
        // dequant: out = raw · a_scale (per-row) · w_scale[n] (per-col). `ascale` is [m,stk] (one scale per
        // query row) read via `col` (out-broadcast over N, row r→row r) — per-row-correct for m>1,
        // byte-identical at decode m=1. `w_scale` is [1,n] per-CHANNEL; for m>1 it broadcasts over the m
        // query rows (`mb`, mirroring rmsnorm's gamma multiply) — a plain `full` would demand m distinct
        // rows from the 1-row w_scale.
        let ascale_h = rb(&ascale, m, stk);
        let raw_h = rb(&raw, m, n);
        let dqa_h = rb(&dqa, m, n);
        ops.push(pw2(
            &opn("fq_dqa_op"),
            "mul",
            m_rows,
            n_cols,
            In::full(&raw_h),
            In::col(&ascale_h),
            &dqa_h,
            sym_id_base,
            layout,
        ));
        let out_h = rb(&out, m, n);
        let ws_h = rb(&ws_name, 1, n);
        let w_scale_in = if m > 1 {
            In::mb(&ws_h)
        } else {
            In::full(&ws_h)
        };
        ops.push(pw2(
            &opn("fq_dqw_op"),
            "mul",
            m_rows,
            n_cols,
            In::full(&dqa_h),
            w_scale_in,
            &out_h,
            sym_id_base,
            layout,
        ));
        return Ok(ops);
    }
    // Internal consistency guards — a mismatch is a build error, not garbage.
    if w.region.rows.len != k {
        return Err(SuperDscError(format!(
            "MatmulTile t{}: W rows {} != A cols (K) {k}",
            node.output.tensor.index() as u32,
            w.region.rows.len
        )));
    }
    if w.region.cols.len != n {
        return Err(SuperDscError(format!(
            "MatmulTile t{}: W cols {} != out cols (N) {n}",
            node.output.tensor.index() as u32,
            w.region.cols.len
        )));
    }
    if node.output.region.rows.len != m {
        return Err(SuperDscError(format!(
            "MatmulTile t{}: out rows {} != A rows (M) {m}",
            node.output.tensor.index() as u32,
            node.output.region.rows.len
        )));
    }
    // ── Build-time GUARDS (priority-1 rule: a contract violation is a `cargo
    //    build` error, never on-card garbage). assemble_matmul emits NO
    //    coordinateMasking_, so the stick dims (K on INPUT, N on OUTPUT) MUST be
    //    64-fp16-stick-aligned; an unaligned dim would silently corrupt the tile
    //    layout on-card (guard #3). ──
    if k % FP16_ELEMS_PER_STICK != 0 {
        return Err(SuperDscError(format!(
            "MatmulTile t{}: K={k} not a multiple of the {FP16_ELEMS_PER_STICK}-fp16 stick \
             (assemble_matmul emits no coordinateMasking_) — emit masking or pad K",
            node.output.tensor.index() as u32
        )));
    }
    // The OUTPUT stick dim N must be a whole 64-fp16 stick on-device. Round the LOGICAL width up to a
    // stick (`n64`); then — if this matmul's MACs cross the util floor AND `n64`'s stick count is
    // prime/awkward (would strand the gemm on <8 cores, e.g. granite lm_head 49216/64=769) — bump the
    // stick count to a multiple of 8 (`bump_sticks_to_splittable`, SHARED with the worker's weight
    // zero-pad so the staged buffer matches the emitted device width). The kernel's extra (n_dev − n)
    // columns are ZERO; the SubtileIR/manifest LOGICAL shape stays `n`; the host reads the leading
    // `vocab = n` (contiguous, m=1). ONLY the on-device layout uses `n_dev` (a whole stick by construction).
    // TYPE-SAFE device width (the padding/alignment invariant): `DeviceWidth::for_output` is the SOLE
    // rule, SHARED with the worker's weight zero-pad + `kernel0`, so they cannot diverge.
    let n_dev = DeviceWidth::for_output(m, n, k).get();
    let macs = m as u64 * n_dev as u64 * k as u64;
    // ── GUARD #11 (util floor) computed on the PROVEN partition `CoreSplit::plan` (Kani: disjoint +
    //    covering, #50-free) — the SAME split the emit uses (`matmul_split_map` defers to CoreSplit for
    //    batch=1). A FLOP-heavy matmul left on <8 cores is the 1.45× sengraph regression this emitter
    //    exists to fix; the padding above is what fills them for a prime-stick output. ──
    let sp = CoreSplit::plan(m, n_dev);
    let cores = sp.ncores();
    if macs >= (1 << 20) && cores < 8 {
        return Err(SuperDscError(format!(
            "MatmulTile t{}: {m}×{n_dev}×{k} ({macs} MACs) CoreSplit-divided onto only {cores} core(s) — \
             below the util floor; the OUTPUT stick count is not splittable to ≥8 even after padding.",
            node.output.tensor.index() as u32
        )));
    }
    // ── GUARD (priority-1, from OBSERVED on-card crash 2026-06-25): the per-core STICK extent (N on
    //    OUTPUT/KERNEL) MUST stay a whole 64-fp16 stick — a sub-stick split DtException's the dxp
    //    scheduler (L3DlOpsScheduler.cpp:1040). `CoreSplit` splits `out` by STICK COUNT (whole sticks by
    //    construction) and never splits K, so this holds; kept as a build-time seal. ──
    if !(n_dev / sp.stick_cores).is_multiple_of(FP16_ELEMS_PER_STICK) {
        return Err(SuperDscError(format!(
            "MatmulTile t{}: CoreSplit stick_cores={} leaves a SUB-STICK per-core N extent (N/core={}) — \
             must be a multiple of {FP16_ELEMS_PER_STICK}.",
            node.output.tensor.index() as u32,
            sp.stick_cores,
            n_dev / sp.stick_cores
        )));
    }
    // ── fp32-SFP-merge split-K (SCRATCHY_KSPLIT) — the fp32-accumulation fix for the f16-PSUM near-tie ──
    // Split K into B stick-aligned blocks (KB=512 = 8 fp16 sticks). Each block is a NORMAL f16 matmul over
    // KB terms writing its OWN f16 partial `{o}_blkN`; the B partials MERGE in fp32 via `broadcast_ops`
    // `add [fp16,fp32]` (DeepTools-confirmed) into an `_fp32` accumulator (`dataformat_for` → IEEE_FP32) — the
    // PE PSUM stays f16 (bmm.ddl) but the cross-block sum is fp32 ⇒ approaches the fp32 golden. PROOF-GATED:
    // ksplit_granite_oproj_partition_stick_aligned, ksplit_partial_merge_equals_full_matmul,
    // mixed_dtype_add_fp16_fp32_byte_offsets, fp32_partial_addressing_4byte_single_row, fp32_affine_stride...
    // GATED OFF by default (working golden path = the single matmul below, byte-for-byte unchanged). Requires
    // the worker to stage `{w}_blkN` = the [KB,N] K-slice of the weight + bind `{o}_fp32zero` = [m,N] zeros
    // (else the block op errors "missing tensor" — the expected FIRST on-card signal to add the staging).
    // FIRST-TEST SCOPE: `n > 16384` restricts KSPLIT to the lm_head (n=vocab≈49155; every other matmul has
    // n ≤ 8192 = intermediate) — a SINGLE SUFFIX op (NOT per-layer-replicated), so its block weights stage
    // ONCE (B=4, no ×40 replication) — the tractable first mechanism test. lm_head K=hidden=2048 ⇒ the
    // partition is `ksplit_granite_oproj_partition_stick_aligned` (K=2048). Extend to o_proj/down_proj
    // (per-layer) once the mechanism is verified on-card.
    // ── F16 blocked split-K merge (SCRATCHY_KSPLIT) — the DD2-VIABLE accumulation-error reduction ──
    // DD2/RCUDD1A has NO fp32 ARITHMETIC (on-card: the fp32 SFP `add` → `Unsupported result precision
    // conversion in DD2`; fp32 is SEN1P5-only, same as the fp32 matmul). So the merge is ALL F16: B block
    // matmuls (each K/B terms, f16 PE) → f16 partials; then f16 SFP adds sum them. `blocked_f16_sum_beats_
    // sequential` (Kani-proven): fewer terms per block ⇒ MORE accurate than the current single full-K f16
    // accumulation — it REDUCES (not eliminates; `split_k_f16_merge_still_lossy`) the accumulation error, so it
    // MAY flip the small pos-65 near-tie. GATED OFF by default (the single matmul below is byte-for-byte the
    // golden path). lm_head-scoped (n>16384). Block weights staged under `ksplit_block_tid(b)` (kernel0 + worker).

    // ── LAYER blocked-f16 (SCRATCHY_LAYER_KSPLIT) — block the DOMINANT-K body matmul (down_proj K=8192) to
    // shorten the f16 accumulation DEPTH (the DDL-confirmed bmm.ddl %ptsum_fp/%pesum f16 accumulator = the
    // 1.375-logit token-7 residual). Reads K-SLICES of the EXISTING per-layer weight at logical row-offset
    // b·KB·n_dev (NO separate block staging), so it works for the symbolic body without ×40 block tids —
    // PROVIDED the opspec/retile admits a logical K-slice offset (the on-card DISCOVERY: if it does not, the
    // card mis-reads → localizes the need for staged block weights). PROOF-GATED: partition by
    // `ksplit_downproj_block_partition_disjoint_covering` (KB=512, K=8192→B=16), merge by
    // `ksplit_partial_merge_equals_full_matmul` (exact, K-indep), error-reduction by
    // `blocked_f16_sum_beats_sequential`. GATED OFF by default (the single matmul below is byte-for-byte the
    // golden path). Large-K only (k>=4096 ⇒ down_proj K=8192; NOT o_proj/gate/up K=2048) for the first test.

    let op_name = format!("matmul_o{}", node.output.tensor.index() as u32);
    let op = assemble_matmul_seeded(
        &op_name,
        m,
        n_dev,
        k,
        1,
        &rb(&ds_name(a), m, k),
        &Stk::<KernelTag>::kernel(k as usize, n_dev as usize, ds_name(w)),
        &rb(&ds_name(&node.output), m, n_dev),
        sym_id_base,
        layout,
    );
    // Default path: one f16 matmul (KSPLIT above replaces it with B blocks + fp32 merge when enabled).
    Ok(vec![op])
}

/// The SFP polynomial constant table the transcendental sfp ops need
/// (silu/exp/sigmoid/gelu/mish/reciprocal/sqrt share it). Emitted with the REAL
/// fp16-pair hex VALUES (verbatim from DeepTools `dsm.cpp:18130-18180`), so the op
/// is correct whether or not the DSM rewrites them, and `setupVariables` never
/// `map::at`-faults on a missing slot. Matches the 20-entry `sdsc_silu.json`.
/// The `scaling_factor` external constant (= 1/N for `mean`, 1.0 for `sum`/`max`)
/// the reduce template `summeanmaxexx2.ddl` requires. `packed` = the SEN169_FP16 bits in
/// the LOW 16 bits of the word (high 16 zero — the const element type is fp16, so a 32-bit
/// word "exceeds bitwidth"; see [`sen169_bits`]). Without it: DtException "Missing external
/// constant in DSC: scaling_factor" (ddl_conversion.cpp:718).
/// Encode an `f32` into **SEN169_FP16** (1 sign, 6-bit exponent bias-31, 9-bit mantissa) —
/// the AIU's native 16-bit float. This is NOT IEEE f16 (1-5-10): `half::f16` bits are
/// silently MIS-READ by the device (1.0 → IEEE 0x3C00 → SEN169 0.5; 1/576 → IEEE 0x171C →
/// SEN169 ≈ 1e-6). Pinned against the SFP constant table (`plus1=0x3E00`, `minus1=0xBE00`).
/// Sub-normals underflow to 0 and out-of-range magnitudes saturate to max-finite — scaling
/// factors `1/N` (N ≤ a few thousand) are always well inside the normal range, so neither
/// clamp ever fires for the reduce path. The result is used as the LOW 16 bits of a u32
/// (NOT replicated — the const element type is fp16; see the call site).
pub fn sen169_bits(v: f32) -> u16 {
    if v == 0.0 || !v.is_finite() {
        return 0;
    }
    // INTEGER-BITS encoder (no libm log2/powi — that was a float leaf Kani could not verify AND needed
    // the `a/exp2(e)` edge fixups because log2 rounds). SEN169_FP16 is 1-6-9 with exponent bias 31; IEEE
    // f32 is 1-8-23 with bias 127. For a normalized f32 the SEN169 exponent FIELD is already implied by the
    // IEEE exponent: E_sen = (ieee_exp − 127) + 31 = ieee_exp − 96, and the SEN169 mantissa is the top 9 of
    // the 23 IEEE mantissa bits, round-to-nearest (half-up; the mantissa is non-negative so this == round-
    // half-away, matching the old `.round()`), with a carry into the exponent. Proven by
    // `sen169_encode_*` Kani harnesses (kani_proofs.rs): to_bits + integer shifts are exactly CBMC's domain.
    let bits = v.to_bits();
    let sign: u16 = ((bits >> 31) as u16) << 15;
    let ieee_exp = ((bits >> 23) & 0xFF) as i32; // biased 1..254 for normals (0 = subnormal/zero)
    let ieee_mant = bits & 0x007F_FFFF; // 23-bit trailing significand

    if ieee_exp == 0 {
        return sign; // f32 subnormal ⇒ far below the SEN169 min normal ⇒ ±0
    }
    let mut exp_field = ieee_exp - 96; // SEN169 biased exponent field
    // Round 23→9 mantissa bits (add half-ULP at bit 13, shift out 14 low bits). mant ∈ [0, 512].
    let mut mant = ((ieee_mant + (1 << 13)) >> 14) as i32;
    if mant == 512 {
        mant = 0; // mantissa rounding carried into the exponent
        exp_field += 1;
    }
    if exp_field <= 0 {
        return sign; // underflow → ±0
    }
    if exp_field >= 63 {
        return sign | (62 << 9) | 0x1FF; // saturate to max finite
    }
    sign | ((exp_field as u16) << 9) | (mant as u16 & 0x1FF)
}

fn scaling_factor_const(packed: u32) -> serde_json::Value {
    serde_json::json!({
        "0": {
            "dataFormat_": "SEN169_FP16",
            "name_": "scaling_factor",
            "data_": [packed],
            "allocations_": {}
        }
    })
}

/// The `scaling_factor` external constant for an **fp32** reduce. Same shape as
/// [`scaling_factor_const`], but `dataFormat_` = `IEEE_FP32` and `bits` = the raw
/// IEEE-754 f32 word (`f32::to_bits`), matching torch-spyre `encodeConstant` which
/// returns `BinaryConvert<uint32_t>(float)` for `IEEE_FP32` (module.cpp:126). The
/// const's dtype MUST equal the op's data_format or DD2 rejects the fp32 reduce as
/// a mixed [fp16,fp32] op ("Unsupported result precision conversion").
fn scaling_factor_const_fp32(bits: u32) -> serde_json::Value {
    serde_json::json!({
        "0": {
            "dataFormat_": "IEEE_FP32",
            "name_": "scaling_factor",
            "data_": [bits],
            "allocations_": {}
        }
    })
}

fn sfp_constant_table() -> serde_json::Value {
    const ENTRIES: &[(&str, u32)] = &[
        ("dontSplatOutput", 0),
        ("negInf", 0xFFFE_FFFE),
        ("plus1", 0x3E00_3E00),
        ("minus1", 0xBE00_BE00),
        ("fastexpVal", 0x54E3_54E3),
        ("expVal1", 0x46DC_46DC),
        ("expVal2", 0x46E2_46E2),
        ("expVal3", 0x34C5_34C5),
        ("expVal4", 0x2121_2121),
        ("expVal5", 0x3E00_3E00),
        ("fastSigmoidConst", 0x3C00_3C00),
        ("geluVal1", 0x3D31_3D31),
        ("geluVal2", 0x3448_3448),
        ("zero", 0),
        ("maskone", 0x0001_0001),
        ("mishConstVal1", 0x3A00_3A00),
        ("mishConstVal2", 0x3AAB_3AAB),
        ("mishConstVal3", 0xBF00_BF00),
        ("mishConstVal4", 0x3CC6_3CC6),
        ("mishConstVal5", 0x54E3_54E3),
    ];
    let mut map = serde_json::Map::new();
    for (i, (name, val)) in ENTRIES.iter().enumerate() {
        map.insert(
            i.to_string(),
            serde_json::json!({
                "dataFormat_": "SEN169_FP16",
                "name_": name,
                "data_": [val],
                "allocations_": {}
            }),
        );
    }
    serde_json::Value::Object(map)
}

/// Guard: a pointwise op's stick (cols) axis must be 64-fp16-aligned, since
/// `assemble_pointwise` emits no coordinateMasking_ (guard #3, pointwise twin).
fn check_pointwise_cols(cols: u32, what: &str, out_id: u32) -> Result<(), SuperDscError> {
    if !cols.is_multiple_of(FP16_ELEMS_PER_STICK) {
        return Err(SuperDscError(format!(
            "{what} t{out_id}: cols={cols} not a multiple of the {FP16_ELEMS_PER_STICK}-fp16 \
             stick (assemble_pointwise emits no coordinateMasking_) — pad cols"
        )));
    }
    Ok(())
}

/// Lower a shape-preserving [`SubOp::Elementwise`] node (Add/Mul binary, Silu
/// unary) to ONE pointwise [`SdscOp`]. All operands are `[rows, cols]` (the
/// output shape); `op_func` + arity per the [`EwKind`].
fn lower_elementwise_node<F: RopeForm>(
    node: &SubtileNode<F>,
    kind: EwKind,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    let (op_func, arity): (&'static str, usize) = match kind {
        EwKind::Silu => ("silu", 1),
        // A REAL DDL primitive (`OpFunc::Gelu`), not a decomposition: the SFP
        // constant table ships gelu's polynomial, so this is one pointwise op.
        EwKind::Gelu => ("gelu", 1),
        EwKind::Mul => ("multiply", 2),
        EwKind::Add => ("add", 2),
        // ⛔ THE DDL HAS NO PRIMITIVE FOR THESE, AND SUBSTITUTING THE NEAREST ONE IS
        // THE BUG. `QuickGelu` is x·σ(1.702x) and `GeluErf` is the exact-erf gelu —
        // neither is `OpFunc::Gelu`'s tanh polynomial, so emitting "gelu" for them
        // would run a DIFFERENT function and report success. `Sub` broadcasts a
        // [m, 1] operand, which `pw2` cannot express (see the two-operand arity law).
        //
        // They reach this emitter because the SHARED front end now expresses the whole
        // arch vocabulary — that is the point: the op exists in the IR, and the TARGET
        // says whether it has a kernel. Before, the lowering panicked for everyone.
        EwKind::QuickGelu | EwKind::GeluErf | EwKind::Sub => {
            return Err(SuperDscError(format!(
                "Elementwise({kind:?}) t{} has no dxp DDL primitive: quick-gelu and \
                 exact-erf gelu are distinct functions from OpFunc::Gelu, and Sub takes \
                 a [m, 1] broadcast operand that pw2 cannot express",
                node.output.tensor.index() as u32,
            )));
        }
    };
    // Transcendental sfp ops (silu/…) now ship the full SFP polynomial table with
    // real fp16 values (`sfp_constant_table`, verbatim from dsm.cpp), so there is
    // no missing-key `map::at` risk — `assemble_pointwise` emits the table via
    // `constant_info(op_func)`. On-card validation confirms whether the DSM also
    // fills them (then the table is belt-and-suspenders) or relies on it (then it
    // is required); either way the emitted op is correct.
    if node.inputs.len() != arity {
        return Err(SuperDscError(format!(
            "Elementwise({op_func}) t{} expects {arity} input(s), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let cols = node.output.region.cols.len;
    check_pointwise_cols(cols, "Elementwise", node.output.tensor.index() as u32)?;
    let in_names: Vec<String> = node.inputs.iter().map(ds_name).collect();
    let in_refs: Vec<&str> = in_names.iter().map(|s| s.as_str()).collect();
    let op_name = format!("{op_func}_o{}", node.output.tensor.index() as u32);
    // SubtileTape -> TileIR -> tiler -> SdscOp, ONE call chain: `node_to_single_tile_op` derives the
    // TileOp straight from this node (the same function `lower_tape_to_tile_ir` uses when walking a
    // whole tape), then `assemble_pointwise_seeded_from_tile` runs the tiler + builds the SdscOp.
    let tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    let op = assemble_pointwise_seeded_from_tile(
        &op_name,
        &tile_op,
        op_func,
        &in_refs,
        &ds_name(&node.output),
        sym_id_base,
        layout,
    );
    Ok(op)
}

/// Lower a [`SubOp::SumReduce`] — which (per `eval_node`) is a SHAPE-PRESERVING
/// ELEMENTWISE SUM across its N inputs (`out = Σ_k inputs[k]`), NOT an axis
/// reduction — as a variadic pointwise `add` over `[rows, cols]` (all operands +
/// the output share the `[mb,out,y]` OUTPUT layout). `out` is the 64-stick axis.
fn lower_sumreduce_node<F: RopeForm>(
    node: &SubtileNode<F>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    if node.inputs.is_empty() {
        return Err(SuperDscError(format!(
            "SumReduce t{} expects >=1 input, found 0",
            node.output.tensor.index() as u32
        )));
    }
    let cols = node.output.region.cols.len;
    check_pointwise_cols(cols, "SumReduce", node.output.tensor.index() as u32)?;
    let in_names: Vec<String> = node.inputs.iter().map(ds_name).collect();
    let in_refs: Vec<&str> = in_names.iter().map(|s| s.as_str()).collect();
    let op_name = format!("add_o{}", node.output.tensor.index() as u32);
    let tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    Ok(assemble_pointwise_seeded_from_tile(
        &op_name,
        &tile_op,
        "add",
        &in_refs,
        &ds_name(&node.output),
        sym_id_base,
        layout,
    ))
}

/// Lower a fused [`SubOp::SiluMul`] (`out = silu(gate) · up`) by DECOMPOSING into
/// two pointwise ops: `silu(gate) → <out>_silu`, then `multiply(<out>_silu, up) →
/// out`. `inputs[0]` = gate, `inputs[1]` = up (per the SubtileIR doc). The `silu`
/// half ships the full SFP polynomial table (`sfp_constant_table`); the
/// intermediate `<out>_silu` is a synthetic dataspace the coloring pass allocates.
fn lower_silumul_node<F: RopeForm>(
    node: &SubtileNode<F>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "SiluMul t{} expects 2 inputs (gate, up), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    // check_pointwise_cols needs the RAW (pre-TileOp) cols to validate stick-alignment before anything
    // rounds it. Route the ENTRY through TileIR (node_to_single_tile_op) right after — same as
    // Elementwise/SumReduce/ScalarMul — closing the "two parallel shape-derivations" gap: `rows`/`cols`
    // downstream come FROM the TileOp, not a second independent read of `node.output.region`. The
    // internal decomposition below (silu then multiply, 2 separate SdscOps, each ALREADY TileOp-routed
    // via `pointwise_broadcast_opspec_df`) is UNCHANGED — this only replaces how `rows`/`cols` are
    // obtained, not what gets emitted.
    check_pointwise_cols(
        node.output.region.cols.len,
        "SiluMul",
        node.output.tensor.index() as u32,
    )?;
    let tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    let rows = tile_op
        .dims
        .iter()
        .find(|d| d.name == "mb")
        .map(|d| d.size)
        .unwrap_or(1);
    let cols = tile_op
        .dims
        .iter()
        .find(|d| d.name == "out")
        .map(|d| d.size)
        .unwrap_or(1);
    let gate = ds_name(&node.inputs[0]);
    let up = ds_name(&node.inputs[1]);
    let out = ds_name(&node.output);
    let tmp_id =
        bundle::PlaceId::Act(node.output.tensor.index() as u32).synth(bundle::SynthRole::Silu);
    let tmp = syn(layout, tmp_id);
    // ⛔ DECLARE BEFORE USE — `silu(gate) -> tmp` then `multiply(tmp, up) -> out`, so `tmp` is a
    // real intermediate. It was never declared, so it took the bump allocator's shared address
    // alongside rope's.
    //
    // ⛔ AT THE OUTPUT TENSOR'S WIDTH, NOT THIS CHUNK'S. `cols` is the column block this call
    // lowers; `tmp` is the WHOLE intermediate, shared by every chunk exactly as gate/up/out are.
    // See [`BundleLayout::synth_like`] — declaring `[rows, cols]` reserved granite-3.1-8b's first
    // chunk (8192 of 12800) and the second chunk then wrote 9216 B past it.
    if let Some(l) = layout {
        l.synth_like(
            tmp_id,
            node.output.tensor.index() as u32,
            &[rows, cols],
            Df::Fp16,
        );
    }
    let silu_name = format!("silu_o{}", node.output.tensor.index() as u32);
    let mul_name = format!("mulsilu_o{}", node.output.tensor.index() as u32);
    // COLUMN-BLOCK OFFSETS — the tape splits a wide SiluMul (granite MLP intermediate
    // 12800) into col-blocks that SHARE one output tensor; each chunk's region carries
    // its `cols.start` (`ds_name` collapses to `t{tid}`, so WITHOUT this every chunk
    // writes/reads the whole-tensor base 0 → later blocks' columns left as ZEROS).
    // `gate`/`up`/`out` all share the same chunk offset (verified: the tape aligns their
    // regions). See [`pointwise_chunk_out_offset`] + `silumul_chunks_cover_output`.
    let gate_off = pointwise_chunk_out_offset(node.inputs[0].region.cols.start);
    let up_off = pointwise_chunk_out_offset(node.inputs[1].region.cols.start);
    let out_off = pointwise_chunk_out_offset(node.output.region.cols.start);
    // The FULL tensor width, not this chunk's. `cols` is the chunk extent (the op's view); the
    // gate/up/tmp/out tensors are the whole intermediate, and a column corner must be taken against
    // the storage that actually holds it. Passing `cols` made the second chunk of granite-3.1-8b
    // (start 8192, width 4608 of a 12800-wide intermediate) address column 8192 of a 4608-wide
    // tensor — caught by the footprint guard. This is the same "op's view standing in for the
    // tensor's storage" the address module exists to stop, committed inside that module's own
    // migration.
    let full_cols = node.output.region.cols.start + node.output.region.cols.len;
    // HANDLE-FLOW: bind gate/up/tmp/out once; `tmp` flows from the silu output INTO the mul input.
    let gate = rb(&gate, rows, cols);
    let up = rb(&up, rows, cols);
    let tmp = rb(&tmp, rows, cols);
    let out = rb(&out, rows, cols);
    Ok(vec![
        assemble_pointwise_broadcast_off(
            &silu_name,
            "silu",
            scratchy_subtile::sdsc_abstract::RowCount::of_token_rows(rows),
            scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(cols),
            &[In::sliced(
                &gate,
                scratchy_subtile::addr::col_of(rows, full_cols, gate_off, Df::Fp16),
            )
            .ew()],
            &tmp,
            // The WRITE needs the same nest the READS above go through. A raw `cols_start` is the
            // flat element index; in the stick-blocked output, column `c` of a `[rows, cols]` tensor
            // starts at `(c/64)*(rows*64)`. Equal only at rows == 1, so decode was right and prefill
            // wrote every later column block on top of the first one.
            scratchy_subtile::addr::col_of(rows, full_cols, out_off, Df::Fp16),
            sym_id_base,
            layout,
        ),
        assemble_pointwise_broadcast_off(
            &mul_name,
            "multiply",
            scratchy_subtile::sdsc_abstract::RowCount::of_token_rows(rows),
            scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(cols),
            &[
                In::sliced(
                    &tmp,
                    scratchy_subtile::addr::col_of(rows, full_cols, out_off, Df::Fp16),
                )
                .ew(),
                In::sliced(
                    &up,
                    scratchy_subtile::addr::col_of(rows, full_cols, up_off, Df::Fp16),
                )
                .ew(),
            ],
            &out,
            scratchy_subtile::addr::col_of(rows, full_cols, out_off, Df::Fp16),
            sym_id_base,
            layout,
        ),
    ])
}

/// Lower a whole [`SubOp::RmsNorm`] by DECOMPOSING into the 6-op sequence IBM's
/// `torch_spyre` uses (`decompositions.py:409 spyre_rms_norm`):
///   sq=x·x → mean=mean(sq) → meps=mean+eps → inv=rsqrt(meps) → tmp=x·inv → y=tmp·gamma
/// `inputs[0]`=x `[m,cols]`, `inputs[1]`=gamma `[1,cols]`, `eps` is the attr. The
/// reduce produces `[m, one-stick]`; `inv` is broadcast over `out` in the scale
/// step (RedStick, the on-card-proven `alpha_=0` broadcast read); `gamma` is
/// broadcast over rows (`mb`=-1); `eps` is a `[1,1]` scalar INPUT (its value is a
/// runtime const, provisioned like a weight — #51). Synthetic intermediates are
/// allocated by the coloring pass (like `lower_silumul_node`'s `<out>_silu`).
fn lower_rmsnorm_node<F: RopeForm>(
    node: &SubtileNode<F>,
    eps: f32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "RmsNorm t{} expects 2 inputs (x, gamma), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    // Route the ENTRY through TileIR (node_to_single_tile_op), same as SiluMul/Elementwise/SumReduce/
    // ScalarMul: check_pointwise_cols validates the RAW node.output cols first, then rows/cols downstream
    // come FROM the TileOp, not a second independent read. The internal 6-op decomposition below (each
    // ALREADY TileOp-routed via the split matmul/pointwise/reduce builders) — and, critically, the
    // decode-vs-prefill ALGORITHM CHOICE it makes (assemble_rmsnorm vs assemble_rmsnorm_mq) — is
    // completely UNCHANGED. This swap touches ONLY how rows/cols are obtained.
    check_pointwise_cols(
        node.output.region.cols.len,
        "RmsNorm",
        node.output.tensor.index() as u32,
    )?;
    let tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    let rows = tile_op
        .dims
        .iter()
        .find(|d| d.name == "mb")
        .map(|d| d.size)
        .unwrap_or(1);
    let cols = tile_op
        .dims
        .iter()
        .find(|d| d.name == "out")
        .map(|d| d.size)
        .unwrap_or(1);
    // rms_norm_eps (config) flows via the scalarmul registry (compute_bundle_layout collected it). Look
    // up its reserved [1,1] const tid so the rmsnorm adds the REAL config eps to the mean-of-squares
    // (previously dropped, `let _ = eps`). Missing ⇒ build Err (unreachable: collector added every eps).
    let eps_idx = layout
        .and_then(|l| l.scalarmul_scales.iter().position(|s| s.to_bits() == eps.to_bits()))
        .ok_or_else(|| {
            SuperDscError(format!(
                "RmsNorm t{}: eps {eps} absent from BundleLayout.scalarmul_scales (registry desync)",
                node.output.tensor.index() as u32
            ))
        })?;
    let eps_const = crate::wiring::act_name(scalarmul_scale_tid(eps_idx));
    let x = ds_name(&node.inputs[0]);
    let gamma = ds_name(&node.inputs[1]);
    let t = node.output.tensor.index() as u32;
    Ok(assemble_rmsnorm(
        &format!("o{t}"),
        rows,
        cols,
        &x,
        &gamma,
        bundle::PlaceId::Act(t),
        &eps_const,
        sym_id_base,
        layout,
    ))
}

/// Lower a [`SubOp::RopeRotate`] / the rotate of [`SubOp::RopeAppend`] to in-bundle
/// SuperDSC ops via the PERMUTATION-MATMUL form (NeoX). The 32-wide rotate-half
/// would violate the 64-fp16-stick constraint, so instead `rot = x·P` where `P` is a
/// fixed `[hd,hd]` sign-permutation (`P[i+half,i]=-1` for i<half, `P[i-half,i]=+1`
/// for i≥half) — a 64-stick-aligned matmul. Then `out = x·cos + rot·sin`. `inputs` =
/// (x, cos, sin); `x` `[mq, heads·hd]` (row-major). One `mb=1 [1,hd]` block is emitted
/// per (row `r`, head `h`) at `rope_prefill_block_offset(r,h,total,hd)`; cos/sin are the
/// worker's `[mq, total]` per-position head-tiled table, read at `rope_prefill_cos_offset(r,total)`
/// (row `r`'s head-0 slice serves every head). `P` is the resident `t{ROPE_P_TID}` weight.
/// Decode (mq=1) reduces to the original per-head loop, byte-identical.
/// ⭐⭐ `HD` IS A CONST GENERIC HERE, and that is the point of this function's whole shape.
///
/// The head dim is a constant of the MODEL, but this emitter is ONE binary serving every model, so inside
/// it the value only becomes known when the proc macro runs. That is why `Shape::<0,0,0,0>` and the
/// `_of(hd, df)` twins existed: an escape hatch for "the const generic cannot be used here". The escape
/// hatch is what made the head-dim-dependent fork below a RUNTIME branch, and a runtime branch is what no
/// compile-time guard can hold onto.
///
/// The fix is a SINGLE dispatch from the value to the const (see the caller), after which everything here
/// is const. `shape.rs`'s own module doc asked for exactly this: "any branch on them would have to be
/// written as a branch on a const, which shows up as a special case in review instead of hiding inside an
/// offset expression."
///
/// What it buys immediately: the collapsed RoPE form and the slab RoPE form become TWO INSTANTIATIONS
/// rather than two arms of one function, so "a head_dim-128 bundle takes the slab path" is a fact the
/// compiler knows.
fn lower_rope_node<F: RopeForm, const HD: u32>(
    node: &SubtileNode<F>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    // ⭐ THE ROW KIND, THREADED. It used to be read from a thread-local `Cell<bool>` set once per
    // bundle, so this function could not be reasoned about locally and no caller was forced to state
    // which kind it was emitting. That ambient carrier is deleted; see `sdsc_abstract::QueryRows`.
    //
    // It matters HERE specifically: at `hd == stick` the collapsed form below is chosen and it is the
    // ONLY branch in this function that knows a row might be an independent sequence. At hd > stick
    // (head_dim 128) `head_major_collapse_valid` is FALSE, the collapse is skipped, and a decode batch
    // falls into the path commented "PREFILL (mq>1) hd>stick: SLAB rope" — whose rows are consecutive
    // positions of ONE sequence. That is correct today only because the worker stages cos/sin per
    // REQUEST and the two layouts agree by construction; nothing in a type says so.
    rows_are_requests: bool,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    // RopeRotate = 3 inputs (x, cos, sin); RopeAppend = 6 inputs
    // [K, cos, sin, V, K_cache, V_cache] (subtile_ir.rs:962/967) — its rotation math
    // is IDENTICAL (rotate inputs[0] with cos/sin); the trailing V/K_cache/V_cache are
    // the cache-write operands, consumed by AttnDecode (new_v = the V tensor; the
    // caches = AttnDecode's prefix_k/prefix_v), NOT part of the rotation. So accept
    // either arity and rotate inputs[0..3] only.
    if node.inputs.len() != 3 && node.inputs.len() != 6 {
        return Err(SuperDscError(format!(
            "Rope t{} expects 3 (RopeRotate) or 6 (RopeAppend) inputs, found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let hd = HD;
    // Validate the RAW (pre-TileOp) cols first — the head_dim divisibility check must run before
    // anything is re-derived. Route the ENTRY through TileIR (node_to_single_tile_op) right after, same
    // as SiluMul/RmsNorm/Elementwise/SumReduce/ScalarMul: `total`/`mq` downstream come FROM the TileOp
    // (whose RopeRotate/RopeAppend mapping is `dims=[mb=rows,out=cols,y=1]` verbatim, no rounding — see
    // subtile_tape_to_tile_ir.rs — so these are byte-identical to the raw region reads). The per-(row,
    // head[,slab]) host-loop decomposition below is COMPLETELY UNCHANGED (Phase B, deferred).
    let total = node.output.region.cols.len;
    if hd == 0 || !total.is_multiple_of(hd) {
        return Err(SuperDscError(format!(
            "RopeRotate t{}: output cols {total} not a multiple of head_dim {hd}",
            node.output.tensor.index() as u32
        )));
    }
    let tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    let mq = tile_op
        .dims
        .iter()
        .find(|d| d.name == "mb")
        .map(|d| d.size)
        .unwrap_or(1);
    let total = tile_op
        .dims
        .iter()
        .find(|d| d.name == "out")
        .map(|d| d.size)
        .unwrap_or(total);
    let heads = total / hd;
    // ROW COUNT: 1 = decode, mq>1 = PREFILL (a [mq, total] roped Q/K). The rope emits one mb=1 [1,hd]
    // block per (row r, head h) — the PROVEN mb=1 primitive. Looping ONLY over heads (mq implicitly 1)
    // was THE prefill bug: it wrote row 0 for every head and NEVER rows 1..mq → roped Q/K collapsed to
    // row 0 → K-cache 1 slot → garbage (measured on-card). CRUCIAL: a [mq,total] activation is
    // STICK-SCATTERED (dev_off: [r,c]→(c/64)·(mq·64)+r·64+c%64), so the (r,h) block's DEVICE offset is
    // rope_prefill_block_offset(r,h,mq,hd)=h·mq·hd+r·hd, NOT the naïve flat r·total+h·hd (that wrote
    // where the attention matmul never reads for r>0 — a baked-but-still-1-row false start). Kani
    // `rope_prefill_bijection_*`: off == dev_off([mq,total],1,[r,h·hd]) (producer==consumer) + bijection.
    // hd>stick in PREFILL: head h spans hd/stick NON-contiguous device sticks (each row r-scattered by
    // mq·stick), so a contiguous [1,hd] block can't address it — hd>stick prefill takes the SLAB path
    // below (per-(row,head,slab) [1,stick] ops). The rotate-half is realized there as a signed slab-swap
    // (rot slab s = ±x slab s±n_slabs/2), which needs half=hd/2 to be a whole number of sticks, i.e.
    // n_slabs even (hd a multiple of 2·stick) — true for every real hd>64 (128/256/512). Guard only the
    // odd-n_slabs case (never a real head_dim) as a build error (not an on-card crash).
    if hd > Fp16::ELEMS_PER_STICK && !(hd / Fp16::ELEMS_PER_STICK).is_multiple_of(2) {
        return Err(SuperDscError(format!(
            "RoPE prefill (t{t}, mq={mq}): head_dim {hd} > stick {stk} with an ODD stick count ({ns}); \
             the rotate-half slab-swap needs half=hd/2 to be a whole number of sticks (hd a multiple of \
             2·stick). Real head_dims (128/256/512) satisfy this.",
            t = node.output.tensor.index() as u32,
            stk = Fp16::ELEMS_PER_STICK,
            ns = hd / Fp16::ELEMS_PER_STICK,
        )));
    }
    let x = ds_name(&node.inputs[0]);
    let cos = ds_name(&node.inputs[1]);
    let sin = ds_name(&node.inputs[2]);
    let out = ds_name(&node.output);
    let t = node.output.tensor.index() as u32;
    // DIAG (2026-07-07): localize the roped-Q/K row-0 collapse. Prints the rope's input (proj
    // output) tid, output tid, and mq. If x (proj out) is 8 rows but out (roped) is 1 row ⇒ rope
    // collapses (mq wrong / emission). If x is 1 row ⇒ the proj matmul mb>1 collapses upstream.
    let p = crate::wiring::act_name(ROPE_P_TID);
    use bundle::SynthRole as R;
    let out_id = bundle::PlaceId::Act(t);
    let rot = syn(layout, out_id.synth(R::Rot)); // x · P  (rotate-half, synthetic seg3)
    let xc = syn(layout, out_id.synth(R::Xc)); // x · cos
    let rs = syn(layout, out_id.synth(R::Rs)); // rot · sin
    // PER-(row,head) [1,hd] at the DEVICE offset rope_prefill_block_offset(r,h,mq,hd) — the roped Q/K
    // tensor is [mq, heads·hd] sticked on its last dim, so head h row r's hd dims are one 64-stick at
    // h·mq·hd+r·hd (hd==STK). A single FULL [mq,heads,hd] op would device-stick differently and mismatch
    // the attention matmul's [mq,total] read; per-(r,h) [1,hd] blocks match it exactly (Kani-proven).
    // cos/sin are the worker's [mq,total] per-position head-tiled table, read at row r's head-0 slice.
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(mq as usize * heads as usize * 4);
    let ew_ph = |ops: &mut Vec<EmittedOp>,
                 name: String,
                 f: &'static str,
                 a: &str,
                 aoff: scratchy_subtile::addr::DevOff,
                 b: &str,
                 boff: scratchy_subtile::addr::DevOff,
                 o: &str,
                 ooff: scratchy_subtile::addr::DevOff,
                 sib: &mut i64|
     -> Result<(), SuperDscError> {
        // MULTI-STICK AT head_dim > 64, AND THAT IS A REAL DIFFERENCE: at rows==1 a `[1,hd]` op is two
        // sticks at head_dim 128, and the work splitter divides a multi-stick `out` ACROSS CORES — 2
        // cores where head_dim 64 uses 1. No working model has run these ops in that mode, and this
        // file documents a dxp defect in exactly that regime (`cols > 64` ⇒ split across cores).
        //
        // Splitting them stick-wide the way prefill's RoPE already is does NOT work: `rot` is produced
        // by a `[1, hd]` matmul, so reading it as `[1, stick]` slices declares a second arrangement for
        // one tensor and the arrangement authority rejects the bundle. Making decode stick-wide means
        // making the P matmul stick-wide too, which is a different rotate formulation, not a slice.
        let op = pointwise_broadcast_opspec(
            op_func_from_str(f),
            1,
            hd,
            &[
                In::sliced(&rbo(a), aoff).ew(),
                In::sliced(&rbo(b), boff).ew(),
            ],
            o,
            ooff.into_raw_elems(),
            false, // RoPE per-(row,head) [1,hd] block (rows==1): rank-3 flat
        )
        .map_err(SuperDscError)?;
        ops.push(emit_sdsc_tiled(
            &name,
            &op,
            &SdscFoldSet::new(op.iter.cores_used()),
            sib,
            layout,
        )?);
        Ok(())
    };
    // ── DECODE (mq==1) HEAD-BATCH (restored 2026-07-28, PERF): collapse the per-head loop into ONE
    // [heads,hd] op each, PROVEN byte-identical to the general per-(row,head) loop below at mq=1 —
    // this was in the original hardware-proven flash decode (superdsc-batch-perf), removed during the
    // Islands/Bridges unification "to eliminate a second, independently-maintained implementation"
    // (a real, deliberate tradeoff, not a bug) at the cost of ~4x more RoPE ops for decode specifically
    // (measured: 40 per-(row,head) ops per op-type vs 1 batched op, ~150 extra ops/layer — the single
    // largest unaccounted-for chunk of the perf gap versus the old proven code). Restoring it as a
    // pure batching optimization (not an mq=1 correctness special-case: the general loop below is
    // UNCHANGED and still handles every other mq/hd combination identically): when mq==1 there is NO
    // row-scatter, so the [1,total] roped tensor is BYTE-IDENTICAL to a [heads,hd] view at hd==stick
    // (dev_off([1,total],0,h·hd+d) == dev_off([heads,hd],h,d) == h·64+d — the per-(row,head) block
    // offset h·mq·hd+r·hd degenerates to h·hd at mq=1, contiguous). So head h is row h, and the
    // head-independent cos/sin/P kernels broadcast across the head rows.
    // ── HEAD-MAJOR COLLAPSE, AT ANY ROW COUNT ───────────────────────────────────────────────────
    // This used to be `mq == 1` only, and the note below measures what that costs: ~40 per-(row,head)
    // ops per op-type, ~150 extra ops a layer, "the single largest unaccounted-for chunk of the perf
    // gap". It is also a fixed cost — the body goes 271 ops a layer at one request to 372 at two, and
    // then only +16 per further request — so a batch pays it in full and amortizes none of it.
    //
    // The collapse generalizes. Head `h` row `r` of the roped `[mq, heads*hd]` tensor sits at
    // `h*mq*hd + r*hd`, and a `[heads*mq, hd]` view indexed `h*mq + r` gives exactly `(h*mq+r)*hd` —
    // the SAME bytes at any `mq`, which is `addr_eq`'s row-expansion law (Kani
    // `row_expansion_is_byte_identical`), not an mq=1 coincidence.
    //
    // cos/sin need one change: at one row every head shares the angles, so they are read mb-broadcast
    // from a single row; above one row they vary per POSITION and must be read per-row. The worker's
    // staging already suits it — element `(p, h*hd+d)` lands at `h*mq*64 + p*64 + d`, which IS row
    // `h*mq+p`, column `d` of the `[heads*mq, hd]` view.
    //
    // Scoped to a decode batch: prefill's own bundles are proven on hardware and stay byte-identical.
    let collapse_rows = if mq == 1 {
        Some(heads)
    } else if rows_are_requests {
        Some(heads * mq)
    } else {
        None
    };
    if let Some(rows) = collapse_rows
        // ⭐ FROM THE TYPE. `HD` is a const generic now, so this is a compile-time constant and the two
        // RoPE forms below are two instantiations of this function rather than two runtime arms.
        .filter(|_| {
            scratchy_subtile::addr::Shape::<HD, 0, 0, 0>::head_major_collapse_valid_here(Df::Fp16)
        })
    {
        // ⛔ DECLARE BEFORE USE. This branch (`hd == stick`, the head-major collapse) referenced
        // `rot`/`xc`/`rs` without ever declaring them, so they fell to `resolve_seg_base`'s bump —
        // which handed all three the SAME address. Rope is `rot = x·P`, `xc = x·cos`,
        // `rs = rot·sin`, `out = xc + rs`: three of its four intermediates were one buffer.
        if let Some(l) = layout {
            for r in [R::Rot, R::Xc, R::Rs] {
                l.synth(out_id.synth(r), &[rows, hd]);
            }
        }
        // rot = x[rows,hd] @ P[hd,hd] — the same P kernel for every row (m=rows GEMM, k=n=hd).
        ops.push(assemble_matmul_off(
            &format!("rope_rot_o{t}"),
            scratchy_subtile::sdsc_abstract::MatM::of_head_major_rows(rows),
            scratchy_subtile::sdsc_abstract::MatN::of_head_dim(hd),
            scratchy_subtile::sdsc_abstract::MatK::of_head_dim(hd),
            scratchy_subtile::sdsc_abstract::MatY::unbatched(),
            crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                bmm_site::TapeLoweringSite::witness(),
            ),
            &rb(&x, rows, hd),
            scratchy_subtile::addr::DevOff::ZERO,
            &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
            scratchy_subtile::addr::DevOff::ZERO,
            &rb(&rot, rows, hd),
            scratchy_subtile::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        let hh = |name: &str| fl(name, rows, hd);
        // One row of angles shared by every head at mq==1; per-row above that.
        let (cos_h, sin_h) = (hh(&cos), hh(&sin));
        let cos_in = if mq == 1 {
            In::mb_at(&cos_h, scratchy_subtile::addr::DevOff::ZERO)
        } else {
            In::full(&cos_h)
        };
        let sin_in = if mq == 1 {
            In::mb_at(&sin_h, scratchy_subtile::addr::DevOff::ZERO)
        } else {
            In::full(&sin_h)
        };
        // The collapse's typed extents: `rows` is the head-major-collapsed view's row extent
        // (`heads*mq`, where `heads` may be nqh or nkvh), swept head-dim wide.
        let hm_rows = scratchy_subtile::sdsc_abstract::RowCount::of_head_major_rows(rows);
        let hd_cols = scratchy_subtile::sdsc_abstract::BlockCols::of_head_dim(hd);
        ops.push(pw2(
            &format!("rope_xc_o{t}"),
            "multiply",
            hm_rows,
            hd_cols,
            In::full(&hh(&x)),
            cos_in,
            &hh(&xc),
            sym_id_base,
            layout,
        ));
        ops.push(pw2(
            &format!("rope_rs_o{t}"),
            "multiply",
            hm_rows,
            hd_cols,
            In::full(&hh(&rot)),
            sin_in,
            &hh(&rs),
            sym_id_base,
            layout,
        ));
        ops.push(pw2(
            &format!("rope_add_o{t}"),
            "add",
            hm_rows,
            hd_cols,
            In::full(&hh(&xc)),
            In::full(&hh(&rs)),
            &hh(&out),
            sym_id_base,
            layout,
        ));
        return Ok(ops);
    }
    // ── PREFILL (mq>1) hd>stick: SLAB rope ──────────────────────────────────────────────────────────
    // A head's hd dims span n_slabs = hd/stick device STICKS, each a [1,stick] block r-scattered by
    // mq·stick in the stick-scattered [mq,total] tensor — a contiguous [1,hd] block (and thus the [hd,hd]
    // P-matmul) can't address them. So rope per-(row r, head h, slab s): out_s = x_s·cos_s + rot_s·sin_s,
    // where the rotate-half `rot_s` = (s < n/2 ? -x_{s+n/2} : +x_{s-n/2}) is a SIGNED SLAB-SWAP — half=hd/2
    // is n_slabs/2 whole sticks (guarded even above), so no cross-stick P-matmul is needed. All offsets are
    // dev_off([mq,total],1,·) (Kani dev_off); cos/sin read head-0's slab s (the head-tiled table serves
    // every head). At mq==1 or hd==stick this reduces to the loop below (one slab), so hd>stick is the
    // only caller. rot_s's sign is folded into the final add/subtract (rs_s carries the magnitude).
    // ANY `mq`, not just prefill. The slab form emits STICK-WIDE ops; the `[1, hd]` form below is
    // MULTI-STICK at head_dim > 64, and the work splitter divides a multi-stick `out` across CORES —
    // 2 cores at head_dim 128 where head_dim 64 uses 1. That per-core mode is one no working model has
    // run these ops in, and this file documents a dxp defect in exactly it. The slab form also drops
    // the `[hd,hd]` P matmul entirely: the rotate becomes a signed SLAB SWAP, so there is no `rot`
    // tensor and no worker-staged permutation on this path at all.
    //
    // The body is already row-batched over `mq`, so `mq == 1` is one row and needs no special case.
    // head_dim 64 never reaches here (`hd > stick` is false), so granite-3.1-2b is untouched.
    if hd > Fp16::ELEMS_PER_STICK {
        let stk = Fp16::ELEMS_PER_STICK;
        let (hd_u, stk_u) = (hd as usize, stk as usize);
        let n_slabs = hd_u / stk_u;
        // `dev_off(&[mq,total],1,[r,col])` IS `Nest(["row","feat"],[mq,total])`; going through the
        // nest means this loop names a COORDINATE and never writes a stride.
        let doff = |r: usize, col: usize| {
            scratchy_subtile::addr::rc_of(mq, total, r as u32, col as u32, Df::Fp16)
        };
        let ew_slab = |ops: &mut Vec<EmittedOp>,
                       name: String,
                       f: &'static str,
                       a: &str,
                       aoff: scratchy_subtile::addr::DevOff,
                       b: &str,
                       boff: scratchy_subtile::addr::DevOff,
                       o: &str,
                       ooff: scratchy_subtile::addr::DevOff,
                       sib: &mut i64|
         -> Result<(), SuperDscError> {
            let op = pointwise_broadcast_opspec(
                op_func_from_str(f),
                mq,
                stk,
                &[
                    In::sliced(&rbo(a), aoff).ew(),
                    In::sliced(&rbo(b), boff).ew(),
                ],
                o,
                ooff.into_raw_elems(),
                // cols == stk, and `stickmajor` needs cols > 64, so it cannot fire either way — this
                // is the same rank-3 flat form regardless of the row count.
                true,
            )
            .map_err(SuperDscError)?;
            ops.push(emit_sdsc_tiled(
                &name,
                &op,
                &SdscFoldSet::new(op.iter.cores_used()),
                sib,
                layout,
            )?);
            Ok(())
        };
        // ROW-BATCHED over `mq`, exactly as the hd==stick path below does. For a FIXED (head h,
        // slab s) the mq rows are CONTIGUOUS: `rc_of(mq,total,r,h*hd+s*stk)` has column term
        // `(h*hd+s*stk)/stk = h*n_slabs+s` (hd is a whole number of sticks, so the residue is 0),
        // giving `(h*n_slabs+s)*mq*stk + r*stk` — i.e. row r of a standalone [mq,stk] block based at
        // r=0. So one [mq,stk] op replaces mq of them, per operand, and the addresses are the SAME
        // bytes the per-row form emitted. Ops go from 3*mq*heads*n_slabs to 3*heads*n_slabs (granite
        // 8b prefill at mq=31: 7440 -> 240). The row loop is gone, so `doff` is evaluated at r=0 and
        // the op's own row stride supplies the rest.
        // ⭐⭐⭐⭐⭐ THREE WHOLE-TENSOR OPS + ONE ROTATE PER HEAD, instead of 3·heads·n_slabs stick blocks.
        // At granite-8b (hd=128, nqh 32 + nkvh 8) that is 240 ops a layer → 43, and RoPE was the single
        // largest op family in the decode body (38% of its 611 non-weight ops).
        //
        // ⭐ WHY IT IS WORTH THE OPS AND NOTHING ELSE: fitting both models' measured layer time against
        // their op counts and `wstride` gives `layer_ms = ops·0.83 µs + weight_MB / 137 GB/s`, and the
        // weight stream is already at 91% of the 150 GB/s IBM's own cost model fits as peak. So per-op
        // overhead is the ONLY bs=1 lever left, and 197 fewer ops a layer is ~6.5 ms/token at 8b.
        //
        // ⭐ MEASURED, and NOT where I expected: granite-8b, alternating A/B against 55648f68, +4.4%
        // median tok/s and the new tree wins ALL TEN PAIRS. But the ITL MODES DO NOT MOVE (~82.4 slow /
        // ~78.8 fast in both trees) — what changes is how often the bundle draws the FAST weight-segment
        // region (1/10 -> 4/10). So this collapse pays through the 4% bimodality
        // ([[decode-itl-is-bimodal-because-of-the-weight-segments-region]]), not by saving per-op
        // compute, and the same effect in reverse is the whole of the cache-write collapse's apparent
        // ~2 ms "regression" (`c1b7e3b7`, which draws fast 1/10 against the baseline's 5/12).
        //
        // ⛔ SO DO NOT PRICE A COLLAPSE IN MACs. `c1b7e3b7`'s message blames its 8·64·64 identity-matmul
        // MACs; the mode data says that explanation is unsupported. This one ADDS a `[mq,hd]·[hd,hd]` P
        // matmul per head (655 K MACs a layer at 8b, against ~200 M for the projections) and wins. Judge
        // a collapse on LAUNCHES and on the region draw, and measure it paired.
        //
        // THE ROTATE IS THE P MATMUL AGAIN, which this path had dropped in favour of a signed slab
        // swap. The swap is what forced the per-block form: it reads slab `s ± n_slabs/2`, a
        // PERMUTATION of the stick planes, and a pointwise op has one offset and one uniform stride per
        // operand — so the read cannot be expressed whole-tensor (the same rank-2 wall the finalize and
        // the mask hit). P carries the swap AND its sign inside a kernel, so every consumer downstream
        // reads its own block: `rs = rot·sin` and `out = xc + rs` become plain elementwise ops, and the
        // ∓ that used to split the combine by half is gone with it.
        //
        // ⛔ THE ARRANGEMENT IS WHY THIS SHAPE AND NOT THE ONE THE `[1,hd]` PATH REJECTED. That one
        // declared `rot` as `[1,hd]` (the matmul) and then read `[1,stick]` SLICES of it — two
        // arrangements for one tensor, which the arrangement authority refuses. Every operand here is
        // declared at the tensor's OWN `[mq,total]` extent and narrowed by OFFSET, exactly as
        // `attn_krep` reads its kv-head window: a stick-aligned column window of a `[mq,total]` tensor
        // has its stick groups `mq·64` apart, which IS a standalone `[mq,hd]` tensor's own law, so the
        // matmul needs no second view and no `phys_mb`.
        //
        // At hd == stick this branch is not taken at all (the head-major collapse above owns that case,
        // and granite-2b is untouched byte-for-byte).
        let _ = &ew_slab; // the per-block emitter stays for the paths below
        // Declare the intermediates at their TRUE full extent, for the reason the mq>1 branch below
        // records: sized by whichever access declares first, a per-head matmul window would reserve ONE
        // head and under-reserve by `heads` — the bump-allocator defect `synth` exists to kill.
        if let Some(l) = layout {
            for r in [R::Rot, R::Xc, R::Rs] {
                l.synth(out_id.synth(r), &[mq, total]);
            }
        }
        // ⭐⭐⭐⭐⭐ ONE ROTATE PER SLAB, EVERY HEAD ON `y` — `nslab` ops, not `heads` (2 at granite-8b's
        // hd=128, 8 at gemma-4's hd=512, against 40 either way before).
        //
        // ⭐ P'S OWN STRUCTURE IS WHAT ALLOWS IT, and it is exactly one block per output slab:
        // `rope_p_entry` is nonzero only at `inn == o ± hd/2`, so output slab `s` draws from the SINGLE
        // input slab `s ∓ nslab/2` through a `[64,64]` block that is ±I. There is no sum over input
        // slabs to accumulate, so a one-stick contraction covers the whole rotate — and the block is read
        // straight out of the ALREADY-STAGED `[hd,hd]` P const at its own (in-slab, out-slab) coordinate,
        // so this needs no new constant and no sign folded anywhere.
        //
        // ⭐ AND BOTH OPERANDS ARE THE SAME FRAMING, which is why the `y` batch is the easy case here:
        // `x` and `rot` are both the `[mq,total]` token stream, so both declare
        // `of_token_stream_by_slab`'s pitch `mq*nslab` and both really do have their heads `mq*hd` apart.
        // The cache write's pair (`mq` rows against `PLANE_SLOTS`) is what needed per-operand pitches;
        // this one needs them to AGREE, and they do.
        //
        // ⛔ HEADS ON `y` AT A FIXED SLAB IS THE SHAPE THAT SURVIVED THE CARD. The slab stays a
        // coordinate of the offsets — folding it into `y` is what produced garbage on the cache write
        // ([[cachewr-y-cannot-cross-a-feature-slab]]), and nothing here retries that.
        //
        // ⛔ WHY THIS IS WORTH DOING WHEN THE ROPE COLLAPSE ITSELF DID NOT MOVE ITL: the per-group timer
        // on the REAL bundle attributes 0.616 ms/layer to group 0 (rmsq → qkv → RoPE → attention →
        // finalize) while that group holds only 25.2 MB of weights (0.18 ms at the measured 139 GB/s
        // marginal), so ~0.44 ms/layer there is OP work. The previous step traded 197 cheap pointwise
        // trips for 40 MATMUL trips, which is why it was an ITL wash; this removes 38 of those matmuls
        // without adding anything. MEASURED: granite-8b slow-mode ITL 82.6 -> 79.5 ms, 10/10 paired wins.
        //
        // ⛔⛔⛔ DECODE ONLY (`mq == 1`), AND THE PREFILL EVIDENCE IS WHY. Emitted at EVERY `mq`, this form
        // is measured WRONG on granite-8b's prompt: the reply reads fluently but says the PROMPT looks
        // jumbled ("the text is a bit jumbled, I'll help you rephrase it"), and TTFT DOUBLES (279.7 vs
        // 124.7 ms) while the decode ITL improves — a mis-encoded prompt that decode then continues
        // fluently from. The addresses are not the cause and that is checked, not assumed:
        // `tests/zz_rope_rot_addr_equiv.rs` proves the ±I block is the right (in-slab, out-slab)
        // coordinate (P is NOT symmetric, so a transposed one would silently negate the rotation), that
        // the slab form reproduces the per-head rotate element for element, and that the offsets step one
        // plane per slab and `mq*hd` per head. So the defect is in the FORM at `mq > 1` — the same shape
        // of finding as the score leg's ("every emitted address is correct and decode is still
        // incoherent"), and the row axis being degenerate at `mq == 1` is exactly what hides it.
        //
        // ⏭ WHAT TO TRY NEXT, in order: `y` = 32 heads is WIDER THAN ANY PROVEN y-batch on this backend
        // (the cache write runs y=8, the score leg y=4 or y=nkvh*nslab=16), and at `mq > 1` the
        // batch-divides-first split hands one core per head and leaves M unsplit — so re-test at y=8
        // (four ops per slab, still 8 against 40) before concluding the form cannot serve prefill.
        let rot_slab_form = mq == 1;
        for s in (0..n_slabs).filter(|_| rot_slab_form) {
            // P's nonzero block for this output slab: `rope_p_entry` pairs `o` with `o ± hd/2`, and
            // `hd/2` is `n_slabs/2` whole sticks (guarded even above), so the partner is a whole slab.
            let p_in = if s < n_slabs / 2 {
                s + n_slabs / 2
            } else {
                s - n_slabs / 2
            };
            // The ±I block as a COORDINATE on P's own `[in, out]` kernel nest — the same law `attn_krep`
            // reads the identity's diagonal block through, never the product it works out to.
            let p_off = scratchy_subtile::addr::Nest::new(&["row", "feat"], &[hd, hd], Df::Fp16)
                .view()
                .at(
                    scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Row>::n(
                        p_in as u32 * stk,
                    ),
                )
                .slab(s as u32)
                .dev();
            let a_place =
                scratchy_subtile::sdsc_abstract::OperandPlacement::of_token_stream_by_slab(
                    scratchy_subtile::sdsc_abstract::QueryRowCount::of_mq(mq),
                    heads,
                    hd,
                    0,
                    p_in as u32,
                    Df::Fp16,
                );
            let o_place =
                scratchy_subtile::sdsc_abstract::OperandPlacement::of_token_stream_by_slab(
                    scratchy_subtile::sdsc_abstract::QueryRowCount::of_mq(mq),
                    heads,
                    hd,
                    0,
                    s as u32,
                    Df::Fp16,
                );
            ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                &if n_slabs == 1 {
                    format!("rope_rot_o{t}")
                } else {
                    format!("rope_rot_s{s}_o{t}")
                },
                scratchy_subtile::sdsc_abstract::MatM::of_query_rows(
                    scratchy_subtile::sdsc_abstract::QueryRowCount::of_mq(mq),
                ),
                scratchy_subtile::sdsc_abstract::MatN::one_stick(
                    scratchy_subtile::sdsc_abstract::Lanes::FP16,
                ),
                scratchy_subtile::sdsc_abstract::MatK::one_stick(
                    scratchy_subtile::sdsc_abstract::Lanes::FP16,
                ),
                scratchy_subtile::sdsc_abstract::MatY::of_gqa_group(heads, a_place, o_place),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::of_head_batched(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(&x, mq, total),
                a_place,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
                p_off,
                &rb(&rot, mq, total),
                o_place,
                sym_id_base,
                layout,
            ));
        }
        // PREFILL KEEPS THE PER-HEAD P MATMUL — the form measured coherent on granite-8b (`7d27d734`),
        // one `[mq,hd]` window per head with the whole head dim contracted. `heads` ops instead of
        // `n_slabs`, and the prompt is right.
        for h in (0..heads as usize).filter(|_| !rot_slab_form) {
            let off_h = doff(0, h * hd_u);
            ops.push(assemble_matmul_off(
                &format!("rope_rot_h{h}_o{t}"),
                scratchy_subtile::sdsc_abstract::MatM::of_query_rows(
                    scratchy_subtile::sdsc_abstract::QueryRowCount::of_mq(mq),
                ),
                scratchy_subtile::sdsc_abstract::MatN::of_head_dim(hd),
                scratchy_subtile::sdsc_abstract::MatK::of_head_dim(hd),
                scratchy_subtile::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(&x, mq, total),
                off_h,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
                scratchy_subtile::addr::DevOff::ZERO,
                &rb(&rot, mq, total),
                off_h,
                sym_id_base,
                layout,
            ));
        }
        // The three elementwise legs, each over the WHOLE tensor: same bytes the per-block ops covered,
        // one op instead of `heads * n_slabs`. cos/sin are the worker's head-TILED table at this site's
        // full width (`tile_one` replicates the hd-wide row across `total/hd` heads), so a whole-tensor
        // read sees, for every head, exactly the head-0 slice the per-block form read.
        let rows_all = scratchy_subtile::sdsc_abstract::RowCount::of_query_rows(
            scratchy_subtile::sdsc_abstract::QueryRowCount::of_mq(mq),
        );
        let cols_all = scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(total);
        ops.push(pw2(
            &format!("rope_xc_o{t}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rbo(&x)),
            In::full(&rbo(&cos)),
            &rbo(&xc),
            sym_id_base,
            layout,
        ));
        ops.push(pw2(
            &format!("rope_rs_o{t}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rbo(&rot)),
            In::full(&rbo(&sin)),
            &rbo(&rs),
            sym_id_base,
            layout,
        ));
        // ADD, unconditionally: P already carries the rotate's sign, so there is no first-half/
        // second-half split left to make this op two ops (which the stick planes' interleaving would
        // have forced — first-half slabs are every OTHER plane, not a contiguous run).
        ops.push(pw2(
            &format!("rope_out_o{t}"),
            "add",
            rows_all,
            cols_all,
            In::full(&rbo(&xc)),
            In::full(&rbo(&rs)),
            &rbo(&out),
            sym_id_base,
            layout,
        ));
        return Ok(ops);
    }
    // ── PREFILL (mq>1) hd==stick: PER-HEAD ROW-BATCHED rope (2026-07-28, TTFT) ──────────────────
    // The general per-(row,head) loop below emits 4 ops per (row,head) — at granite prefill that is
    // mq(31)*nqh(32)*4 = 3968 ops for Q plus mq*nkvh*4 = 992 for K, i.e. ~83% of the ENTIRE 5990-op
    // prefill body. Decode dodges this via the mq==1 head-batch short-circuit above; prefill had no
    // equivalent.
    //
    // Heads CANNOT be batched at mq>1: head h row r lives at `h*mq*hd + r*hd`, so consecutive heads
    // are mq*hd apart while an op's own row stride is hd. But for a FIXED head the rows ARE
    // contiguous at stride hd, which is exactly what a [mq, hd] op wants:
    //   * x/rot/xc/rs/out: base `h*mq*hd`, internal dev_off(r,d) = r*hd + d  (cols=hd=stick, so the
    //     op is rank-3 flat and the address is base + r*hd + d) -> matches
    //     rope_prefill_block_offset(r,h,mq,hd) = h*mq*hd + r*hd exactly, for every r.
    //   * cos/sin: the worker stages them [mq,total] stick-scattered, so element (r,i) sits at
    //     (i/64)*(mq*64) + r*64 + (i%64); for head-0's slice (i<hd=64) that is just r*hd + i — a
    //     contiguous [mq,hd] block at offset 0. The table is head-tiled, so head 0's slice serves
    //     every head (same fact the per-row path relies on via rope_prefill_cos_offset).
    // So one [mq,hd] op per head replaces mq of them: 4*mq*heads -> 4*heads ops (Q: 3968 -> 128,
    // K: 992 -> 32). Mathematically identical work, just not unrolled over rows.
    // Gated to hd==stick (the multi-slab hd>stick case is handled by the slab path above) and to
    // mq>1, so decode (mq==1) never reaches here and its emission is untouched.
    // ⭐ FROM THE TYPE — `HD` is this function's const generic, so the gate is a compile-time constant.
    // ⛔ AND NOTE THE `mq > 1` TERM IS A *WIDTH* TEST, NOT A KIND TEST: it reads "more than one query row",
    // which is true of a prefill chunk AND of a decode batch. The comment below claims "decode (mq==1)
    // never reaches here", and that is FALSE for a BATCHED decode, whose mq is the batch width. The kind
    // is `rows_are_requests`; the count cannot stand in for it.
    if mq > 1
        && scratchy_subtile::addr::Shape::<HD, 0, 0, 0>::head_major_collapse_valid_here(Df::Fp16)
    {
        // Declare the rope intermediates at their TRUE full extent. Without this they are sized by
        // whichever access declares first, which was the per-head `[mq,hd]` matmul — ONE head, so the
        // tensor under-reserved by a factor of `heads` (the exact bump-allocator defect
        // `BundleLayout::synth` exists to kill). It also makes the whole-tensor `[heads*mq, hd]` reads
        // below the ones that speak for the layout, since a per-head access no longer covers the
        // footprint and so no longer declares. Scoped to the mq>1 branch so decode's allocation —
        // and therefore its bundle fingerprint — is untouched.
        if let Some(l) = layout {
            for r in [R::Rot, R::Xc, R::Rs] {
                l.synth(out_id.synth(r), &[heads * mq, hd]);
            }
        }
        let ew_rows = |ops: &mut Vec<EmittedOp>,
                       name: String,
                       f: &'static str,
                       rows: u32,
                       a: &str,
                       aoff: scratchy_subtile::addr::DevOff,
                       b: &str,
                       boff: scratchy_subtile::addr::DevOff,
                       o: &str,
                       ooff: scratchy_subtile::addr::DevOff,
                       sib: &mut i64|
         -> Result<(), SuperDscError> {
            let op = pointwise_broadcast_opspec(
                op_func_from_str(f),
                rows,
                hd,
                &[
                    In::sliced(&rbo(a), aoff).ew(),
                    In::sliced(&rbo(b), boff).ew(),
                ],
                o,
                ooff.into_raw_elems(),
                // cols == hd == stick, so `stickmajor` cannot fire (it needs cols>64) and this stays
                // the SAME rank-3 flat form the per-row path used — only the row count differs.
                false,
            )
            .map_err(SuperDscError)?;
            ops.push(emit_sdsc_tiled(
                &name,
                &op,
                &SdscFoldSet::new(op.iter.cores_used()),
                sib,
                layout,
            )?);
            Ok(())
        };
        // The rotate MATMUL stays per-head: its kernel `P` is shared, so batching it would need a
        // `y` dim, and the splitter stops dividing `y` once `mb` fills the cores at mq>1 — which
        // silently swaps the mb/y strides (see the `batched` gate in
        // `ir::bridge::tiled_op_sdsc_op::attn`, where the same trap is documented in full).
        // ⭐⭐⭐⭐⭐ ONE MATMUL FOR EVERY HEAD AND EVERY ROW — no per-head loop and NO BATCH AXIS.
        //
        // `heads` ops become ONE (32 -> 1 at granite). `P` is the SAME rotation for every head and every
        // row, so this never needed a `y` axis at all: sweep `m = heads*mq` rows against the shared 2-D
        // kernel `[hd, hd]` and every head is covered.
        //
        // ⛔ THE `y`-BATCHED FORM IS WHAT DOES NOT WORK, AND THE REASON IS ARRANGEMENT, NOT STRIDES.
        // Measured on granite-3.1-2b fp8 (mq=7):
        //
        //   tensor 't729': device arrangement StickLayout { rows: 32, cols: 448, Flat }
        //   conflicts with the earlier StickLayout { rows: 7, cols: 2048 }
        //
        // A `y`-batched rank-3 view makes `y` the LEADING (rows) axis, and a tensor gets ONE device
        // arrangement per bundle. The rank-2 sweep has no such problem: `[heads*mq, hd]` is EXACTLY the
        // arrangement the three pointwise legs below already read, and the comment there proves the
        // bytes line up — row `j = h*mq + r` lands at `j*hd = h*mq*hd + r*hd`, which is
        // `rope_prefill_block_offset(r, h, mq, hd)` for every `(h, r)`.
        //
        // So the per-head loop was never buying correctness; it was one op per head doing what one op
        // over all rows does, because the row axis already enumerates (head, row) contiguously.
        ops.push(assemble_matmul_off(
            &format!("rope_rot_o{t}"),
            scratchy_subtile::sdsc_abstract::MatM::of_token_rows(heads * mq),
            scratchy_subtile::sdsc_abstract::MatN::of_head_dim(hd),
            scratchy_subtile::sdsc_abstract::MatK::of_head_dim(hd),
            scratchy_subtile::sdsc_abstract::MatY::unbatched(),
            crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                bmm_site::TapeLoweringSite::witness(),
            ),
            &rb(&x, heads * mq, hd),
            scratchy_subtile::addr::DevOff::ZERO,
            &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
            scratchy_subtile::addr::DevOff::ZERO,
            &rb(&rot, heads * mq, hd),
            scratchy_subtile::addr::DevOff::ZERO,
            sym_id_base,
            layout,
        ));
        // ── The three POINTWISE legs collapse to ONE op EACH, across all heads. ──────────────────
        // These were `heads` ops apiece (160 of the 402 trips in a granite prefill layer body — the
        // single largest family, 40%) purely because they inherited the per-head loop. They do not
        // need it: at `rows = heads*mq` the op's own row stride is `hd`, so row `j = h*mq + r` lands
        // at `j*hd = h*mq*hd + r*hd` — EXACTLY `rope_prefill_block_offset(r,h,mq,hd)`, for every
        // (h,r). So x/rot/xc/rs/out tile the whole tensor contiguously with no gaps and no reorder.
        //
        // cos/sin line up too, WITHOUT restaging: the worker writes element (r,i) of its [mq,total]
        // table at `(i/64)*(mq*64) + r*64 + (i%64)`, and for i = h*hd+d (hd==stick) that is
        // `h*mq*hd + r*hd + d` — the same row-major `[heads*mq, hd]` image this op reads. The table
        // is head-tiled (every head holds the same row), so head h's copy already sits at row h*mq+r.
        // Byte-identical bytes, read whole instead of `heads` times at offset 0.
        //
        // The rank-3 flat form is unchanged: `cols == hd == stick`, so `stickmajor` still cannot fire
        // (it needs cols>64) — only the row count differs, exactly as when this loop went per-row →
        // per-head. Decode (mq==1) never reaches this branch.
        let all = heads * mq;
        ew_rows(
            &mut ops,
            format!("rope_xc_o{t}"),
            "multiply",
            all,
            &x,
            scratchy_subtile::addr::DevOff::ZERO,
            &cos,
            scratchy_subtile::addr::DevOff::ZERO,
            &xc,
            scratchy_subtile::addr::DevOff::ZERO,
            sym_id_base,
        )?;
        ew_rows(
            &mut ops,
            format!("rope_rs_o{t}"),
            "multiply",
            all,
            &rot,
            scratchy_subtile::addr::DevOff::ZERO,
            &sin,
            scratchy_subtile::addr::DevOff::ZERO,
            &rs,
            scratchy_subtile::addr::DevOff::ZERO,
            sym_id_base,
        )?;
        ew_rows(
            &mut ops,
            format!("rope_add_o{t}"),
            "add",
            all,
            &xc,
            scratchy_subtile::addr::DevOff::ZERO,
            &rs,
            scratchy_subtile::addr::DevOff::ZERO,
            &out,
            scratchy_subtile::addr::DevOff::ZERO,
            sym_id_base,
        )?;
        return Ok(ops);
    }
    for r in 0..mq {
        // Row r's cos/sin slice = dev_off row r head 0 (per-position, head-tiled ⇒ serves every head).
        let coff = scratchy_subtile::addr::rc_of(mq, total, r, 0, Df::Fp16);
        for h in 0..heads {
            // DEVICE offset of row r, head h's [1,hd] block in the stick-scattered [mq,total] tensor
            // (== dev_off([mq,total],1,[r,h·hd]) for hd==STK). mq=1 ⇒ h·hd (decode, byte-identical).
            // `[row, head, feat]` corner. `rope_prefill_block_offset`'s `h*mq*hd + r*hd` is that
            // nest's hd == stick special case; the nest is right at every hd.
            let off = scratchy_subtile::addr::Nest::new(
                &["row", "head", "feat"],
                &[mq, heads, hd],
                Df::Fp16,
            )
            .view()
            .at(scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Row>::n(r))
            .at(scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Head>::n(h))
            .dev();
            // rot[r,h] = x[r,h]·P  (per-head [1,hd] @ [hd,hd] kernel; same P kernel, mb=1)
            ops.push(assemble_matmul_off(
                &format!("rope_rot_r{r}_h{h}_o{t}"),
                scratchy_subtile::sdsc_abstract::MatM::single_row(),
                scratchy_subtile::sdsc_abstract::MatN::of_head_dim(hd),
                scratchy_subtile::sdsc_abstract::MatK::of_head_dim(hd),
                scratchy_subtile::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(&x, 1, hd),
                off,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
                scratchy_subtile::addr::DevOff::ZERO,
                &rb(&rot, 1, hd),
                off,
                sym_id_base,
                layout,
            ));
            ew_ph(
                &mut ops,
                format!("rope_xc_r{r}_h{h}_o{t}"),
                "multiply",
                &x,
                off,
                &cos,
                coff,
                &xc,
                off,
                sym_id_base,
            )?; // x·cos
            ew_ph(
                &mut ops,
                format!("rope_rs_r{r}_h{h}_o{t}"),
                "multiply",
                &rot,
                off,
                &sin,
                coff,
                &rs,
                off,
                sym_id_base,
            )?; // rot·sin
            ew_ph(
                &mut ops,
                format!("rope_add_r{r}_h{h}_o{t}"),
                "add",
                &xc,
                off,
                &rs,
                off,
                &out,
                off,
                sym_id_base,
            )?; // xc+rs
        }
    }
    Ok(ops)
}

/// Lower a [`SubOp::AttnDecode`] to in-bundle SuperDSC ops — BATCHED over q-heads
/// (BatchMatmul, batch=`num_q_heads`), reading the resident transposed-K + replicated
/// K/V cache (seg2, worker-filled) over the full `cap` masked by `t{ATTN_MASK_TID}`.
/// Decode (mq=1): per head `scores[1,cap] = Q·Kᵀ·scale + mask`, softmax, `out=probs·V`.
/// Viewed as `[nqh, cap]` for the ew/softmax ops (mb=nqh, out=cap; `[nqh,1,cap]`≡
/// `[nqh,cap]` bytes). Softmax reduce-outputs are one stick (like rmsnorm's mean).
/// ⭐⭐ `HD` IS A CONST GENERIC — same reason as [`lower_rope_node`]. The head dim parameterises the device
/// layout (slabs = head_dim/lanes; the head-major collapse is byte-identical ONLY at head_dim == lanes),
/// so every decision it drives must be a branch on a const, not on a value. The value becomes a const at
/// ONE dispatch in `lower_one_node`.
fn lower_attn_node<F: RopeForm, const NQH: u32, const NKVH: u32, const HD: u32>(
    node: &SubtileNode<F>,
    // The geometry witness the door minted, carrying the GQA divisibility proof. Every head count
    // and head dim this function uses is read off it, so none of them is a value the node handed
    // over and there is nothing here to check against the consts.
    geom: scratchy_subtile::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
    cap: u32,
    active_cap: ActiveCap,
    // TRUE when this bundle's query rows are separate requests (a batched decode) rather than
    // consecutive positions of one sequence (a prefill chunk). Only the KV writes care: a prompt's
    // rows share a page and differ in slot, requests differ in both.
    rows_are_requests: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    // ⭐ THE GEOMETRY IS THE COMPILER'S HERE, NOT THE NODE'S. `lower_one_node`'s door instantiated
    // this function FROM the node's own `ModelAttnGeometry`, so there is no second copy of the head
    // counts in scope to disagree with the consts. What the node is still asked for is its cache
    // identity and its scale, which the geometry does not carry.
    let (k_id, v_id, scale_val) = match &node.op {
        SubOp::AttnDecode {
            layout: kv, scale, ..
        } => (
            kv.cache_tensor().index() as u32,
            kv.v_cache_tensor().index() as u32,
            *scale,
        ),
        _ => {
            return Err(SuperDscError(
                "lower_attn_node: node is not AttnDecode".into(),
            ));
        }
    };
    let (nqh, nkvh, hd) = (geom.nqh(), geom.nkvh(), geom.hd());
    if node.inputs.len() < 5 {
        return Err(SuperDscError(format!(
            "AttnDecode t{}: expects 5 inputs [q, prefix_k, prefix_v, new_k, new_v], found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let stick = Fp16::ELEMS_PER_STICK; // 64
    let pool = scratchy_subtile::sdsc_abstract::PagedKvPool::new(nkvh as usize, hd as usize);
    // ── PAGED-ATTENTION COMPUTE EXTENT ── see the header doc above for `active_cap` semantics.
    let active_cap: u32 = active_cap.resolve(cap, stick);
    // SPAN-OVERFLOW GUARD (ported from torch-spyre's span_overflow_hint_analysis.py). Real
    // corrective re-tiling of the resident cache's PHYSICAL STORAGE (not just the compute sweep
    // `active_cap` already bounds) means paging the cache into multiple physical buffers — a
    // structural change to the resident-KV allocation, not something this one call site can retrofit
    // in place. So this guard does what torch-spyre's planner does BEFORE re-tiling: run the actual
    // search (`cheapest_split_clearing_span`) and report the split it finds, rather than a bare
    // "not implemented". Unreachable for every real model config checked so far (granite hd=64,
    // cap up to several thousand: span ~0.5 MB against a 256 MB budget — see
    // `ir::bridge::span_overflow`'s own tests) — kept as a real `cargo build` guard, not a runtime
    // assert, so a future config that DOES trip it fails loudly with the fix already computed.
    {
        use crate::ir::bridge::span_overflow::{
            MAX_SPAN_BYTES, cheapest_split_clearing_span, physical_span_bytes,
        };
        let span = physical_span_bytes(cap, hd, stick, Fp16::WORD_LENGTH);
        if span > MAX_SPAN_BYTES {
            let cap_dim = ItDim {
                name: "cap",
                size: cap,
                is_reduction: false,
                is_stick: true,
                df: Df::Fp16,
            };
            let split_report = match cheapest_split_clearing_span(&cap_dim, |split| {
                physical_span_bytes(cap / split.max(1), hd, stick, Fp16::WORD_LENGTH)
            }) {
                Ok(split) => format!(
                    "the cheapest legal split of `cap` that clears the limit is {split}-way \
                     (cap/{split}={} slots/core) — but applying it requires paging the resident \
                     KV allocation, not implemented at this call site",
                    cap / split.max(1)
                ),
                Err(e) => format!("no legal split of `cap` clears the limit either: {e}"),
            };
            return Err(SuperDscError(format!(
                "AttnDecode t{}: resident K/V cache [cap={cap}, hd={hd}] physical span {span} B \
                 exceeds the {MAX_SPAN_BYTES} B hardware addressing limit. {split_report}.",
                node.output.tensor.index() as u32,
            )));
        }
    }
    // GENERAL-mq: mq = the chunk's query-row count (1 = decode; >1 = prefill / chunked-prefill). ONE
    // algorithm below for any mq — see ir::bridge::tiled_op_sdsc_op::attn's module doc (the port of
    // torch-spyre's spyre__sdpa_overrideable).
    let attn_tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    let mq = attn_tile_op
        .dims
        .iter()
        .find(|d| d.name == "mb")
        .map(|d| d.size)
        .unwrap_or(1);
    // ⛔ ONE SOURCE FOR THE PADDED ROW COUNT AND THE ROW LAWS: `attn_bundle_rows` is the parse
    // boundary from the bundle's runtime width to the pad law WITH the geometry's consts in scope,
    // and the SAME carrier rides into `assemble_attn`, so the two cannot disagree. A decode width
    // (one row, or rows that are requests) must be a baked ladder rung there — the pad is
    // `Rung<MQ>`'s compile-time arithmetic and every row extent is `RungRowLaws`' const
    // (`MaskRows = NQH*MQ` by the compiler), and an unlisted width is this loud bake error, the
    // same boundary discipline as the geometry door in `lower_one_node`. A prefill chunk's width is
    // a runtime quantity and takes the runtime arm of the same law.
    // `PaddedRows` below, because everything THIS function does with the value is a ROW question — the
    // staging tensors' `[mq_pad, nkvh·hd]` row extent and the row sweeps over them; the slot/score-width
    // roles are `assemble_attn`'s, drawn there from the same carrier.
    let bundle_rows = scratchy_subtile::sdsc_abstract::attn_bundle_rows(
        geom,
        mq,
        rows_are_requests,
    )
    .ok_or_else(|| {
        SuperDscError(format!(
            "AttnDecode t{}: a decode bundle at {mq} query rows — not a width the decode ladder \
                 bakes (1, or PagedKvPool::BATCH_RUNGS). The decode width is a const of the bundle \
                 (`Rung<MQ>`); bake a listed rung, or grow the ladder and its dispatch together.",
            node.output.tensor.index() as u32
        ))
    })?;
    let width = bundle_rows.width();
    let mq_pad = width.pad().rows();
    let t = node.output.tensor.index() as u32;
    let q = ds_name(&node.inputs[0]); // roped Q [mq, nqh·hd]
    let new_k = ds_name(&node.inputs[3]); // roped new-K [mq_pad, nkvh·hd] (worker zero-pads rows)
    let new_v = ds_name(&node.inputs[4]); // new-V       [mq_pad, nkvh·hd]
    let kc = crate::wiring::act_name(k_id); // natural K cache [nqh, cap, hd] (seg2, GQA-replicated, slab-major)
    let vc = crate::wiring::act_name(v_id); // natural V cache [nqh, cap, hd] (seg2, GQA-replicated, slab-major)
    let kct = crate::wiring::act_name(kct_resident_tid(k_id)); // resident Kᵀ scratch [nqh, hd, cap]
    // attention_multiplier (config) — see the ORIGINAL header doc: NO 1/sqrt(hd) recompute.
    let scale_idx = layout
        .and_then(|l| l.scalarmul_scales.iter().position(|s| s.to_bits() == scale_val.to_bits()))
        .ok_or_else(|| {
            SuperDscError(format!(
                "AttnDecode t{}: scale {scale_val} absent from BundleLayout.scalarmul_scales (registry desync)",
                node.output.tensor.index() as u32
            ))
        })?;
    let _scale = crate::wiring::act_name(scalarmul_scale_tid(scale_idx)); // unused directly: torch-spyre splits into sqrt_scale on both Q and K.
    let sqrt_scale_val = scale_val.sqrt();
    let sqrt_scale_idx = layout
        .and_then(|l| l.scalarmul_scales.iter().position(|s| s.to_bits() == sqrt_scale_val.to_bits()))
        .ok_or_else(|| {
            SuperDscError(format!(
                "AttnDecode t{}: √scale {sqrt_scale_val} absent from scalarmul_scales (registry desync)",
                node.output.tensor.index() as u32
            ))
        })?;
    let sqrt_scale = crate::wiring::act_name(scalarmul_scale_tid(sqrt_scale_idx)); // [1,1] = √attention_multiplier
    let pmask = crate::wiring::act_name(ATTN_MASK_TID); // [nqh, cap]    prefix validity (worker-tiled, mb-broadcast over mq)
    let cmask = crate::wiring::act_name(ATTN_CAUSAL_TID); // [mq, mq_pad] causal triu (worker-tiled)
    // [mq, nqh·hd] — the identity; every scratch name below is a rendering of it.
    let attn_id = bundle::PlaceId::Act(node.output.tensor.index() as u32);
    let n = |r: bundle::SynthRole| syn(layout, attn_id.synth(r));
    let ident = crate::wiring::act_name(IDENTITY_TID);

    let mut ops: Vec<EmittedOp> = Vec::new();

    // ── (0) ZERO new_k/new_v's PADDING rows [mq..mq_pad) before anything reads them (BUG #3, see
    // ATTN_ZERO_TID's placement comment above). `new_k`/`new_v` are a lifetime-reused seg3
    // intermediate — never explicitly re-zeroed between steps/layers — so rows beyond the real `mq`
    // can alias whatever unrelated tensor last lived at that byte range. The causal mask only
    // reliably neutralizes BOUNDED garbage; this makes the padding actually zero instead of relying
    // on that. Per-kv-head (nkvh, not nqh — new_k/new_v are still nkvh-wide here, before GQA-replicate
    // expands them), one zero-copy each, matching ATTN_ZERO's own [mqp,hd] per-head-width shape.
    if mq_pad.row_axis_extent() > mq {
        let zero = crate::wiring::act_name(ATTN_ZERO_TID);
        let pad_rows = mq_pad.row_axis_extent() - mq;
        // PER (kv-head, SLAB), at the address the CONSUMER reads its padding from. RoPE packs each
        // (kv-head, slab) as `mq` rows of one stick, so head `kvh` slab `sl`'s pad rows begin one row
        // block past its real rows -- `rc_of(.., mq, ..)` names exactly that. The previous single
        // `[pad_rows, hd]` copy per head wrote at `mq*nkvh*hd + kvh*hd`, which is past the packed data
        // entirely: at BOTH head_dims the zeros landed where nothing reads them, while the Kt
        // restickify went on reading the NEXT head's real K/V as this head's padding. One stick wide,
        // so the op has no plane term of its own to disagree about.
        // THE ONE OFFSET LEFT THAT A NEST CANNOT EXPRESS, and the reason is a real layout defect, not
        // a missing abstraction. `new_k`/`new_v` are ALLOCATED `[mq_pad, nkvh*hd]` but RoPE PACKS them
        // by the REAL row count `mq`, so a per-head block holds exactly `mq` rows and has no room of
        // its own for padding: "row mq of head kvh" is not an address that exists. Framing the zero-fill
        // on the packed nest trips the footprint check at decode (mq=1) immediately, and framing it on
        // the padded nest moves every other consumer. So the zeros go where they have always gone --
        // past the packed data -- which is NOT where the Kt restickify reads this head's padding from.
        // It reads the NEXT head's real K/V there and relies on the causal mask to neutralise it.
        //
        // Reconciling that means making RoPE write `mq_pad`-framed, which is a worker change as well as
        // an emitter one. Until then this stays byte-identical rather than becoming a second convention.
        for kvh in 0..nkvh {
            ops.push(assemble_pointwise_broadcast_off(
                &format!("attn_kzero{kvh}_o{t}"),
                "identity",
                scratchy_subtile::sdsc_abstract::RowCount::of_zero_pad_rows(pad_rows),
                scratchy_subtile::sdsc_abstract::BlockCols::of_head_dim(hd),
                &[In::full(&rb(&zero, pad_rows, hd)).ew()],
                &rb(&new_k, mq_pad.row_axis_extent(), nkvh * hd),
                // FLAT, and that is the finding: this lands past the packed data, in a region no
                // consumer reads (see the note above), so it is placed row-major rather than through
                // the stick law. Saying `flat` makes which arrangement was meant explicit instead of
                // leaving `mq*nkvh*hd + kvh*hd` for a reader to infer.
                scratchy_subtile::addr::Nest::flat(
                    &["row", "feat"],
                    &[mq_pad.row_axis_extent(), nkvh * hd],
                    Df::Fp16,
                )
                .view()
                .at(scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Row>::n(mq))
                .at(scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Feat>::n(kvh * hd))
                .dev(),
                sym_id_base,
                layout,
            ));
            ops.push(assemble_pointwise_broadcast_off(
                &format!("attn_vzero{kvh}_o{t}"),
                "identity",
                scratchy_subtile::sdsc_abstract::RowCount::of_zero_pad_rows(pad_rows),
                scratchy_subtile::sdsc_abstract::BlockCols::of_head_dim(hd),
                &[In::full(&rb(&zero, pad_rows, hd)).ew()],
                &rb(&new_v, mq_pad.row_axis_extent(), nkvh * hd),
                // FLAT, and that is the finding: this lands past the packed data, in a region no
                // consumer reads (see the note above), so it is placed row-major rather than through
                // the stick law. Saying `flat` makes which arrangement was meant explicit instead of
                // leaving `mq*nkvh*hd + kvh*hd` for a reader to infer.
                scratchy_subtile::addr::Nest::flat(
                    &["row", "feat"],
                    &[mq_pad.row_axis_extent(), nkvh * hd],
                    Df::Fp16,
                )
                .view()
                .at(scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Row>::n(mq))
                .at(scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Feat>::n(kvh * hd))
                .dev(),
                sym_id_base,
                layout,
            ));
        }
    }

    // ── (1) GQA-replicate: NEITHER K nor V's freshly-computed new-token block replicates to nqh
    // anymore. `kct`/(the value-side equivalent for the new block) only ever need ONE representative
    // copy per kv-head group — the resident-cache side (`kc`/`vc`, still genuinely nqh-sized for
    // their own OWN separately-proven cache-write convention) is untouched, but the new-token block's
    // OWN scratch buffers (new_k_rep/new_v_rep) are fresh allocations with no such constraint, so
    // dedup them too. `assemble_attn`'s new-block reads (both K, from before, and V, now) map query
    // head h to its kv-head via `gqa_kv_head`, matching producer/consumer by construction.
    // PERF (2026-07-28): cuts attn_krep + attn_vrep from nqh(32) to nkvh(8) launches each.
    // ── DECODE TRIM (always on at decode): the GQA-dedup above made these
    // copies DEAD. Each `attn_krep{kvh}` reads `new_k` as `rb(new_k, mq_pad, nkvh*hd)` at
    // `kvh*mq*hd`, multiplies by the IDENTITY kernel, and writes `new_k_rep` — same declared shape,
    // same offset, same width. It is a pure buffer rename that costs hd*hd*mq_pad = 4.19 M MACs per
    // layer across the 16 ops (~6% of the layer's MACs) to move one real row of 64 elements.
    //
    // Safe by construction: an identity copy makes `new_k_rep` byte-equal to `new_k` over exactly
    // the region it covers, so every consumer reading through the copy sees what it would read from
    // the source. (Anything the copy did NOT cover was uninitialised in `new_k_rep`, so reading the
    // source is strictly better-defined.) The pad-row zeroing at (0) already writes `new_k`/`new_v`
    // themselves, upstream of this, so it is preserved either way.
    //
    // DECODE-ONLY (mq==1): prefill keeps the copies and stays byte-identical, so this cannot collide
    // with prefill work — the same discipline as the batched-attention gate.
    // A DECODE BATCH TRIMS THEM TOO. The copies exist so a per-head reader finds its own replicated
    // slot; the GQA-group-batched attention reads the REPRESENTATIVE slot directly and never looks at
    // them, and a batched decode now runs that form. So the `nkvh*2` identity copies a layer are pure
    // work at any batch width, for the same reason they are at one row.
    //
    // Prefill keeps them — it runs the per-head reader — so its bundles stay byte-identical.
    let decode_trim =
        // ⭐ THE PARAMETER, NOT THE GLOBAL. This function already RECEIVES the kind; reading the
        // thread-local here was a second carrier for one fact, and the two could disagree.
        // ⛔ AND `mq == 1` IS NOT "SOLO DECODE": it is also a one-token prefill chunk. The kinds
        // coincide at width one, which is exactly why `bs=1` passing never proved anything about the
        // batched kind. Keep both terms, but understand the first as a WIDTH fact and the second as a
        // KIND fact — they are not two spellings of the same test.
        mq == 1 || rows_are_requests;
    let (new_k_rep, new_v_rep) = if decode_trim {
        (new_k.clone(), new_v.clone())
    } else {
        (
            syn(layout, attn_id.synth(bundle::SynthRole::NewKRep)),
            syn(layout, attn_id.synth(bundle::SynthRole::NewVRep)),
        )
    };
    if !decode_trim && let Some(l) = layout {
        for r in [bundle::SynthRole::NewKRep, bundle::SynthRole::NewVRep] {
            l.synth(attn_id.synth(r), &[mq_pad.row_axis_extent(), nkvh * hd]);
        }
    }
    // PER-KV-HEAD BASE = `kvh * mq * hd` (2026-07-28, CORRECTED from an earlier `mq_pad` version that
    // regressed decode). The stride between kv-head column-blocks in these `[*, nkvh*hd]` tensors is
    // set by their PRODUCER, RoPE, which writes head h row r at
    // `rope_prefill_block_offset(r,h,mq,hd) = h*mq*hd + r*hd` -- stride `mq`, the chunk's REAL query-row
    // count, NOT the stick-padded `mq_pad`. Consumers must use the same `mq`:
    //   * decode (mq=1): `kvh*mq*hd == kvh*hd` -- exactly the long-standing original offset, which was
    //     correct all along; a briefly-committed `kvh*mq_pad*hd` (= kvh*64*hd) broke it.
    //   * prefill (mq=31): `kvh*31*hd` -- the original bare `kvh*hd` WAS wrong here (it only ever
    //     reached the right block for kvh==0), and `kvh*mq_pad*hd` (= kvh*64*hd) was wrong too.
    // Matches `qs_off`'s `h*mq*hd` in assemble_attn_block (2aeb691a), which is the same tensor class
    // and was independently confirmed decode-neutral (byte-identical bundle at mq==1).
    // SLAB-SPLIT, for the same reason as every other head-dim-spanning op here. A single
    // `[mq_pad, hd]` copy addresses its own second stick at `mq_pad*stick` — its declared row count
    // — but RoPE, which WROTE this buffer, puts it at `mq*stick`, because it packs each (head, slab)
    // as `mq` tightly-strided rows. Those coincide only when a head is ONE stick, and at
    // head_dim 128 / mq 31 / mq_pad 64 they are 2112 elements apart: every kv-head's upper half of K
    // and V is copied from, and to, the wrong place on every prefill.
    //
    // Per (kv-head, slab) the operands are one stick wide, so the op has no plane term of its own to
    // get wrong and the address is entirely the corner -- taken from `rc_of`, the same law RoPE
    // emitted through. `k = stick` too: reading all `hd` input columns would reintroduce the very
    // stride this is removing on the A side. The identity kernel's matching diagonal block sits at
    // `s*stick*(hd+stick)` in its `[in, out]` residency.
    //
    // At nslab == 1 there is one iteration, every slab term is 0, `stick == hd` makes `n`/`k`
    // identical, and the name collapses -- granite-3.1-2b emits exactly what it did before.
    let krep_nslab = (hd / stick).max(1);
    for kvh in 0..nkvh {
        if decode_trim {
            break; // the copies are the identity — `new_k_rep`/`new_v_rep` ARE `new_k`/`new_v`.
        }
        for s in 0..krep_nslab {
            // ⛔⛔⛔ NAMED AXES, NOT HAND-SUMMED PRODUCTS — the same conversion as `attn.rs`'s kv stream,
            // and the same reason: `kvh * hd + s * stick` adds two products of DIFFERENT UNITS (kv-head x
            // head_dim, slab x lanes) into one column, and the two are IDENTICAL at hd == stick == 64. A
            // transposition there is a plausible wrong column for every kv-head above 0 at hd=128, which is
            // the only shape whose batch decode is wrong. `View::slab` owns the `s * lanes` multiplier, so
            // there is no pair left at the call site to swap, and `View::at` refuses an axis the nest lacks.
            // Pinned value-identical over 462 cases (hd 64/128/256 x nkvh 1/2/8 x mq 1/2/4/8/96) in
            // `tests/zz_kv_stream_nest_equiv.rs` BEFORE this conversion.
            let slab_off = scratchy_subtile::addr::Nest::new(
                &["row", "head", "feat"],
                &[mq, nkvh, hd],
                Df::Fp16,
            )
            .view()
            .at(scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Head>::n(kvh))
            .slab(s)
            .dev();
            // The identity's (in-slab s, out-slab s) diagonal block, as a coordinate on the [in, out]
            // kernel nest rather than the product `s*stick*(hd+stick)` it works out to.
            // ⛔ THE SAME PRODUCT ON BOTH AXES is the one spelling where a transposition cannot be seen at
            // all; pinned separately (7 cases, hd 64/128/256) before conversion.
            let ident_off =
                scratchy_subtile::addr::Nest::new(&["row", "feat"], &[hd, hd], Df::Fp16)
                    .view()
                    .at(scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Row>::n(s * stick))
                    .slab(s)
                    .dev();
            let nm = |base: &str| {
                if krep_nslab == 1 {
                    format!("attn_{base}{kvh}_o{t}")
                } else {
                    format!("attn_{base}{kvh}s{s}_o{t}")
                }
            };
            ops.push(assemble_matmul_off(
                &nm("krep"),
                scratchy_subtile::sdsc_abstract::MatM::of_padded_chunk_rows(mq_pad),
                scratchy_subtile::sdsc_abstract::MatN::one_stick(
                    scratchy_subtile::sdsc_abstract::Lanes::FP16,
                ),
                scratchy_subtile::sdsc_abstract::MatK::one_stick(
                    scratchy_subtile::sdsc_abstract::Lanes::FP16,
                ),
                scratchy_subtile::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(&new_k, mq_pad.row_axis_extent(), nkvh * hd),
                slab_off,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &ident),
                ident_off,
                &rb(&new_k_rep, mq_pad.row_axis_extent(), nkvh * hd),
                slab_off,
                sym_id_base,
                layout,
            ));
            ops.push(assemble_matmul_off(
                &nm("vrep"),
                scratchy_subtile::sdsc_abstract::MatM::of_padded_chunk_rows(mq_pad),
                scratchy_subtile::sdsc_abstract::MatN::one_stick(
                    scratchy_subtile::sdsc_abstract::Lanes::FP16,
                ),
                scratchy_subtile::sdsc_abstract::MatK::one_stick(
                    scratchy_subtile::sdsc_abstract::Lanes::FP16,
                ),
                scratchy_subtile::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(&new_v, mq_pad.row_axis_extent(), nkvh * hd),
                slab_off,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &ident),
                ident_off,
                &rb(&new_v_rep, mq_pad.row_axis_extent(), nkvh * hd),
                slab_off,
                sym_id_base,
                layout,
            ));
        }
    }

    // ── (2) qs = Q·sqrt_scale, new_k_scaled = new_k_rep·sqrt_scale (torch-spyre scales BOTH Q and K
    // by sqrt(scale), so `scores = (Q·√s)@(K·√s)ᵀ = Q@Kᵀ·s`). The resident cache stores K ALREADY
    // scaled (see the cache-write below), so the prefix score needs only qs scaled. new_k_scaled is
    // nkvh-wide now (K dedup, see above) — assemble_attn's "new block" score reads it per-query-head
    // via the SAME gqa-dedup mapping kct_base already uses (matching producer/consumer by construction).
    let (qs, new_k_scaled) = (n(bundle::SynthRole::Qs), n(bundle::SynthRole::NewKScaled));
    if let Some(l) = layout {
        l.synth(attn_id.synth(bundle::SynthRole::Qs), &[mq, nqh * hd]);
        l.synth(
            attn_id.synth(bundle::SynthRole::NewKScaled),
            &[mq_pad.row_axis_extent(), nkvh * hd],
        );
    }
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_qs_o{t}"),
        "multiply",
        // Q is `[mq, nqh*hd]`: the chunk's REAL rows, the full query feature width.
        scratchy_subtile::sdsc_abstract::RowCount::of_query_rows(
            scratchy_subtile::sdsc_abstract::QueryRowCount::of_mq(mq),
        ),
        scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(nqh * hd),
        &[In::full(&rbo(&q)).ew(), In::scalar(&rbo(&sqrt_scale)).ew()],
        &rbo(&qs),
        scratchy_subtile::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_nks_o{t}"),
        "multiply",
        // new-K is allocated `[mq_pad, nkvh*hd]`: the scale covers the PADDED rows, zeros included.
        scratchy_subtile::sdsc_abstract::RowCount::of_padded_chunk_rows(mq_pad),
        scratchy_subtile::sdsc_abstract::BlockCols::of_feature_cols(nkvh * hd),
        &[
            In::full(&rbo(&new_k_rep)).ew(),
            In::scalar(&rbo(&sqrt_scale)).ew(),
        ],
        &rbo(&new_k_scaled),
        scratchy_subtile::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));

    // ── (3) the unified score/softmax/output computation — ONE algorithm for any mq (see
    // ir::bridge::tiled_op_sdsc_op::attn's module doc for the torch-spyre correspondence). The head
    // geometry travels as the minted type, not as three integers this call could reorder, and the
    // width travels fused to its row laws on the boundary's one carrier.
    ops.extend(assemble_attn(
        t,
        geom,
        bundle_rows,
        cap,
        active_cap,
        &qs,
        &new_k_scaled,
        &new_v_rep,
        &kct,
        &vc,
        &pmask,
        &cmask,
        attn_id,
        rows_are_requests,
        sym_id_base,
        layout,
    )?);

    // ── (4) KV CACHE WRITE: append this step's new_k_scaled (nkvh-wide, dedup)/new_v_rep (nqh-wide)
    // into the resident cache. K writes only the nkvh representative slots (kvh*gqa) — `kctpost`
    // (below) only ever reads those; the other slots were always dead. V is UNCHANGED (nqh writes,
    // genuinely all read via `vc`'s own nqh-replicated convention). Writes K ALREADY sqrt-scaled, so
    // future steps' prefix scores need no further K scaling.
    //
    // PLAIN COPY, not matmul (2026-07-28): the OLD proven flash-decode (worktree superdsc-batch-perf)
    // writes the cache via a genuine pointwise IDENTITY copy (`OpFunc::Identity`, head_major=true,
    // rows=1 -- see its `copy()` helper), never a matmul. This file used a matmul-by-identity-kernel
    // workaround at M=mq_pad (writing all 64 rows -- 1 real + 63 zero-padding -- every single step),
    // a mechanism the old code never exercises at all (it only ever writes the real row(s), never
    // padding). That M=mq_pad matmul-into-a-cap-strided-kernel pattern has no precedent in the proven
    // code, and real hardware testing this session never showed it behaving correctly. Restored to a
    // plain per-row identity copy (Stk<FlatTag>, unconditionally row-major -- no stick-degeneracy
    // ambiguity, matching the old code's own head_major=true choice), looping over only the REAL `mq`
    // rows (never the zero-padding) -- matching decode's mq=1 (one copy) and generalizing to prefill's
    // mq>1 (mq copies) the same way every other per-row op in this file already does.
    // ROW-BATCHED (2026-07-28, TTFT): ONE [mq, hd] copy per kv-head, not `mq` single-row copies.
    // SOURCE offset `kvh*mq*hd` (HEAD-MAJOR, not the naive flat `p*nkvh*hd + kvh*hd`):
    // `new_k_scaled` inherits RoPE's producer convention (rope_prefill_block_offset = h*mq*hd + r*hd),
    // carried unchanged through attn_krep (a stick=hd matmul copy) and attn_nks (an offset-0
    // whole-buffer scale), so kv-head kvh's rows are CONTIGUOUS at stride hd from kvh*mq*hd.
    // DEST `kc` is [nqh, cap, hd], so head qh's slot p is at qh*cap*hd + p*hd -- also stride hd.
    // Both sides therefore step by exactly `hd` per row, which is what a [mq, hd] op does natively:
    // cols == hd == stick so `stickmajor` cannot fire, the op stays rank-3 flat, and its internal
    // address is base + r*hd + d. `slot_stride_bytes` is unchanged (the runtime shifts the whole
    // block by seq_pos*hd*2 exactly as it shifted each single-row copy).
    // Saves nkvh*(mq-1)*2 ops/layer = 480 at granite prefill (496 -> 16), ~19k ops/chunk.
    // At mq==1 this emits ONE op per kv-head with rows=1 -- the same op count and identical
    // addressing as the per-row loop it replaces.
    // ⭐ THE SLAB COUNT IS NOT A LOCAL ANY MORE. It reaches this write only as `Shape::slabs_of`, inside
    // `Shape::slabs_of`, once, and the two placements take it from there — so there is no second
    // spelling of `hd/64` in scope to disagree with the one the walk's pitch uses.
    // REQUEST-MAJOR, so that one request's kv-head copies are CONSECUTIVE and fuse into a
    // single launch. A group resolves one request's page table and write cursor, so the run
    // breaks where the request changes (`GroupKind::Slot`'s `req`) — kv-head-major order would
    // alternate requests every op and leave every copy its own launch, which is what made a
    // batch of 8 slower than 8 separate forwards. One request ⇒ this loop runs once and the
    // order is exactly what it was.
    let per_request = rows_are_requests && mq > 1;
    // ONE REQUEST LOOP OVER BOTH K AND V. Fusion needs the ops to be CONSECUTIVE and to name one
    // request: a group is a launch, and a launch resolves one page table and one write cursor.
    // Request-major over each tensor separately still costs two launches per request, because
    // K's last request and V's first are different requests; spanning both puts a request's
    // whole `nkvh * 2` copies together, so it is ONE launch. kv-head-major order would alternate
    // requests every op and fuse nothing, which is what made a batch of 8 slower than 8 separate
    // forwards. One request ⇒ this loop runs once and the emission order is exactly what it was.
    // ⭐⭐⭐⭐⭐ ONE OP PER (REQUEST, SLAB, TENSOR) — the KV-HEAD loop is GONE, replaced by a `y` axis
    // over the kv heads of one slab. `nkvh * nslab * rows` copies per family become `nslab * rows`: an
    // 8x cut at granite (nkvh=8), and it does not grow with the head count.
    //
    // ⛔⛔⛔ AND THE SLAB CANNOT JOIN THAT `y` AXIS — MEASURED ON CARD, NOT INFERRED. The pair
    // `(h, s)` linearizes as `h*nslab + s`, and that index steps exactly ONE STICK PLANE on BOTH
    // operands (source `mq*hd = nslab*(mq*stick)`, dest `PLANE_SLOTS*hd = nslab*(PLANE_SLOTS*stick)`),
    // so ONE `y` of `nkvh*nslab` should cover both loops. It passes every check this backend has —
    // the builder's y-stride check at hd 64/128/256/512, a conservation law differenced out of both
    // nests, two Kani proofs of coverage + injectivity — and granite-2b (hd=64, where nslab==1 makes
    // the two forms identical) stays coherent 10/10. On granite-8b at hd=128 it produces GARBAGE from
    // the first token ("The is a 1", 10/10 runs) at an ITL of 78-82 ms against the same emission's
    // 80-85 — the recorded tell: a FASTER ITL with degenerate output. The head-only form below, from
    // the same laws and the same walk, is coherent 10/10 on the same bake. So whatever makes a `y`
    // step cross a feature slab is NOT the address arithmetic; do not retry the pair walk on the
    // strength of the arithmetic, which is correct and insufficient.
    //
    // WHY A MATMUL AND NOT THE POINTWISE COPY IT REPLACES: the two operands need DIFFERENT row pitches
    // on one axis (`mq*nslab` rows per source head, `PLANE_SLOTS*nslab` per cache block), and
    // `assemble_pointwise_broadcast_off` takes rows/cols plus one offset per operand — rank-2, no
    // per-operand extents, no `y`. A matmul carries `BatchStrides::{a_pitch, o_pitch}` and a `y` axis:
    // exactly the machinery that killed the score leg's per-head loop (`ab1c8af7`/`c65e8262`), used the
    // same way, with the SAME three requirements that one needed —
    //   (1) a ONE-STICK contraction (dxp is measured twice incoherent on a multi-stick contraction
    //       under a `y` batch), which an identity kernel gives for free: `I[64,64]` is the same matrix
    //       for every slab, and the `[hd,hd]` identity's top-left block IS it, byte-for-byte, in kernel
    //       residency (`(out/64)*(hd*64) + in*64 + out%64` is `in*64+out` for `in,out < 64`);
    //   (2) the HEAD-OUTERMOST order (`of_kv_cache_write`) — `BatchInnerMq1Proven` is `[mb, y, in]`, so
    //       `mb` sits outside `y` and the walk strides `y` by 64 whatever pitch an operand declares;
    //   (3) each operand at its OWN pitch, the output declaring its own device extent.
    //
    // ⛔ THE ROW FACTOR STAYS, and it is not an oversight: which PAGE a row's token lands in is the
    // host's block table, resolved per LAUNCH from `kv_request`, so rows are not a uniform stride to
    // walk. Collapsing them is what the pool's row STRIPE would make possible, and that is a separate
    // change ([[eliminate-host-routing-slab-plan]]).
    //
    // ⛔ IT IS NOT AN ITL WIN AT hd=128, AND THAT IS THE MEASUREMENT, NOT AN EXPECTATION. granite-8b
    // decode: 12.2-12.9 tok/s and ITL 80.0-84.8 ms over 10 runs, against 12.1-12.2 / 82.4 for the
    // per-head loops — i.e. removing SEVEN EIGHTHS of this family's ops moved nothing measurable. The
    // "cachewr is 54% of the decode bundle's ops" reading counted `op_manifest.json` ENTRIES, and a
    // request's K and V writes share ONE launch group (`group_N/bundle.mlir` executes both), so the
    // ops inside a group were never the launch cost. Keep the form for its op count at gemma-4's
    // hd=512 and for the loop it removes; do not attribute ITL to it.
    //
    // ⛔ AND THE ~2 ms A LATER PAIRED A/B SHOWED IS THE REGION DRAW, NOT THIS OP'S MACs. The ITL MODES
    // are identical across every tree measured (~82.4 slow / ~78.8 fast); only the fast-region hit rate
    // differs (this tree 1/10, baseline 5/12, the RoPE collapse 4/10). `c1b7e3b7`'s message blames the
    // y-batched identity matmul's 8·64·64 MACs — that explanation is UNSUPPORTED by the mode data. See
    // [[every-itl-delta-today-was-the-region-lottery]] and the RoPE note in `lower_rope_node`.
    let src_nest =
        scratchy_subtile::addr::Nest::new(&["row", "head", "feat"], &[mq, nkvh, hd], Df::Fp16);
    // ⛔ `PLANE_SLOTS`, NOT `PAGE_SLOTS`, AND LOAD-BEARING AT EVERY HEAD DIM — this extent IS the
    // write's kv-head stride (the `y` step is derived from it), and every READER takes its head stride
    // from `PagedKvPool::plane_block_elems()`, which is physical. Declaring the ADDRESSABLE page while
    // the pool strides by the PHYSICAL plane splits one law into two: writes land at `kvh*PAGE_SLOTS*hd`
    // and reads at `kvh*PLANE_SLOTS*hd`, so every kv head but 0 reads keys nobody wrote. MEASURED: that
    // regressed granite-2b (hd=64) from coherent to garbled, which is how it was found. The physical
    // plane also holds the padded-write slack, so the addressable count would place stick 1 where stick
    // 0's own slack still lives.
    let dst_nest = scratchy_subtile::addr::Nest::new(
        &["slot", "feat"],
        &[
            scratchy_subtile::sdsc_abstract::PagedKvPool::PLANE_SLOTS as u32,
            hd,
        ],
        Df::Fp16,
    );
    // ASKED OF THE SHAPE, once: the slab count reaches this write only through `slabs_of`, so there is
    // no `hd/64` in scope for the loop and the pitches to spell differently.
    let slabs = scratchy_subtile::addr::Shape::<0, 0, 0, 0>::slabs_of(hd, Df::Fp16);
    for req in 0..if per_request { mq } else { 1 } {
        // ⛔ NO REQUEST TERM IN THE ADDRESS. `req` here indexes a ROW OF THE ACTIVATION and nothing more:
        // which page that row's token lands in is the host's page map, applied as a per-launch shift. A page
        // holds slots, not requests, so the address inside it is `(kv head, slot, feature)`.
        //
        // ONE `y`-BATCHED COPY PER TENSOR when the rows are separate requests, otherwise one covering every
        // row. A prompt's rows go to consecutive slots of ONE page, which `m = mq` plus the runtime's one
        // slot-shift expresses exactly. A batch's rows sit at DIFFERENT SLOTS in DIFFERENT pages, and no
        // single shift expresses that — so each row is its own op, tagged with the request whose page and
        // write cursor the runtime must resolve.
        let rq = if per_request {
            format!("_r{req}")
        } else {
            String::new()
        };
        // ONE OP PER SLAB, and the slab is a COORDINATE of both placements rather than a term either one
        // adds: `View::slab` owns the `s * lanes` multiplier. At nslab == 1 this loop runs once and the
        // name collapses, so granite-2b emits what it did.
        // ⛔⛔⛔ THE HEADS GO ON `y` ONLY WHEN THIS OP COVERS ONE ROW, and `rows_per_op` is that test.
        //
        // The plane-walk placement declares `pitch = rows * nslab`, which describes head-outermost planes of
        // `rows` rows. The source is row-outermost — head `h` of row `r` is at `r*(nkvh*hd) + h*hd`, heads
        // INTERLEAVED inside a row — so the two readings agree only at one row, where the row axis is
        // degenerate. Decode qualifies both ways: solo decode has `mq == 1`, and a batched decode step emits
        // one op per request (`per_request`), each covering its single row. A PREFILL CHUNK does not: one op
        // spans `mq` rows, and `y` then strides heads as if each owned `mq` consecutive rows.
        //
        // MEASURED, granite-2b (hd=64, nslab=1) with the collapse ungated: every prompt from 400 characters
        // up decodes fluent garbage (`'(\x03d. .exaggeret'` for a prompt whose answer is `Paris`), while
        // `hits`/`short`/`longhits` pass — those reach their answer through cached KV rather than a long
        // prefill. Bisected to the commit that introduced this collapse; the decode ITL win it was made for
        // is untouched, because decode is exactly the side that keeps it.
        //
        // ⭐ SAME SHAPE, SAME CALL as RoPE's slab-form rotate (`rot_slab_form = mq == 1` above), which was
        // gated for prefill on its own card evidence. Two collapses, one precondition: an op whose row axis
        // is degenerate can carry the head on `y`; one that sweeps rows cannot.
        let rows_per_op = if per_request { 1 } else { mq };
        let heads_on_y = rows_per_op == 1;
        for s in 0..slabs.get() {
            for h in 0..if heads_on_y { 1 } else { nkvh } {
                let a_place = if heads_on_y {
                    scratchy_subtile::sdsc_abstract::OperandPlacement::of_plane_walk_by_head(
                        &src_nest,
                        scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Row>::n(req),
                        s,
                        slabs,
                    )
                } else {
                    scratchy_subtile::sdsc_abstract::OperandPlacement::at_head(
                        &src_nest,
                        scratchy_subtile::addr::Idx::<scratchy_subtile::addr::Row>::n(req),
                        h,
                        s,
                        slabs,
                    )
                };
                let o_place = if heads_on_y {
                    scratchy_subtile::sdsc_abstract::OperandPlacement::of_plane_walk_by_head_block_base(
                        &dst_nest, s, slabs,
                    )
                } else {
                    scratchy_subtile::sdsc_abstract::OperandPlacement::at_block(
                        &dst_nest,
                        scratchy_subtile::sdsc_abstract::PagedKvPool::block_of(kv_head_of(
                            h, nkvh,
                        )?),
                        s,
                        slabs,
                    )
                };
                // K writes only the nkvh representative slots (`kctpost` only ever reads those); V is the
                // nkvh-wide `new_v_rep`, whose reader is kv-head indexed too. Both sides of both writes are
                // therefore kv-head indexed, which is what lets ONE `y` axis serve them.
                for (base, src, dst) in [("kc", &new_k_scaled, &kc), ("vc", &new_v_rep, &vc)] {
                    ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                        &if heads_on_y {
                            format!("cachewr_{base}s{s}{rq}_o{t}")
                        } else {
                            format!("cachewr_{base}{h}s{s}{rq}_o{t}")
                        },
                        // The kv stream's REAL rows this copy covers: one per request-copy, else the chunk's.
                        scratchy_subtile::sdsc_abstract::MatM::of_query_rows(
                            scratchy_subtile::sdsc_abstract::QueryRowCount::of_mq(if per_request {
                                1
                            } else {
                                mq
                            }),
                        ),
                        scratchy_subtile::sdsc_abstract::MatN::one_stick(
                            scratchy_subtile::sdsc_abstract::Lanes::FP16,
                        ),
                        scratchy_subtile::sdsc_abstract::MatK::one_stick(
                            scratchy_subtile::sdsc_abstract::Lanes::FP16,
                        ),
                        // `y` WALKS THE KV HEADS OF THIS SLAB — the same door the score leg's GQA group uses, and
                        // for the same reason: the batch axis carries BOTH operands' head strides, so it cannot be
                        // asked for without stating where the heads are, and the builder refuses a walk that
                        // strides by neither.
                        if heads_on_y {
                            scratchy_subtile::sdsc_abstract::MatY::of_gqa_group(
                                nkvh, a_place, o_place,
                            )
                        } else {
                            // The head is in both offsets, so there is no batch axis to stride — and nothing for the
                            // builder's stride check to be wrong about.
                            scratchy_subtile::sdsc_abstract::MatY::unbatched()
                        },
                        crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::of_kv_cache_write(
                        ),
                        &rb(src, mq, nkvh * hd),
                        a_place,
                        // I[64,64] — the identity for EVERY slab, so the shared 2-D kernel a `y` batch demands is
                        // exactly what an identity copy needs. Offset 0: see the residency argument above.
                        &Stk::<KernelTag>::kernel(stick as usize, stick as usize, &ident),
                        scratchy_subtile::addr::DevOff::ZERO,
                        &Stk::<KernelTag>::kernel(
                            scratchy_subtile::sdsc_abstract::PagedKvPool::PLANE_SLOTS,
                            hd as usize,
                            dst,
                        ),
                        o_place,
                        sym_id_base,
                        layout,
                    ));
                    if let Some(o) = ops.last_mut() {
                        o.slot_stride_bytes = stick * 2;
                        // PAGED: the slot is the position WITHIN its page; the page comes from the block table.
                        o.kv_page_slots =
                            scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
                        // ⭐ TAGGED WITH ITS ROW, and it has to be. Untagged, this op took the page from the
                        // LAUNCH's position — fine while the address carried a request term, because then the row
                        // was already in the address. That term is gone: a page holds slots, so the ONLY thing that
                        // says which page this row's token lands in is the tag, which `nonfold_page_delta` turns
                        // into that row's page through the host's block table. Untagged after the address change,
                        // every row wrote into ROW 0's page — both rows started correctly (prefill had written their
                        // KV) and then degraded, because each was reading a page the other had just overwritten.
                        o.kv_request = req;
                    }
                }
            }
        }
    }

    // ── (5) RE-TRANSPOSE the resident Kᵀ cache from the natural kc the cachewr just wrote, so NEXT
    // step's prefix score reads this step's new K too (hd==stick only — guarded above).
    //
    // `kct` is reserved nkvh-sized (`kct_bytes = nkvh*hd*cap*2`, see its placement comment — "4x less
    // restickify traffic + 4x smaller resident KV"), NOT nqh-sized like `kc`/`vc`. This loop used to run
    // `for h in 0..nqh`, writing `kct` at `h*hd*cap` for h up to nqh-1 — for h>=nkvh that walks PAST
    // kct's own reservation into whatever tensor sits right after it in seg2 (the next layer's k/v cache,
    // per the placement comment: "placed IMMEDIATELY after this layer's k,v"). Every layer, every step,
    // silently corrupting adjacent memory — matches weeks of "garbage but no crash," and the eventual
    // real hardware PCIe bus fence once it walks far enough. Loop over the nkvh DISTINCT kv-heads instead
    // (matching `gqa_dedup_kv_kernel_base`'s contract, and `assemble_attn_head`'s `kct_base` read, which
    // was fixed to match this): `kc` holds nqh REPLICATED copies within a GQA group (from the earlier
    // GQA-replicate step), so any one query head in the group — the group's first, `kvh*gqa` — carries
    // that kv-head's real K data. (`gqa` already computed above, step 1.)
    // REVERTED (2026-07-28): the Stage-2 slab-restickify wiring attempted here was tested on real
    // hardware and produced GRADUAL decode degradation (garbling that worsens over many steps, not an
    // immediate collapse) -- consistent with a compounding addressing error rather than the original
    // (still-not-understood) decoherence. `restickify_kt_opspec_2d`'s output device layout is
    // `[cap/64, hd, 64]`, computed from the SAME `cap` parameter passed in; passing `stick` (64) instead
    // of the true resident `cap` changes what "one slab" means to the op's own internal addressing in a
    // way I could not fully verify matches how the runtime's `(seq_pos/64)*slab_stride_bytes` base-shift
    // is actually implemented on the C++/hardware side. Given real hardware regressed after this change,
    // reverted forward to the plain full-cap restickify (unconditionally, for any mq) rather than keep
    // an unverified, now-suspect optimization in the tree.
    // ONE RE-TRANSPOSE PER (REQUEST, KV-HEAD) — not one for the whole batch.
    //
    // This loop used to run over kv-heads only, so a batch of 8 re-transposed ONE page: the one
    // `pool.write_base(kvh, hd)` names, which after the shim's per-op shift is request 0's. Every
    // other request's freshly-written K therefore never reached `kct`, so its NEXT step scored the
    // prefix against a stale Kᵀ. That is exactly the observed symptom — two or three good tokens,
    // then decay, worse the longer the batch runs — and no shape or bounds check can see it, because
    // the read is in-bounds, just wrong.
    //
    // AND THE REQUEST IS THE ADDRESS, NOT A TAG — so the batch's re-transposes are ONE launch.
    // Tagging each one made it its own launch (a group is keyed on the tag), which at bs=8 bought 8
    // launches at the ~93 us fixed device cost where 8 trips inside one launch cost ~3.4 us each.
    // Both offsets move by `req * request_stride`: source and destination hold the same elements in
    // different orders inside the SAME page, and the pool separates requests by exactly that stride
    // in both planes, so one constant walks both.
    //
    // 🛑 UNLIKE THE CACHE WRITE, THIS NEEDS NO UNIFORM WRITE SLOT. A re-transpose reads and writes a
    // WHOLE page, so it never consults the slot (`slot_write` and `slab_write` are both false and
    // `slot_write_delta`/`slab_delta` are therefore zero). Its only precondition is the one the batch
    // already satisfies: the live pool rows are consecutive, so `kv_rows[r] == kv_rows[0] + r` and the
    // launch's own `nonfold_page_delta` supplies `kv_rows[0]` once for the whole group.
    //
    // Flat product loop (not nested) so the body below keeps the indentation — and the shape — of the
    // whole-page form it is otherwise unchanged from.
    let kct_reqs = if per_request { mq } else { 1 };
    for (req, kvh) in (0..kct_reqs).flat_map(|r| (0..nkvh).map(move |k| (r, k))) {
        let rq = if per_request {
            format!("_r{req}")
        } else {
            String::new()
        };
        // KEPT WHOLE-PAGE over this branch's slab-split variant: upstream records that the
        // slab-granular re-transpose was tried ON CARD and regressed decode. A measurement beats the
        // symmetry argument that produced the split, so the split stays out here even though the
        // cachewr it reads IS slab-split -- the two are separate ops and only the cachewr had a
        // demonstrated addressing defect.
        // WHOLE-PAGE re-transpose, structurally the proven full-`cap` form with the page (equal to
        // that cap) substituted — NOT the slab-granular variant, whose on-card attempt regressed
        // decode. Source and destination take the identical `layer + page` shift, which is why
        // natural K lives in the page rather than a segment of its own.
        // ⛔ NO REQUEST TERM. `req` indexes a ROW of this launch; which page that row's keys live in is
        // the host's page map, applied as a per-launch shift. A page holds slots, so the address inside
        // it is `(kv head, slot, feature)` and nothing else.
        ops.push(assemble_restickify_kt_2d(
            &format!("attn_kctpost{kvh}{rq}_o{t}"),
            scratchy_subtile::sdsc_abstract::KtTileSlots::of_page(),
            scratchy_subtile::sdsc_abstract::KtTileFeats::of_head_dim(hd),
            &kc,
            // The two planes are the whole content of this op: it reads natural K and writes Kᵀ, both at
            // the same `(kv head)` block of the page the launch was shifted to.
            scratchy_subtile::addr::DevOff::from_view_step(pool.addr(
                scratchy_subtile::sdsc_abstract::KvCoord::block(
                    scratchy_subtile::sdsc_abstract::KvPlane::Knat,
                    kv_head_of(kvh, nkvh)?,
                ),
            )),
            &kct,
            scratchy_subtile::addr::DevOff::from_view_step(pool.addr(
                scratchy_subtile::sdsc_abstract::KvCoord::block(
                    scratchy_subtile::sdsc_abstract::KvPlane::Kt,
                    kv_head_of(kvh, nkvh)?,
                ),
            )),
            sym_id_base,
            layout,
        ));
        // ⭐ TAGGED WITH ITS ROW, for the same reason the cache writes are: this op reads natural K and
        // writes Kᵀ within ONE row's page, and with no request term left in the address the tag is the only
        // thing that tells the runtime which page that is. Untagged, every row re-transposed into row 0's
        // page — which corrupts the very keys the fold is about to read.
        if let Some(o) = ops.last_mut() {
            o.kv_request = req;
        }
    }

    Ok(ops)
}

/// The stable name of a HOST-ROUTED (data-movement) SubOp, for the host-routed
/// reconnaissance set. These ops never become SuperDSC tiles — the host threads
/// activations across them — so this is a label, not a lowering.
fn host_glue_kind<F: RopeForm>(op: &SubOp<F>) -> &'static str {
    match op {
        SubOp::RopeRotate { .. } => "RopeRotate",
        SubOp::RopeAppend { .. } => "RopeAppend",
        SubOp::AttnDecode { .. } => "AttnDecode",
        SubOp::RmsNormReduce { .. } => "RmsNormReduce",
        SubOp::RmsNormApply { .. } => "RmsNormApply",
        // The pure-compute ops are never passed here.
        _ => "?",
    }
}

/// Lower one [`SubOp::ScalarMul`] node (granite embedding/residual/attn/logits multipliers) to a pointwise
/// `mul` by its scale constant. `out[i,j] = x[i,j] · scale`: operand-0 is `x` (full), operand-1 is the
/// `[1,1]` scale const [`scalarmul_scale_tid`] broadcast on BOTH dims — exactly the ATTN_SCALE mechanism
/// (a bound scalar × tensor, proven on-card). The split/address/fold are the standard pointwise structure
/// (Kani-proven via CoreSplit/dev_off/OpDims); the scale VALUE is the shared SEN169 leaf the worker binds.
/// NO weight-fold, NO host-route.
fn lower_scalarmul_node<F: RopeForm>(
    node: &SubtileNode<F>,
    scale: f32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "ScalarMul t{}: expected 1 input (x), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let x = ds_name(&node.inputs[0]);
    let out = ds_name(&node.output);
    let rows = node.output.region.rows.len;
    // DEVICE width (the padding invariant): a ScalarMul on the padded logits `[.,49159]` must use the SAME
    // device width its producer matmul emitted (49664), not the logical 49159 (whose sub-stick 7 the dxp
    // scheduler rejects). `for_pointwise` == the producer's `for_output` for macs≥2^20 producers. A no-op
    // for 64-aligned tensors (residual/embedding [.,4096]).
    let cols = DeviceWidth::for_pointwise(node.output.region.cols.len).get();
    // The scale's reserved const TID via the BundleLayout registry (shared with the worker). Missing ⇒
    // build Err (compute_bundle_layout collected every ScalarMul scale, so this is unreachable in practice).
    let idx = layout
        .and_then(|l| l.scalarmul_scales.iter().position(|s| s.to_bits() == scale.to_bits()))
        .ok_or_else(|| {
            SuperDscError(format!(
                "ScalarMul t{}: scale {scale} absent from BundleLayout.scalarmul_scales (registry desync)",
                node.output.tensor.index() as u32
            ))
        })?;
    let scale_name = crate::wiring::act_name(scalarmul_scale_tid(idx));
    let op_name = format!("scalarmul_o{}", node.output.tensor.index() as u32);
    let x_h = rbo(&x);
    let scale_h = rbo(&scale_name);
    let inputs = [In::full(&x_h).ew(), In::scalar(&scale_h).ew()];
    // SubtileTape -> TileIR -> tiler -> SdscOp, ONE call chain: `node_to_single_tile_op` derives the
    // TileOp straight from this node (its `out` extent already applies the SAME `DeviceWidth::
    // for_pointwise` padding computed above), then `assemble_pointwise_broadcast_off_from_tile` runs
    // the tiler + builds the SdscOp.
    let tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    Ok(assemble_pointwise_broadcast_off_from_tile(
        &op_name,
        &tile_op,
        "multiply",
        rows,
        cols,
        &inputs,
        &rbo(&out),
        0,
        sym_id_base,
        layout,
    ))
}

/// Narrow a node to its FIRST row: the OUTPUT and the leading (activation) input keep their columns but
/// contract to one row. Used by the m>1 prefill lm-head tail, whose activation is a `[1, ·]` slice
/// written at offset 0 by the ops above it, so the whole tail lowers through the SAME `lower_*_node` the
/// PROVEN m=1 decode path uses — no parallel emitter for the folded form.
///
/// ONLY `inputs[0]` is contracted. A matmul's `inputs[1]` is the WEIGHT `[k, n]`, whose `rows` is the
/// REDUCTION extent K, not the query count — narrowing it to one row claims K=1 and fails
/// `lower_matmul_node`'s `W rows != A cols (K)` guard. `inputs[2]` (the fp8 per-channel `w_scale`) is
/// likewise not row-indexed by M. Both are M-independent and pass through untouched.
fn node_at_one_row<F: RopeForm>(node: &SubtileNode<F>) -> SubtileNode<F> {
    let one_row = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
        scratchy_subtile::subtile_ir::TensorRegion {
            tensor: tr.tensor,
            region: scratchy_subtile::subtile_ir::Region {
                rows: scratchy_subtile::subtile_ir::Range::new(tr.region.rows.start, 1),
                cols: tr.region.cols,
            },
        }
    };
    let mut inputs = node.inputs.clone();
    if let Some(a) = inputs.first_mut() {
        *a = one_row(a);
    }
    SubtileNode {
        id: node.id,
        op: node.op,
        inputs,
        output: one_row(&node.output),
    }
}

/// Lower the m>1 PREFILL lm-head matmul as the m=1 tail it really is: extract the LAST prompt row of the
/// final-norm output into [`LAST_HIDDEN_TID`], then re-lower the SAME node at `m=1` reading it.
///
/// The extraction is `hidden/64` single-stick `[1,64]` identity copies. `hidden[mq, hidden]` is
/// `RowBlocked`, so logical `(r,c)` sits at `(c/64)·(mq·64) + r·64 + (c%64)`: row `r`'s stick-group `j`
/// is 64 CONTIGUOUS elements at `(j·mq + r)·64`, and the `[1, hidden]` destination's group `j` is 64
/// contiguous elements at `j·64`. Both ends are single-row/single-stick, so both walk at `lanes` — the
/// only coordInfo stride fp16 can represent (see [`LAST_HIDDEN_TID`]).
///
/// The re-lowered matmul goes through [`lower_matmul_node`], NOT a hand-rolled `assemble_matmul_off`.
/// That path owns everything the vocab-wide output depends on — the `DeviceWidth::for_output` stick
/// padding that keeps granite's awkward 49155 columns on all 32 cores, the util-floor and sub-stick
/// guards, and (for an arity-3 weight) the whole per-token fp8 activation-quantize chain, which a hand
/// emit would silently skip and feed the fp8 kernel raw fp16. granite-3.1-2b's own lm_head happens to
/// be arity-2 fp16 (the `[MM]` bake line reports `arity=2`), but nothing here depends on that.
fn lower_prefill_lm_head_at_m1<F: RopeForm>(
    node: &SubtileNode<F>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let a = node.inputs.first().ok_or_else(|| {
        SuperDscError(format!(
            "prefill lm_head t{}: MatmulTile with no A operand",
            node.output.tensor.index() as u32
        ))
    })?;
    let (mq, hidden) = (a.region.rows.len, a.region.cols.len);
    let stk = Fp16::ELEMS_PER_STICK;
    if hidden % stk != 0 {
        return Err(SuperDscError(format!(
            "prefill lm_head t{}: hidden={hidden} is not a whole {stk}-fp16 stick, so the last prompt \
             row is not a run of whole stick-groups — the per-stick extraction cannot address it",
            node.output.tensor.index() as u32
        )));
    }
    // The row to extract, from the SSOT the Kani proof `selector_lastrow_picks_last_row` pins.
    let row = scratchy_subtile::sdsc_abstract::selector_lastrow_col(mq as usize) as u32;
    let last_hidden = crate::wiring::act_name(LAST_HIDDEN_TID);
    if let Some(l) = layout {
        l.synth(bundle::PlaceId::Act(LAST_HIDDEN_TID), &[1, hidden]);
    }
    let src = rb(&ds_name(a), 1, stk);
    let dst = rbo(&last_hidden);
    let mut ops: Vec<EmittedOp> = (0..hidden / stk)
        .map(|j| {
            assemble_pointwise_broadcast_off(
                &format!("lmlast{j}_o{}", node.output.tensor.index() as u32),
                "identity",
                scratchy_subtile::sdsc_abstract::RowCount::of_token_rows(1),
                scratchy_subtile::sdsc_abstract::BlockCols::of_one_stick(
                    scratchy_subtile::sdsc_abstract::Lanes::FP16,
                ),
                &[In::sliced(
                    &src,
                    // row `j*mq + row` of a `[.., stk]` staging — a corner, not a product.
                    scratchy_subtile::addr::rc_of(
                        (j + 1) * mq + row + 1,
                        stk,
                        j * mq + row,
                        0,
                        Df::Fp16,
                    ),
                )
                .ew()],
                &dst,
                // Column `j*stk` of the single-row `[1, hidden]` last-hidden staging — a corner on
                // that tensor, not a product. At one row the stick plane term is 0, so this is the
                // same address either way; saying it as a coordinate keeps the last hand-built offset
                // in this file from being the one nobody notices.
                scratchy_subtile::addr::rc_of(1, hidden, 0, j * stk, Df::Fp16),
                sym_id_base,
                layout,
            )
        })
        .collect();
    // Re-lower at m=1 reading `last_hidden` — the weight (and its fp8 scale) are untouched.
    let mut at_m1 = node_at_one_row(node);
    at_m1.inputs[0] = scratchy_subtile::subtile_ir::TensorRegion {
        tensor: scratchy_subtile::subtile_ir::TensorId::from_index(LAST_HIDDEN_TID as usize),
        region: scratchy_subtile::subtile_ir::Region {
            rows: scratchy_subtile::subtile_ir::Range::new(0, 1),
            cols: scratchy_subtile::subtile_ir::Range::new(0, hidden),
        },
    };
    ops.extend(lower_matmul_node(&at_m1, sym_id_base, layout, quantized)?);
    Ok(ops)
}

/// Outcome of lowering ONE SubtileIR node. Shared by the UNROLLED
/// [`lower_graph_to_superdsc`] walk and the RE-ROLLED tape-driven walk so the
/// per-op `match` lives exactly once (the reroll just changes WHICH nodes are
/// walked + how many times the body runs, not how each op lowers).
enum NodeLowering {
    Ops(Vec<EmittedOp>),
    /// An op kind not yet lowerable (collected into the hard-error worklist).
    Unhandled(String),
    /// A recognized data-movement op routed host-side (not a SuperDSC tile).
    HostRouted(&'static str),
}

/// The rotary lowering, waiting for its head dim to become a const — the consumer side of
/// [`with_config_head_dim`]. It exists so the arms of that door can be GENERATED: a callback trait
/// takes one impl and any number of arms, where a `match` written here would need one line per
/// head dim, written by a human, and therefore a list of head dims in a source file.
struct LowerRope<'a, F: RopeForm> {
    node: &'a SubtileNode<F>,
    sym_id_base: &'a mut i64,
    layout: Option<&'a BundleLayout>,
    rows_are_requests: bool,
}

impl<F: RopeForm> scratchy_subtile::model_geometry::OnHeadDim for LowerRope<'_, F> {
    type Out = Result<Vec<EmittedOp>, SuperDscError>;
    fn on_head_dim<const HD: u32>(self) -> Self::Out {
        lower_rope_node::<F, HD>(
            self.node,
            self.sym_id_base,
            self.layout,
            self.rows_are_requests,
        )
    }
}

/// The attention lowering, waiting for its head geometry to become consts — the consumer side of
/// [`with_config_attn_geometry`], and the same reason as [`LowerRope`].
struct LowerAttn<'a, F: RopeForm> {
    node: &'a SubtileNode<F>,
    cap: u32,
    active_cap: ActiveCap,
    rows_are_requests: bool,
    sym_id_base: &'a mut i64,
    layout: Option<&'a BundleLayout>,
}

impl<F: RopeForm> scratchy_subtile::model_geometry::OnAttnGeometry for LowerAttn<'_, F> {
    type Out = Result<Vec<EmittedOp>, SuperDscError>;
    fn on_geometry<const NQH: u32, const NKVH: u32, const HD: u32>(
        self,
        geom: scratchy_subtile::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
    ) -> Self::Out {
        lower_attn_node::<F, NQH, NKVH, HD>(
            self.node,
            geom,
            self.cap,
            self.active_cap,
            self.rows_are_requests,
            self.sym_id_base,
            self.layout,
        )
    }
}

/// Lower ONE [`SubtileNode`] to its SuperDSC op(s) — the single source of the
/// per-`SubOp` match. `ir` is needed only for `AttnDecode`'s cache-capacity lookup.
fn lower_one_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    // Swept KV extent for AttnDecode (paged-attn ladder rung); FULL ⇒ full cap. See `lower_attn_node`.
    active_cap: ActiveCap,
    // See `lower_attn_node`: whether this bundle's rows are separate requests.
    rows_are_requests: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    // Threaded to `lower_matmul_node` so fp8 activation quantization is shared across matmuls (see there).
    quantized: &mut std::collections::HashSet<String>,
) -> NodeLowering {
    use NodeLowering::{HostRouted, Ops, Unhandled};
    // ── PREFILL (m>1) LM-HEAD TAIL, FOLDED TO m=1 — the prefill bundle produces the FIRST generated
    // token's logits itself, so TTFT is ONE forward, not two. ──
    // The vocab-wide (≈49159-col) lm_head cannot run at m>1: it ALWAYS time-tiles, and per-row (m>1)
    // time-tiling is unimplemented (design-risk-4 Err). Only the LAST prompt token's logits are ever
    // read, so the tail runs at m=1 over `last_hidden[1, hidden]` = row `selector_lastrow_col(mq)` of the
    // final-norm output — which is exactly the shape the PROVEN decode path lowers. See
    // [`LAST_HIDDEN_TID`] for why the extraction is per-stick copies rather than a one-hot matmul.
    //
    // DETECT BY VOCAB-WIDTH, not `output tid == ir.result`: granite has a LOGITS ScalarMul AFTER the
    // lm_head matmul, so `ir.result` is the ScalarMul's output — the matmul's OWN tid never equals it (the
    // observed miss: the lm_head matmul t1129 kept time-tiling because ir.result was the ScalarMul t1130).
    // The lm_head matmul AND the logits ScalarMul are the ONLY ops whose output spans the result cols
    // (vocab); every intermediate is hidden/intermediate width. So BOTH re-lower at m=1.
    // The mq=1 DECODE bundle is UNAFFECTED (out_rows==1 ⇒ not the prefill tail ⇒ lowered as before).
    let result_cols = ir.tensors[ir.result.index() as u32 as usize].cols;
    // NOT WHEN THE ROWS ARE REQUESTS. The fold is sound only because a prompt's rows are one
    // sequence, so all but the last row's logits are dead. In a batched-decode bundle each row is a
    // DIFFERENT request and every row's logits are sampled — folding to the last row would hand the
    // whole batch request B-1's token. The tail then time-tiles at m=B, which is now addressable:
    // the per-trip advance comes from `StickLayout::dev_off` (`time_tile_sticklayout_stride_tiles_disjoint`),
    // not the flat row-major form that aliased above one row.
    let is_prefill_lm_head_tail = node.output.region.cols.len == result_cols
        && node.output.region.rows.len > 1
        && !rows_are_requests;
    match &node.op {
        // The rest of the arch vocabulary. It reaches this emitter because the SHARED
        // front end expresses every op instead of asserting the unsupported ones away
        // in `lower_region` — which is the point: the IR carries the fact and the
        // TARGET says whether it has a kernel. Enumerated, never `_`, so adding a
        // SubOp is E0004 here rather than a surprise at emission.
        SubOp::TanhSoftCap
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::GateSplit { .. }
        | SubOp::GateApply
        | SubOp::GateScale
        | SubOp::LoadPixels { .. }
        | SubOp::LoadPosEmbeds { .. }
        | SubOp::EmbeddingGather { .. }
        | SubOp::VisionRope
        | SubOp::VarlenAttention { .. }
        | SubOp::EncoderAttn { .. }
        | SubOp::GatedDeltaNet
        | SubOp::GemmaMoe { .. }
        | SubOp::Moe { .. }
        | SubOp::Mean => Unhandled(format!("{:?} has no SuperDSC kernel", node.op)),
        // ⛔ THE ONE PLACE THAT MUST IMPLEMENT IT, so the refusal lives here
        // and names the required lowering rather than the op.
        SubOp::Reshape => Unhandled(
            "SubOp::Reshape reached the SuperDSC lowering. It must become a RESTICKIFY \
                 (a real re-laying copy), NOT a placement alias: `dev_off_stk` places (i,j) \
                 at (j/stk)*(a*stk)+i*stk+(j%stk) where `a` is the ROW COUNT, so two views \
                 over one buffer with different extents disagree about every element. And \
                 `declare_arrangement` will NOT catch an alias — it keys on tensor NAME, and \
                 an alias gives the two views two names."
                .to_string(),
        ),
        SubOp::MatmulTile { .. } if is_prefill_lm_head_tail => {
            match lower_prefill_lm_head_at_m1(node, sym_id_base, layout, quantized) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::MatmulTile { .. } => match lower_matmul_node(node, sym_id_base, layout, quantized) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::SumReduce => match lower_sumreduce_node(node, sym_id_base, layout) {
            Ok(o) => Ops(vec![o]),
            Err(e) => Unhandled(e.0),
        },
        SubOp::Elementwise(kind) => {
            match lower_elementwise_node(node, *kind, sym_id_base, layout) {
                Ok(o) => Ops(vec![o]),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::SiluMul => match lower_silumul_node(node, sym_id_base, layout) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        // ⛔ SPYRE'S RMSNORM MULTIPLIES BY THE STORED GAIN. The gemma-class (1 + w)
        // convention needs a different kernel, and running the Scale one over a
        // zero-centred gain scales every normalized activation by roughly nothing — a
        // model that loads, runs, and is quietly wrong. So it refuses BY NAME.
        //
        // It can reach here at all because the shared front end now EXPRESSES the
        // convention instead of asserting it away in `lower_region`. That is the trade:
        // the IR carries the fact, and the target says whether it has a kernel for it.
        SubOp::RmsNorm {
            eps,
            gain: scratchy_subtile::subtile_ir::GainConvention::Scale,
        } => match lower_rmsnorm_node(node, *eps, sym_id_base, layout) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::RmsNorm {
            gain: scratchy_subtile::subtile_ir::GainConvention::OnePlusScale,
            ..
        } => Unhandled(format!(
            "RmsNorm t{} uses the (1 + w) gain convention, for which this emitter has \
             no kernel — its rmsnorm multiplies by the stored gain",
            node.output.tensor.index() as u32,
        )),
        SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
            // ⭐⭐ THE ONE PLACE THE HEAD DIM STOPS BEING A VALUE. Every head_dim-dependent decision
            // downstream is a branch on a CONST, which is reviewable and guardable; a branch on a
            // value is neither. The door's arms are every head dim the workspace's model configs
            // declare, generated by the build script — a head dim with no arm is a loud bake error,
            // and it is answered by a `config.json`, never by an edit here.
            match with_config_head_dim(
                *head_dim,
                LowerRope {
                    node,
                    sym_id_base,
                    layout,
                    rows_are_requests,
                },
            ) {
                Some(Ok(v)) => Ops(v),
                Some(Err(e)) => Unhandled(e.0),
                None => Unhandled(format!(
                    "RopeRotate/RopeAppend head_dim {} has no const-generic instantiation. The head \
                     dim parameterises the device layout (slabs = head_dim/lanes, and the head-major \
                     collapse is valid only at head_dim == lanes), so it must be a const, not a \
                     value. The instantiations are read from the model configs in scope ({}); this \
                     head dim belongs to none of them.",
                    head_dim.get(),
                    scratchy_subtile::model_geometry::geometry_sources(),
                )),
            }
        }
        SubOp::AttnDecode {
            layout: kv, geom, ..
        } => {
            let cap = ir.tensors[kv.cache_tensor().index() as u32 as usize].rows;
            // ⭐⭐ THE ONE PLACE THE MODEL'S HEAD GEOMETRY STOPS BEING VALUES for the attention path.
            // The `#[forward]` macro parsed these numbers out of the model config and minted them
            // onto the tape node; the lowering runs INSIDE that macro's expansion, so this door is
            // where they meet the type system — one instantiation of the whole attention lowering
            // per geometry the workspace's configs declare, and the arms come from the build
            // script's read of those same files. A geometry with no arm is a loud bake error; a
            // geometry whose kv-head count does not divide its query-head count has no arm at all,
            // because `AttnGeometry` cannot be named at it.
            match with_config_attn_geometry(
                *geom,
                LowerAttn {
                    node,
                    cap,
                    active_cap,
                    rows_are_requests,
                    sym_id_base,
                    layout,
                },
            ) {
                Some(Ok(v)) => Ops(v),
                Some(Err(e)) => Unhandled(e.0),
                None => Unhandled(format!(
                    "AttnDecode geometry ({geom}) has no const-generic instantiation. The head \
                     counts and the head dim parameterise the device layout (GQA grouping, slabs, \
                     head strides), so they must be consts, not values. The instantiations are read \
                     from the model configs in scope ({}); this geometry belongs to none of them.",
                    scratchy_subtile::model_geometry::geometry_sources(),
                )),
            }
        }
        SubOp::RmsNormReduce { .. } | SubOp::RmsNormApply { .. } => {
            HostRouted(host_glue_kind(&node.op))
        }
        // ScalarMul (granite embedding/residual/attn/logits multipliers): on-device pointwise `mul` by a
        // bound `[1,1]` scale const (the ATTN_SCALE mechanism) — NOT folded into weights, NOT host-routed.
        // The vocab-wide LOGITS ScalarMul is the second half of the lm_head tail, so in the m>1 prefill
        // bundle it re-lowers at m=1 over the single logits row the matmul above wrote (same reason, same
        // reshape — see is_prefill_lm_head_tail above).
        SubOp::ScalarMul { scale } if is_prefill_lm_head_tail => {
            match lower_scalarmul_node(&node_at_one_row(node), *scale, sym_id_base, layout) {
                Ok(o) => Ops(vec![o]),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::ScalarMul { scale } => match lower_scalarmul_node(node, *scale, sym_id_base, layout)
        {
            Ok(o) => Ops(vec![o]),
            Err(e) => Unhandled(e.0),
        },
    }
}

/// Walk the fused [`SubtileIR`] and emit one SuperDSC op per tile-op, OWNING the
/// 32-core work-division. Returns `(op_name, SdscOp)` in topological order
/// (`ir.nodes` is already a valid eval order). FLOP-dominant matmuls are lowered
/// first — they carry the 32-core split that the sengraph path wastes
/// (`SENCORES 1→14.4ms vs 32→9.9ms`, only 1.45×).
///
/// CLEAN PARTITION (the full-model integration): the tape is split into three
/// classes — (1) PURE-COMPUTE ops (matmul/ew/reduce/rmsnorm/silu) → work-divided
/// SuperDSC tiles; (2) HOST-GLUE ops (RoPE rotate / rope_append / decode attention
/// / pre-chunked RmsNorm halves) → explicitly ROUTED host-side (run through the
/// existing resident worker/shim machinery, not a SuperDSC tile, matching
/// torch-spyre's own host/SDSC split); (3) anything else → a HARD build-time `Err`
/// (never a silent skip → never silently-wrong on-card output). The `match` is
/// EXHAUSTIVE over `SubOp` with no catch-all, so a newly-added variant forces a
/// `cargo build` error here until it is classified.
/// The ACTIVE KV sweep extent baked into a decode-attention bundle — one `sk_bucket` ladder rung.
///
/// The KV STORAGE is always the full `cap`; a bundle baked at a smaller `ActiveCap` sweeps only its
/// first [`resolve`](ActiveCap::resolve) slots (O(active) attention over the SAME resident cache). A
/// LADDER of decode bundles — one per rung — lets the runtime pick the smallest sweep ≥ the live KV
/// length while sharing one resident weights+KV. A newtype so the swept extent is never confused with
/// the storage `cap`, the query count `mq`, or a bare token count floating around as `u32`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ActiveCap(u32);

impl ActiveCap {
    /// Sweep the full storage cap — the ceiling rung (ladder disabled). Emits byte-identically to the
    /// pre-ladder path.
    pub const FULL: ActiveCap = ActiveCap(0);

    /// Sweep NO resident prefix at all — `nb == 0`, so the attention emits ZERO prefix blocks and the
    /// new-token block alone seeds and finalizes the online softmax.
    ///
    /// Valid ONLY for a chunk whose `start == 0`, i.e. one with no resident prefix to attend. That is
    /// every single-chunk prompt, which is the common TTFT case — and there the 4 prefix blocks are
    /// masked out in full by pmask, so they launch 308 of the 710 ops in a prefill layer (43%) to
    /// compute nothing: 12,320 launches per forward, ~4-6 ms at the measured 0.31-0.47 us per launch,
    /// plus ~1.2 ms of masked MACs.
    ///
    /// Correct because the new-token block is already the `first` block: with `first` it writes
    /// `run_m`/`run_l`/`run_o` directly rather than folding, and the finalize is `out = run_o / run_l`
    /// — which for a single block is exactly `exp(s - max) / sum`. (`assemble_attn`'s note that the
    /// seed "is still followed by at least one prefix-block combine" describes the numerical ORDERING
    /// argument for seeding from the new block, not a correctness requirement for a combine to exist.)
    ///
    /// A DISTINCT sentinel is required because `FULL` is `0`: "sweep everything" and "sweep nothing"
    /// would otherwise be the same value. `u32::MAX` can never be a legitimate slot request (it is
    /// neither `<= cap` nor stick-aligned), so it cannot collide with a real rung.
    pub const NONE: ActiveCap = ActiveCap(u32::MAX);

    /// A rung requesting a swept extent of `slots` KV positions. Stick-alignment + `≤ cap` validation
    /// is applied by [`resolve`](ActiveCap::resolve) against the bundle's real storage cap.
    pub const fn new(slots: u32) -> ActiveCap {
        ActiveCap(slots)
    }

    /// The raw requested slot count (0 for [`FULL`](ActiveCap::FULL)). For serialization/display only;
    /// use [`resolve`](ActiveCap::resolve) for the concrete tile extent.
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The concrete swept extent for a bundle whose storage cap is `cap` (`stick` = 64-elem tile
    /// alignment): [`FULL`](ActiveCap::FULL), or any out-of-range / mis-aligned request, resolves to
    /// the full `cap` (sweep everything); otherwise the requested extent. The ONLY place a rung becomes
    /// a tile extent.
    pub fn resolve(self, cap: u32, stick: u32) -> u32 {
        // NONE resolves to 0 ⇒ `nb = 0` ⇒ no prefix blocks at all. Checked FIRST so it can never be
        // mistaken for an out-of-range request and silently widened to the full cap.
        if self == ActiveCap::NONE {
            return 0;
        }
        if self.0 > 0 && self.0 <= cap && self.0.is_multiple_of(stick) {
            self.0
        } else {
            cap
        }
    }

    /// The INTERIOR decode ladder rungs strictly below `cap` — the KV spans where bounding the sweep is
    /// worth a separate bundle. The caller adds `cap` itself as the ceiling rung (the full-cap fallback
    /// that MUST exist so any context ≤ cap is correct). Env `SCRATCHY_SUPERDSC_DECODE_RUNGS="512,2048"`
    /// overrides the default set. Each rung is stick-aligned (up) and `< cap`; sorted ascending, deduped.
    /// Empty when `cap` is small enough that no interior rung fits (⇒ single full-cap bundle = disabled).
    pub fn decode_ladder(cap: u32) -> Vec<ActiveCap> {
        let stick = Fp16::ELEMS_PER_STICK;
        // THE LADDER, and there is no other. It used to sit behind
        // `SCRATCHY_SUPERDSC_DECODE_RUNGS`, so which rungs a bundle baked depended on the shell
        // that ran the build.
        //
        // Geometric: each rung bounds the decode sweep to ~2x the true length. The low rungs
        // (64/128/256) recover short-context tok/s; the high rungs (1024/2048) keep long context
        // bounded below the full cap. Only rungs < cap survive; `cap` is always the ceiling.
        // Cheap: N x KB body programs, one shared resident KV.
        //
        // 64 added (2026-07-28): the attention emitter blocks the resident cache in single-stick
        // (64-element) chunks (a real hardware reduce-MAX limit — see
        // reference_spyre_multirow_reduce_not_broken.md). At rung 128 (the previous floor) that is
        // ALWAYS 2 resident blocks (128/64) even at the very start of generation, when the true
        // context is far shorter — one whole block-fold's worth of matmuls + reduce/pointwise ops
        // wasted on masked-out slots every single step. A 64 rung makes nb=1 for short contexts,
        // recovering that op count through a proven-safe mechanism (a smaller sweep, not a wider
        // reduce) — unlike widening the block itself past one stick, which a real on-card test
        // showed breaks coherence (see the reverted "LX-tile block" commit).
        let base: Vec<u32> = vec![64, 128, 256, 512, 1024, 2048];
        let mut rungs: Vec<u32> = base
            .into_iter()
            .map(|r| r.next_multiple_of(stick))
            .filter(|&r| r > 0 && r < cap)
            .collect();
        rungs.sort_unstable();
        rungs.dedup();
        rungs.into_iter().map(ActiveCap::new).collect()
    }
}

pub fn lower_graph_to_superdsc<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // Swept KV extent for the decode attention (paged-attn ladder rung); FULL ⇒ full cap (byte-identical).
    active_cap: ActiveCap,
    // See `lower_attn_node`. Prefill callers pass false.
    rows_are_requests: bool,
) -> Result<(Vec<EmittedOp>, BundleLayout), SuperDscError> {
    // GLOBAL ≤7-segment memory layout (task #55) computed ONCE for the whole bundle.
    // Every op resolves each tensor's HBM address from this (by `t{id}` name) so a
    // tensor shared across ops gets the SAME address — fixing the per-op `arg_index`
    // segment-aliasing bug. Threaded as `Some(&layout)` into every node-lowering; the
    // populated layout (incl. synthetic seg3 offsets) is RETURNED for the manifest.
    let mut bundle_layout = compute_bundle_layout(ir, weight_ids, rows_are_requests)?;
    let layout = Some(&bundle_layout);
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(ir.nodes.len());
    // The SINGLE monotonic negative-symbol-id counter for the WHOLE bundle (design
    // risk #1): every tiled tensor/core gets `-(++sym_id_base)`, threaded through
    // each node-lowering so ids never collide across ops in one bundle (torch-spyre
    // `symbol_id_offset_counter`). A per-op restart would alias addresses and
    // silently corrupt — we assert disjointness below.
    let mut sym_id_base: i64 = 0;
    // Collect EVERY distinct unhandled op kind (the full worklist) in one pass —
    // reconnaissance, not a silent skip: a non-empty set is a HARD error so a
    // partial (silently-wrong) bundle is never baked (guard-every-crash rule).
    let mut unhandled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Ops explicitly ROUTED host-side (RoPE rotate / rope_append / decode attention /
    // pre-chunked RmsNorm halves) — recognized data-movement glue, not a silent skip.
    let mut host_routed: std::collections::BTreeSet<&'static str> =
        std::collections::BTreeSet::new();
    // UNROLLED walk: lower every node via the shared per-op `lower_one_node`. (The
    // RE-ROLLED tape-driven path reuses the SAME helper, walking only the loop body
    // once — see `lower_subtile_tape_to_superdsc`.) MatmulTile's Err is still a hard
    // stop; everything else collects into the worklist/host-routed sets below.
    // fp8 activation-quantize dedup, bundle-scoped: keyed on the activation `t{id}`, so q/k/v (same rms₁
    // output) share ONE quant, while a different layer's activation (distinct tid) never false-shares.
    let mut fp8_quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    for node in &ir.nodes {
        match lower_one_node(
            node,
            ir,
            active_cap,
            rows_are_requests,
            &mut sym_id_base,
            layout,
            &mut fp8_quantized,
        ) {
            NodeLowering::Ops(v) => ops.extend(v),
            NodeLowering::Unhandled(s) => {
                // A MALFORMED matmul is an immediate hard stop (it must never bake);
                // every other op kind accumulates into the worklist for one combined Err.
                if matches!(node.op, SubOp::MatmulTile { .. }) {
                    return Err(SuperDscError(s));
                }
                unhandled.insert(s);
            }
            NodeLowering::HostRouted(s) => {
                host_routed.insert(s);
            }
        }
    }
    if !unhandled.is_empty() {
        return Err(SuperDscError(format!(
            "{} SubtileIR op kind(s) not yet lowered to SuperDSC ({} SuperDSC op(s) emitted ok, \
             {} host-routed). WORKLIST: [{}]",
            unhandled.len(),
            ops.len(),
            host_routed.len(),
            unhandled.into_iter().collect::<Vec<_>>().join(", "),
        )));
    }
    // (The former negative-symbol-id disjointness guard is GONE: the concrete-unroll
    // emits no symbols — per-trip files are globally indexed `sdsc_{i}.json`, unique
    // by construction — so there is no symbol space to alias.)
    // GUARD (priority-1, from OBSERVED on-card failure 2026-06-25): dxp_standalone
    // throws `DtException: allocNode (dsc2.cpp:3999)` when a dsc has an empty
    // scheduleTree_ — the per-tensor HBM allocate nodes are REQUIRED. Refuse to
    // bake a scheduleTree_-less op at BUILD time rather than dxp-crash on-card.
    for e in &ops {
        let name = &e.op_name;
        let op = &e.op;
        for dsc_map in &op.dscs_ {
            for (dname, dsc) in dsc_map {
                if dsc.scheduleTree_.is_empty() {
                    return Err(SuperDscError(format!(
                        "op {name} (dsc {dname}): empty scheduleTree_ — dxp_standalone \
                         DtException's at allocNode (dsc2.cpp:3999) without the per-tensor HBM \
                         allocate nodes. Emit them via the page_coalesce HBM coloring (task #50) \
                         before baking."
                    )));
                }
                // GUARD (priority-1, OBSERVED on-card 2026-06-25): an AllocNode with
                // an empty `coordinates_.coordInfo` makes the scheduler throw
                // `DtException: "There must be at least one valid candidate."`
                // (L3DlOpsScheduler.cpp:1195) — the per-dim tile-folds are required.
                for an in &dsc.scheduleTree_ {
                    let empty_coord = an
                        .coordinates_
                        .get("coordInfo")
                        .and_then(|c| c.as_object())
                        .map(|o| o.is_empty())
                        .unwrap_or(true);
                    if empty_coord {
                        return Err(SuperDscError(format!(
                            "op {name} (dsc {dname}, alloc {}): empty coordinates_.coordInfo — \
                             dxp's scheduler DtException's ('at least one valid candidate', \
                             L3DlOpsScheduler.cpp:1195) without the per-dim tile-folds. Emit the \
                             coordInfo (core_fold/corelet_fold/elem_arr from the work-division) \
                             before baking.",
                            an.name_
                        )));
                    }
                    // GUARD (priority-1, from OBSERVED on-card crash 2026-06-25):
                    // the startAddressCoreCorelet_ fold's [core,corelet] factors
                    // MUST equal the SDSC's declared coreFoldProp_/coreletFoldProp_
                    // sizes, else the FoldManager import throws "Different
                    // cardinality between json and caller" (foldInfrastructure.h:2775).
                    let attrs = &an.startAddressCoreCorelet_.dim_prop_attr;
                    let core_ok =
                        attrs.first().map(|f| f.factor_) == Some(op.coreFoldProp_.factor_);
                    let corelet_ok =
                        attrs.get(1).map(|f| f.factor_) == Some(op.coreletFoldProp_.factor_);
                    if !core_ok || !corelet_ok {
                        return Err(SuperDscError(format!(
                            "op {name} (alloc {}): startAddressCoreCorelet_ fold factors \
                             [{:?}] disagree with coreFoldProp_={} / coreletFoldProp_={} — dxp \
                             FoldManager throws 'Different cardinality' (foldInfrastructure.h:2775).",
                            an.name_,
                            attrs.iter().map(|f| f.factor_).collect::<Vec<_>>(),
                            op.coreFoldProp_.factor_,
                            op.coreletFoldProp_.factor_
                        )));
                    }
                }
            }
        }
    }
    // Synthetic intermediates were assigned seg3 offsets ABOVE the colored
    // intermediates during the walk; grow the Intermediate segment's byte count to
    // cover them so the executor allocates a seg3 region large enough (task #55/#56).
    let seg3 = SegRole::Intermediate.segment();
    let synth_high = bundle_layout.synth.borrow().next;
    if synth_high > bundle_layout.segment_bytes[seg3] {
        bundle_layout.segment_bytes[seg3] = synth_high;
    }
    Ok((ops, bundle_layout))
}

/// One RE-ROLLED SuperDSC decode — the layer loop stays ROLLED (not 30× unrolled).
/// `prefix` = pre-loop ops (embed), `body` = ONE layer's ops (the executor runs it
/// `iters`×), `suffix` = post-loop ops (final norm + lm_head). `per_layer[t0]` =
/// `[t0, t1, …, t_{iters-1}]` — the layer-v tensor id for each tensor the body
/// references by layer-0's id `t0` (the node outputs + per-layer weights/KV); the
/// executor binds `per_layer[t][v]`'s (already-placed) address at iteration `v` — the
/// ivar weight/KV/hidden threading. dxp then compiles THREE SMALL bundles (seconds)
/// instead of one 2047-op unrolled monster (40 min). `layout` places EVERY layer's
/// tensors (the full resident set), so each `per_layer[t][v]` has a real address.
pub struct RolledSuperDsc {
    pub prefix: Vec<EmittedOp>,
    pub body: Vec<EmittedOp>,
    pub suffix: Vec<EmittedOp>,
    pub iters: u32,
    pub layout: BundleLayout,
    pub per_layer: std::collections::BTreeMap<u32, Vec<u32>>,
    /// Per-layer byte stride of the WEIGHT segment (seg1): the executor binds layer
    /// `v`'s weights by passing `seg1_base + v·weight_stride` (the body's baked
    /// layer-0 offsets shift to layer-v). UNIFORM across all per-layer weights (a
    /// build guard enforces it). 0 if no per-layer weights.
    pub weight_stride: u64,
    /// Per-layer byte stride of the KV segment (seg2), same contract. 0 if none.
    pub kv_stride: u64,
    /// Bytes between two REQUESTS' KV within one page+layer — the launch shifts seg2 by
    /// `request * this` on top of the layer and page terms. 0 if this bundle has no paged KV.
    /// Straight from [`BundleLayout::kv_request_stride_bytes`]; see there for why it is published
    /// rather than re-derived.
    pub kv_request_stride: u64,
    /// Requests one page holds — `PagedKvPool::ROWS`, so the runtime can refuse a pool row this bundle
    /// cannot address. 0 when there is no paged KV.
    pub kv_request_rows: u32,
    /// The body's residual-stream INPUT tensor id (the first body node's input[0] —
    /// the layer's hidden-in). OUTPUT id (the last body node's output — hidden-out).
    /// The executor threads `hidden_out → hidden_in` between iterations (the
    /// loop-carried residual). `u32::MAX` if the body is empty.
    pub hidden_in_tid: u32,
    pub hidden_out_tid: u32,
    /// The suffix's residual INPUT tensor id (the first suffix node's input[0], e.g.
    /// the last layer's post-attn residual t780). The rerolled body writes its output
    /// to `hidden_out_tid` (the representative-iteration tid t360), which differs from
    /// `suffix_in_tid`, so the executor copies `hidden_out → suffix_in` ONCE after the
    /// loop — the body→suffix seam (analogous to the per-iter `hidden_out → hidden_in`).
    /// `u32::MAX` if there is no suffix. When it equals `hidden_out_tid` the copy is a
    /// no-op (placements coincide, as the prefix→body seam does).
    pub suffix_in_tid: u32,
}

/// RE-ROLLED tape-driven SuperDSC lowering — the mirror of `lower_subtile_tape_to_tk_tape`
/// for SuperDSC (the fix for the 40-min unrolled-bundle compile). Walks the rerolled
/// `tape` (`subtile_tape::reroll_subtile_tape`): pre-loop Computes → `prefix`; the
/// `OpenLoop(Const(iters))`..`CloseLoop` body → `body` (lowered ONCE via the shared
/// `lower_one_node`); post-loop Computes → `suffix`. Collects the per-layer tid map
/// (`Compute::per_layer_out` for outputs + `ComputeInput::External::per_layer` for
/// weights) for the executor's per-iteration address advance.
pub fn lower_subtile_tape_to_superdsc<F: RopeForm>(
    tape: &scratchy_subtile::subtile_tape::SubtileTape,
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // Swept KV extent for the decode attention (paged-attn ladder rung); FULL ⇒ full cap (byte-identical).
    // The driver calls this once per rung with a different `active_cap`; every rung shares the SAME
    // resident KV (storage stays `cap`) and differs only in the body's swept extents.
    active_cap: ActiveCap,
    // See `lower_attn_node`. Prefill callers pass false.
    rows_are_requests: bool,
) -> Result<RolledSuperDsc, SuperDscError> {
    use scratchy_subtile::subtile_tape::{ComputeInput, Instr, LoopBound};
    // ⛔ THE `set_rows_are_requests` / `RestoreRar` DANCE IS DELETED. It pushed the row KIND into a
    // thread-local for the duration of one bundle's lowering, with a `Drop` guard to restore it, purely
    // so the matmul splitter would not need the fact in its signature. That made the kind ambient: no
    // site was obliged to receive it, ~97 sites branched on the row COUNT instead, and a batched decode
    // was emitted as a prefill chunk everywhere the count could not tell them apart. The kind is a
    // COMPILE-TIME CONSTANT of the bundle (this crate is driven by a proc macro that knows the model and
    // the rung as literals), so it belongs in a type — `sdsc_abstract::QueryRows<ROWS_ARE_REQUESTS>` —
    // and in the signatures that need it.
    //
    // What survives is ONE perf gate: a decode batch must not split `mb` (the PT array holds the weight
    // stationary and streams M through it, so an `mb` split reloads the weight per split and cancels the
    // amortization batching exists to buy). That is a throughput/compatibility choice, not a statement
    // about what a row means, and it is named accordingly. See `matmul/dims.rs` for the const-generic
    // fix that removes even this.
    let _prev_split_gate =
        crate::ir::bridge::tiled_op_sdsc_op::matmul::set_split_mb_forbidden(rows_are_requests);
    struct RestoreSplitGate(bool);
    impl Drop for RestoreSplitGate {
        fn drop(&mut self) {
            crate::ir::bridge::tiled_op_sdsc_op::matmul::set_split_mb_forbidden(self.0);
        }
    }
    let _restore_split_gate = RestoreSplitGate(_prev_split_gate);
    let mut bundle_layout = compute_bundle_layout(ir, weight_ids, rows_are_requests)?;
    // ON-CARD RESIDUAL (unconditional): thread the loop-carried hidden IN-PLACE, no copy. Pre-scan the
    // loop body for hidden_in (first body node input[0]) + hidden_out (last body node output) and ALIAS
    // hidden_out's placement to hidden_in's → every iteration reads+writes ONE resident buffer, so the
    // residual threads with NO host round-trip and NO device copy (replaces the host thread_hidden).
    // WAR-safe: within a layer hidden_in's last read (computing h1 = h_in+attn) precedes hidden_out's
    // write (the final h_out = h1+mlp add); across iters the shared buffer carries the residual. The
    // shim's thread_hidden becomes a no-op once the placements coincide.
    {
        let (mut pseg, mut pf, mut plast, mut psuf): (u8, Option<u32>, Option<u32>, Option<u32>) =
            (0, None, None, None);
        for instr in tape.instrs() {
            match instr {
                Instr::OpenLoop { .. } => pseg = 1,
                Instr::CloseLoop { .. } => pseg = 2,
                Instr::Compute { node, .. } => {
                    if pseg == 1 {
                        if pf.is_none() {
                            pf = Some(node.index() as u32);
                        }
                        plast = Some(node.index() as u32);
                    } else if pseg == 2 && psuf.is_none() {
                        psuf = Some(node.index() as u32);
                    }
                }
                _ => {}
            }
        }
        if let (Some(f), Some(l)) = (pf, plast)
            && let Some(p) = bundle_layout
                .placements
                .get(&(ir.nodes[f as usize].inputs[0].tensor.index() as u32))
                .cloned()
        {
            let hout = ir.nodes[l as usize].output.tensor.index() as u32;
            // hidden_out (last body op) → hidden_in's buffer: per-iter residual threads in place.
            bundle_layout.placements.insert(hout, p);
            // suffix_in (first suffix op's input[0]) → the same buffer: the body→suffix seam threads
            // in place too, so BOTH host routings (thread_hidden + thread_to_suffix) are unnecessary.
            if let Some(s) = psuf {
                let sin = ir.nodes[s as usize].inputs[0].tensor.index() as u32;
                bundle_layout.placements.insert(sin, p);
            }
        }
    }
    let layout = Some(&bundle_layout);
    let mut sym_id_base: i64 = 0;
    let (mut prefix, mut body, mut suffix): (Vec<EmittedOp>, Vec<EmittedOp>, Vec<EmittedOp>) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut iters: u32 = 0;
    let mut seg: u8 = 0; // 0 = prefix, 1 = body (inside the layer loop), 2 = suffix
    // fp8 activation-quantize dedup (see `lower_matmul_node`): keyed on the activation `t{id}`. Within the
    // once-walked body, q/k/v (same rms₁ tid) share ONE quant; distinct-tid activations never false-share,
    // and the reusing matmul lands in the SAME segment as the quant it reuses (shared tid ⇒ same segment).
    let mut fp8_quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut unhandled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut per_layer: std::collections::BTreeMap<u32, Vec<u32>> =
        std::collections::BTreeMap::new();
    // First/last body Compute node (for the residual-stream hidden in/out tids).
    let mut first_body_node: Option<u32> = None;
    let mut last_body_node: Option<u32> = None;
    // First SUFFIX Compute node — its input[0] is the residual the suffix reads (the
    // last layer's post-attn residual, e.g. t780). The rerolled body writes its output
    // to the REPRESENTATIVE-iteration tid (hidden_out_tid, e.g. t360), NOT t780, so the
    // body→suffix seam must thread hidden_out → suffix_in at runtime (else the suffix
    // reads an unwritten slot = 0 → rmsnorm(0)=inf → garbage logits).
    let mut first_suffix_node: Option<u32> = None;
    // Matmul KERNEL weights (layer-0 tid, in=k, out=n) for the device re-tile manifest.
    let mut kernel0: Vec<(u32, u32, u32)> = Vec::new();
    // LAYER blocked-f16 (SCRATCHY_LAYER_KSPLIT): collected down_proj `(dp_per_layer tids, dev_out, b_blocks)`
    // for POST-LOOP block-weight placement overlay (bundle_layout is borrowed by `layout` inside the loop).
    for instr in tape.instrs() {
        match instr {
            Instr::OpenLoop { bound, .. } => {
                iters = match bound {
                    LoopBound::Const(it) => *it,
                    LoopBound::Runtime(_) => {
                        return Err(SuperDscError(
                            "reroll-superdsc: a runtime-bounded layer loop is unsupported (the \
                             decode layer count is a compile-time Const)"
                                .into(),
                        ));
                    }
                };
                seg = 1;
            }
            Instr::CloseLoop { .. } => seg = 2,
            Instr::Compute {
                node,
                inputs,
                per_layer_out,
                ..
            } => {
                let n = &ir.nodes[node.index()];
                // Collect matmul KERNEL weights for the device re-tile manifest: w =
                // inputs[1] is [k,n]=[in,out] row-major; the PT array needs the device
                // tile layout [out/64, in, 64]. in=A.cols (k), out=output.cols (n).
                // Arity-3 fp8 W8A8 matmuls MUST be collected here too: their weight (inputs[1]; inputs[2] is
                // w_scale) needs the 1-byte packed RetileDescriptor from the `fp8_weight_tids` branch below.
                // They were EXCLUDED (arity-2 only), so the shim staged them FLAT → scrambled weights → garbage
                // (`scr chat` incoherent). in_k/out derive identically (inputs[0].cols=k, output.cols=n); the
                // KSPLIT branch (out>16384) never fires for fp8 projs. Root-caused + on-card confirmed 2026-07-16.
                if matches!(n.op, SubOp::MatmulTile { .. })
                    && (n.inputs.len() == 2 || n.inputs.len() == 3)
                {
                    kernel0.push((
                        n.inputs[1].tensor.index() as u32,
                        n.inputs[0].region.cols.len,
                        // DEVICE out extent via the TYPE-SAFE `DeviceWidth` (the SAME rule as
                        // `lower_matmul_node`'s `n_dev` and the worker's weight zero-pad): granite lm_head
                        // 49159→49664. So the RetileDescriptor device_size, the per-core address, and the
                        // staged buffer all agree by construction.
                        DeviceWidth::for_output(
                            n.output.region.rows.len,
                            n.output.region.cols.len,
                            n.inputs[0].region.cols.len,
                        )
                        .get(),
                    ));
                    // KSPLIT fp32-merge (lm_head first-test, n>16384): the split weight ALSO needs B
                    // block-weight descriptors `[KB,N]` under reserved tids `ksplit_block_tid(b)`, matching
                    // the emit branch's block matmuls. Same gate + condition. The kernel0 descriptor loop
                    // below builds each as `[dev_out/64, KB, 64]` (in=KB) — exactly the block matmul's read
                    // (`ksplit_block_weight_kslice_offset_matches_read`). Worker gathers the K-slice bytes.
                }
                if seg == 1 {
                    if first_body_node.is_none() {
                        first_body_node = Some(node.index() as u32);
                    }
                    last_body_node = Some(node.index() as u32);
                }
                if seg == 2 && first_suffix_node.is_none() {
                    first_suffix_node = Some(node.index() as u32);
                }
                // Per-layer OUTPUT tids (the node's output tensor in each layer copy).
                if iters > 1 && per_layer_out.len() as u32 == iters {
                    let outs: Vec<u32> = per_layer_out
                        .iter()
                        .map(|nid| ir.nodes[nid.index()].output.tensor.index() as u32)
                        .collect();
                    per_layer
                        .entry(n.output.tensor.index() as u32)
                        .or_insert(outs);
                }
                // Per-layer WEIGHT/external tids.
                for ci in inputs.iter() {
                    if let ComputeInput::External {
                        tensor,
                        per_layer: pl,
                        ..
                    } = ci
                        && iters > 1
                        && pl.len() as u32 == iters
                    {
                        per_layer
                            .entry(tensor.index() as u32)
                            .or_insert_with(|| pl.iter().map(|t| t.index() as u32).collect());
                    }
                }
                // RESIDENT kct per-layer registration (kill-the-restickify residency): the score reads a
                // per-layer RESIDENT Kᵀ kernel `kct_resident_tid(k_id)`. It's neither an External nor a node
                // output, so the loops above never add it — insert it manually (mirror the downproj block
                // insert below): map each layer's K-cache source tid k_Lv → kct_resident_tid(k_Lv), keyed
                // under the layer-0 kct tid. REQUIRED so the seg2 uniformity guard validates kct's per-layer
                // stride matches k/v — the executor advances kct's seg2 base by v·kv_stride automatically, so
                // a non-uniform kct packing would SILENTLY read the wrong layer (runtime-errors-need-compile-
                // time-checks). per_layer[k_id] was just inserted by the External loop above.
                if let SubOp::AttnDecode { layout: kv, .. } = &n.op
                    && iters > 1
                    && let Some(k_layers) =
                        per_layer.get(&(kv.cache_tensor().index() as u32)).cloned()
                    && k_layers.len() as u32 == iters
                {
                    let kct_tids: Vec<u32> = k_layers
                        .iter()
                        .map(|&k_lv| kct_resident_tid(k_lv))
                        .collect();
                    per_layer
                        .entry(kct_resident_tid(kv.cache_tensor().index() as u32))
                        .or_insert(kct_tids);
                }
                // LAYER blocked-f16 (SCRATCHY_LAYER_KSPLIT): down_proj (large K) → B per-layer block weights.
                // Push each block's kernel0 `[KB,dev_out]` descriptor + its per-layer tid list; collect the
                // down_proj weight's per-layer list for the POST-LOOP placement overlay (block (v,b) at
                // dp_Lv.offset + b·KB·dev_out·2, so weight_stride is unchanged). All offsets Kani-proven.
                match lower_one_node(
                    n,
                    ir,
                    active_cap,
                    rows_are_requests,
                    &mut sym_id_base,
                    layout,
                    &mut fp8_quantized,
                ) {
                    NodeLowering::Ops(v) => match seg {
                        0 => prefix.extend(v),
                        1 => body.extend(v),
                        _ => suffix.extend(v),
                    },
                    NodeLowering::Unhandled(s) => {
                        if matches!(n.op, SubOp::MatmulTile { .. }) {
                            return Err(SuperDscError(s));
                        }
                        unhandled.insert(s);
                    }
                    NodeLowering::HostRouted(_) => {}
                }
            }
            Instr::AllocSlot { .. } | Instr::FreeSlot { .. } => {}
        }
    }
    if !unhandled.is_empty() {
        return Err(SuperDscError(format!(
            "reroll-superdsc: {} unhandled op kind(s): [{}]",
            unhandled.len(),
            unhandled.into_iter().collect::<Vec<_>>().join(", "),
        )));
    }
    if iters == 0 {
        return Err(SuperDscError(
            "reroll-superdsc: no layer loop in the rerolled tape (no OpenLoop/CloseLoop) — \
             reroll_subtile_tape found no repeating body"
                .into(),
        ));
    }
    // Device re-tile manifest: each matmul kernel weight (layer 0) + its per-layer
    // copies → a RetileDescriptor built SOLELY from the DeviceTileLayout witness (the
    // SAME source per_core_addr uses for the stride), so the shim's host re-tile and the
    // on-card per-core address cannot diverge. KERNEL layout = [in,out] sticked on out.
    // fp8 W8A8 weights (`input[1]` of an arity-3 MatmulTile) stage 1-byte / 128-elem stick (SEN143_FP8);
    // dense weights stay fp16 2-byte / 64-stick. The shim reads `word_length` from this descriptor to
    // size the H2D re-tile, so an fp8 weight MUST carry the fp8 descriptor or it is staged as 2-byte.
    let fp8_weight_tids: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.op, SubOp::MatmulTile { .. }) && n.inputs.len() == 3)
        .map(|n| n.inputs[1].tensor.index() as u32)
        .collect();
    for (w_tid, in_k, out_n) in &kernel0 {
        let desc = if fp8_weight_tids.contains(w_tid) {
            // fp8 W8A8 weight = the AIU matmulfp8 PACKED tile. Each 128-byte stick holds 64 N-cols each
            // carrying its 2 K-bytes BYTE-ADJACENT — the matmulfp8 in-fold (gen_fp8_kernel_in_fold) reads the
            // 2-pack as "2 fp8 per fp16-width slot, CONTIGUOUS". Device order (outer→inner):
            // [N/64 n-sticks, K/2 k-pairs, 64 n-inner, 2 k-inner]; device[n_stick][k_outer][n_in][k_inner] =
            // host[64·n_stick + 2N·k_outer + n_in + N·k_inner] = weight[2·k_outer+k_inner][64·n_stick+n_in].
            // The [.,.,2,64] order (2 K-rows as two separate 64-N blocks) and a FLAT 128-stick layout BOTH
            // scramble the weights → garbage; only this [.,.,64,2] order runs coherent (on-card 2026-07-16,
            // `scr chat` → "Paris"). (K even + N%64==0 hold for every granite fp8 proj.)
            let k = *in_k as u64;
            let n = *out_n as u64;
            if !k.is_multiple_of(2) || !n.is_multiple_of(64) {
                return Err(SuperDscError(format!(
                    "fp8 W8A8 weight t{w_tid}: packed tile needs K({k})%2==0 and N({n})%64==0"
                )));
            }
            RetileDescriptor {
                device_size: vec![n / 64, k / 2, 64, 2],
                // DISK ORDER, like the fp16 tile below. The worker no longer transposes, so this map
                // reads the `[out, in]` buffer safetensors stores. Converting it is mechanical: a term
                // that stepped an IN index by `x` was `x*n` against `[in, out]` and becomes `x*1`; a
                // term that stepped an OUT index by `y` was `y*1` and becomes `y*k`. So
                // `[64, 2n, 1, n]` → `[64k, 2, k, 1]`, which resolves `host[o*k + i]` for every
                // coordinate (checked exhaustively against the transposed map's `host[i*n + o]`).
                //
                // MISSING THIS BRANCH is what made granite-3.1-8b emit garbage: it is fp8, so its
                // GEMM weights come through here and not the fp16 path, and they were still being read
                // as though something had transposed them.
                stride_map: vec![64 * k, 2, k, 1],
                stick_size: Fp8::ELEMS_PER_STICK,
                word_length: Fp8::WORD_LENGTH,
            }
        } else {
            let tile = DeviceTileLayout::<Fp16>::new(
                &["in", "out"],
                "out",
                &[*in_k as u64, *out_n as u64],
            )?;
            RetileDescriptor {
                device_size: tile.device_size(),
                // DISK ORDER: the worker binds a GEMM weight in the `[out, in]` orientation
                // safetensors stores it, so the re-tile reads it there rather than from a transposed
                // copy. Same elements, one fewer pass over the model at load.
                stride_map: tile.stride_map_disk_order(),
                stick_size: Fp16::ELEMS_PER_STICK,
                word_length: Fp16::WORD_LENGTH,
            }
        };
        bundle_layout.kernel_weights.insert(*w_tid, desc.clone());
        if let Some(layers) = per_layer.get(w_tid) {
            for &t in layers {
                bundle_layout.kernel_weights.insert(t, desc.clone());
            }
        }
    }
    let seg3 = SegRole::Intermediate.segment();
    let synth_high = bundle_layout.synth.borrow().next;
    if synth_high > bundle_layout.segment_bytes[seg3] {
        bundle_layout.segment_bytes[seg3] = synth_high;
    }
    // Per-layer SEGMENT strides for the executor: WEIGHTS (seg1) + KV (seg2) advance
    // by `v·stride` per iteration (the body's baked layer-0 offsets shift to layer-v
    // when the executor passes `seg_base + v·stride`). Verify UNIFORMITY here (a
    // build-time guard) — a non-uniform packing would make the seg-base advance read
    // the WRONG layer's weights (silent garbage). seg3 intermediates (the loop-carried
    // hidden) are EXCLUDED (host-threaded, not strided). 0 = no per-layer tensor there.
    let w_seg = SegRole::Weight.segment();
    let kv_seg = SegRole::Kv.segment();
    let mut weight_stride: u64 = 0;
    let mut kv_stride: u64 = 0;
    for tids in per_layer.values() {
        if tids.len() < 2 {
            continue;
        }
        let Some(p0) = bundle_layout.placements.get(&tids[0]) else {
            continue;
        };
        let seg = p0.segment;
        let target = if seg == w_seg {
            &mut weight_stride
        } else if seg == kv_seg {
            &mut kv_stride
        } else {
            continue; // seg3 hidden / other: not stride-advanced
        };
        for w in tids.windows(2) {
            let (Some(a), Some(b)) = (
                bundle_layout.placements.get(&w[0]),
                bundle_layout.placements.get(&w[1]),
            ) else {
                return Err(SuperDscError(
                    "reroll-superdsc: a per-layer tensor is missing a layout placement".into(),
                ));
            };
            if a.segment != seg || b.segment != seg {
                return Err(SuperDscError(
                    "reroll-superdsc: a per-layer tensor changes segment across layers".into(),
                ));
            }
            let d = b.offset.wrapping_sub(a.offset);
            if *target == 0 {
                *target = d;
            } else if *target != d {
                return Err(SuperDscError(format!(
                    "reroll-superdsc: NON-UNIFORM per-layer stride in seg{seg} ({} vs {} bytes) — \
                     the executor advances the segment base by v·stride, which needs layers packed \
                     at a uniform stride. Reorder compute_bundle_layout to pack each layer's \
                     {} contiguously.",
                    *target,
                    d,
                    if seg == w_seg { "weights" } else { "KV" },
                )));
            }
        }
    }
    // Residual-stream hidden in/out tids for the executor's loop-carried threading:
    // the FIRST body node's input[0] (the layer's hidden-in, e.g. the rmsnorm x) and
    // the LAST body node's output (the layer's hidden-out, the final residual add).
    let hidden_in_tid = first_body_node
        .and_then(|nid| ir.nodes[nid as usize].inputs.first())
        .map(|r| r.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    let hidden_out_tid = last_body_node
        .map(|nid| ir.nodes[nid as usize].output.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    // The suffix's residual input (first suffix node's input[0]) — the executor threads
    // hidden_out → suffix_in once after the layer loop (the body→suffix seam). Same
    // input-ordering convention as hidden_in_tid (rmsnorm input[0] = x = the residual).
    let suffix_in_tid = first_suffix_node
        .and_then(|nid| ir.nodes[nid as usize].inputs.first())
        .map(|r| r.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    // ── DISCOVERY dump (SCRATCHY_SUPERDSC_SEGDUMP) ── which segment/tensor drives the
    //    footprint. Emit runs at cargo-build (AoT bake), so this lands in the build log.
    if std::env::var_os("SCRATCHY_SUPERDSC_SEGDUMP").is_some() {
        let sb = &bundle_layout.segment_bytes;
        let tot: u64 = sb.iter().sum();
        eprintln!(
            "[SEGDUMP] iters={iters} total={:.3}GB segs(GB)=[{}]",
            tot as f64 / 1e9,
            sb.iter()
                .map(|b| format!("{:.3}", *b as f64 / 1e9))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let mut pls: Vec<(&u32, &TensorPlacement)> = bundle_layout.placements.iter().collect();
        pls.sort_by_key(|(_, p)| std::cmp::Reverse(p.size));
        for (tid, p) in pls.into_iter().take(15) {
            eprintln!(
                "[SEGDUMP]   t{tid} seg{} role={:?} size={:.4}GB ({} B)",
                p.segment,
                p.role,
                p.size as f64 / 1e9,
                p.size,
            );
        }
        let syn = bundle_layout.synth.borrow();
        let mut szs: Vec<(&String, &u64)> = syn.sizes.iter().collect();
        szs.sort_by_key(|(_, s)| std::cmp::Reverse(**s));
        for (name, s) in szs.into_iter().take(10) {
            eprintln!(
                "[SEGDUMP]   synth {name} size={:.4}GB ({} B)",
                *s as f64 / 1e9,
                *s
            );
        }
    }
    let kv_request_stride = bundle_layout.kv_request_stride_bytes;
    // Stated only when there IS a request dimension, so an unpaged bundle refuses nothing.
    // ⛔ ALWAYS 0. There is no request dimension in a page any more, so there are no "request rows" for
    // the runtime to bound a shift by. A request is reached by its PAGE, through the host's block table.
    let kv_request_rows = 0u32;
    Ok(RolledSuperDsc {
        prefix,
        body,
        suffix,
        iters,
        layout: bundle_layout,
        per_layer,
        weight_stride,
        kv_stride,
        kv_request_stride,
        kv_request_rows,
        hidden_in_tid,
        hidden_out_tid,
        suffix_in_tid,
    })
}

// NOTE — STILL TO MIRROR FROM THE FIXTURE (the dsc tile-schedule internals, the
// hard tail of the emitter): the per-`dscs_` entry fields T_/Tel_/P_/Pel_/B_/
// ChipD_/CoreD_/CoreletD_/loopOrder_/loopProperties_/dataStageParam_/
// scheduleTree_ (per-core HBM start addresses via affine folds)/primaryDsInfo_/
// labeledDs_ (memOrg_ hbm/lx/l0)/computeOp_/pdsRelation_/pcfg_/target_. These
// come from torch-spyre scheduler.py + scratchpad planning; emitted next,
// matmul-first (mirror sdsc_bmm_autoBuffer.json), then add/mul/silu/rmsnorm/
// softmax, then the SubtileIR walk. Each on-card dxp failure during the in-
// scratchy AoT bake becomes a build-time guard (guard_superdsc_crash_patterns).

#[cfg(test)]
mod tests {

    /// ⭐⭐⭐ AN INDEX PAST ITS REGION IS A REFUSAL, NOT AN ALIAS. Every reserved tid used to be
    /// raw `u32` subtraction from a base — `SCALARMUL_SCALE_BASE - idx`, unchecked — so the 100th
    /// scale silently took `ksplit_block_tid(0)`'s slot. Two tensors, one tid, one placement: the
    /// second producer overwrites the first's bytes and the first's consumer reads them. On a
    /// device with no stack to attach to that is wrong output or a hang, reported by nothing.
    #[test]
    #[should_panic(expected = "reserved tid region overflow")]
    fn an_index_past_its_region_refuses_instead_of_aliasing_the_next_one() {
        // scalarmul owns 100 slots; the 101st is past its floor.
        let _ = scalarmul_scale_tid(RESERVED_REGIONS[1].slots as usize);
    }

    /// The regions tile the space downward without gaps between the ones that abut, and every
    /// declared base IS its region's base — the constants are derived from the table, so there is
    /// no second copy to drift.
    #[test]
    fn the_declared_bases_are_the_regions_bases() {
        assert_eq!(SCALARMUL_SCALE_BASE, RESERVED_REGIONS[1].base);
        assert_eq!(KCT_RESIDENT_BASE, RESERVED_REGIONS[2].base);
        // Each region sits strictly below the one before it. They need NOT abut: the K-split
        // regions were removed and their span is deliberately left as a hole, because a reserved
        // id is a number that has been baked into artifacts and sliding one up to close a gap
        // would silently repoint it.
        for w in RESERVED_REGIONS.windows(2) {
            assert!(
                w[1].base < w[0].floor(),
                "{} overlaps {} or sits above it",
                w[1].name,
                w[0].name
            );
        }
    }

    /// The last slot of each region is still ITS OWN, and the first slot of the next is not.
    #[test]
    fn a_regions_last_slot_belongs_to_it_and_the_next_tid_does_not() {
        for r in RESERVED_REGIONS {
            assert_eq!(r.at(r.slots - 1), r.floor(), "{}", r.name);
        }
        assert_eq!(scalarmul_scale_tid(99), RESERVED_REGIONS[1].floor());
    }

    /// ⭐⭐⭐ THE ARGUMENT ORDER OF `for_output` IS LOAD-BEARING, AND THIS IS THE MODEL THAT PROVES
    /// IT. granite-3.x-2b: lm_head is `[n=49155, k=2048]` on disk. Padding the OUTPUT axis `n` is
    /// the whole point — 49155 rounds to 769 sticks, which is PRIME, so the full-occupancy bump
    /// takes it to 800 sticks = 51200. Padding `k` instead is a NO-OP, because 2048 is already 32
    /// sticks and fills the machine.
    ///
    /// So a caller that passes `(k, n)` where `(n, k)` is meant gets a PLAUSIBLE answer — the
    /// weight's own contraction width, unpadded — and stages 49155 columns into a placement sized
    /// for 51200. The executor catches it as
    /// `host size 201338880 B != prod(device_size)*word_length (209715200 B)`, which is exactly
    /// `2048 * 49155 * 2` against `2048 * 51200 * 2`. That is a real failure this repo shipped:
    /// the staged `[rows, cols]` pair moved from a JSON manifest (which recorded the LOGICAL
    /// `[k, n]`) to `BoundWeight::staged_shape` (which returned the ON-DISK `[n, k]`) under the
    /// same field names, so every downstream use read `n` where it meant `k`.
    #[test]
    fn the_output_axis_is_what_gets_padded_and_swapping_k_and_n_is_silent() {
        const N: u32 = 49155; // granite vocab
        const K: u32 = 2048; // granite hidden
        let right = DeviceWidth::for_output(1, N, K).get();
        assert_eq!(right, 51200, "the output axis pads to full occupancy");
        assert_eq!(right % 64, 0);
        assert_eq!(
            right / 64 % MAX_CORES,
            0,
            "800 sticks = 25 per core on 32 cores"
        );

        // The swapped call is not an ERROR — it is a different, plausible number.
        let swapped = DeviceWidth::for_output(1, K, N).get();
        assert_eq!(
            swapped, K,
            "k is already core-splittable, so padding it is a no-op"
        );
        assert_ne!(swapped, right);

        // And the byte sizes are the two in the failure message: the host staged the weight
        // UNPADDED (`n * k`, because the no-op bump left the width alone) into a placement the
        // emitter had sized at the padded width (`k * n_dev`).
        assert_eq!(
            N as usize * K as usize * 2,
            201_338_880,
            "what the host staged"
        );
        assert_eq!(
            K as usize * right as usize * 2,
            209_715_200,
            "what the device wanted"
        );
    }
    use super::*;
    use scratchy_subtile::lower::GemmWeight;
    // ⛔ THIS WHOLE MODULE WAS DEAD. `matmul_opspec`/`matmul_dims`/`matmul_split_map`/
    // `reduce_opspec_df` moved out of this file into `ir::bridge::tiled_op_sdsc_op::{matmul,reduce}`
    // and the `use super::*` no longer reached them, so `cargo test -p scratchy-subtile --lib` failed
    // to COMPILE — which means every assertion below has been reporting nothing, including the three
    // fused-epilogue tests, while two epilogue fusions were built on that mechanism.
    //
    // ⭐ AN UNCOMPILABLE TEST TARGET IS INDISTINGUISHABLE FROM A PASSING ONE unless you read past the
    // integration-test results, and this crate's `tests/` directory is green — so `cargo test -p …`
    // printed dozens of `ok` lines with this target's `error:` above them. Same shape as the
    // `#![cfg(kani)]` proofs that were vacuous for ~100 commits.
    use crate::ir::bridge::tiled_op_sdsc_op::matmul::dims::{matmul_dims, matmul_split_map};
    use crate::ir::bridge::tiled_op_sdsc_op::matmul::opspec::matmul_opspec;
    use crate::ir::bridge::tiled_op_sdsc_op::reduce::reduce_opspec_df;

    /// Guard the SEN169_FP16 (1-6-9) encoding the reduce `scaling_factor` const needs.
    /// The bit patterns are pinned against the SFP-constant table (`plus1=0x3E00`,
    /// `minus1=0xBE00`, `fastSigmoidConst=0x3C00`=0.5) — those are SEN169, NOT IEEE f16.
    /// A regression here re-introduces the silent reduce mis-scale (mean 4600× too small,
    /// sum halved) that produced inf attention scores / garbage output on-card.
    #[test]
    fn sen169_encoding_pinned_to_sfp_table() {
        assert_eq!(sen169_bits(1.0), 0x3E00, "SEN169 1.0 (cf. SFP plus1)");
        assert_eq!(sen169_bits(-1.0), 0xBE00, "SEN169 -1.0 (cf. SFP minus1)");
        assert_eq!(
            sen169_bits(0.5),
            0x3C00,
            "SEN169 0.5 (cf. SFP fastSigmoidConst)"
        );
        assert_eq!(sen169_bits(0.0), 0x0000, "SEN169 +0");
        // The reduce scale that was the bug: must NOT equal the IEEE f16 bits.
        let inv576 = 1.0f32 / 576.0;
        assert_ne!(
            sen169_bits(inv576),
            half::f16::from_f32(inv576).to_bits(),
            "1/576 SEN169 must differ from IEEE f16 (the on-card mis-read)"
        );
        // Decode-round-trip within fp16 precision (1-6-9, bias 31).
        for v in [1.0f32 / 576.0, 1.0 / 9.0, 1.0 / 256.0, 1.0 / 2048.0] {
            let b = sen169_bits(v);
            let exp = ((b >> 9) & 0x3F) as i32 - 31;
            let mant = (b & 0x1FF) as f32 / 512.0;
            let decoded = (1.0 + mant) * 2f32.powi(exp);
            assert!(
                (decoded - v).abs() / v < 0.005,
                "SEN169 round-trip {v} → {decoded}"
            );
        }
    }

    /// The fp32 reduce (mq>1 rmsnorm mean(x²)) must encode its `scaling_factor`
    /// external const in the OP's data_format — IEEE_FP32 with the RAW f32 bits,
    /// NOT a SEN169_FP16 word. A fp16 const feeding an fp32 op is a mixed
    /// [fp16,fp32] op → DD2 "Unsupported result precision conversion"
    /// (SentientToProgIR/Utils.cpp:34). Mirrors torch-spyre `encodeConstant` →
    /// `BinaryConvert<uint32_t>(float)` for IEEE_FP32 (module.cpp:126). The fp16
    /// reduce path (`sen169_encoding_pinned_to_sfp_table`) is unchanged.
    #[test]
    fn fp32_reduce_scale_const_is_ieee_fp32() {
        // fp32 reduce → ReduceScalingFp32 with raw f32 bits of 1/N.
        let f32_spec = reduce_opspec_df(OpFunc::Mean, 64, 576, "r_x", "r_acc", true, Df::Fp32)
            .expect("fp32 mean reduce opspec");
        assert_eq!(
            f32_spec.op_info,
            OpInfo::ReduceScalingFp32((1.0f32 / 576.0).to_bits()),
            "fp32 reduce const must be raw IEEE f32 bits of 1/N"
        );
        // The serialized const carries dataFormat_ = IEEE_FP32 and the 32-bit word.
        let ci = scaling_factor_const_fp32((1.0f32 / 576.0).to_bits());
        assert_eq!(ci["0"]["dataFormat_"], "IEEE_FP32");
        let got = ci["0"]["data_"][0].as_u64().unwrap();
        assert_eq!(got, (1.0f32 / 576.0).to_bits() as u64);
        assert!(
            got > 0xFFFF,
            "a true fp32 word exceeds 16 bits; got {got:#x}"
        );

        // fp16 reduce path is untouched: still SEN169_FP16, 16-bit word.
        let f16_spec = reduce_opspec_df(OpFunc::Mean, 64, 576, "r_x", "r_acc", true, Df::Fp16)
            .expect("fp16 mean reduce opspec");
        assert_eq!(
            f16_spec.op_info,
            OpInfo::ReduceScaling(sen169_bits(1.0f32 / 576.0) as u32),
            "fp16 reduce const must stay SEN169_FP16"
        );
    }

    #[test]
    fn core_split_matches_fixture() {
        assert_eq!(core_split(384, 32), 32); // 384=2^7·3 → 32 (÷12)
        assert_eq!(core_split(320, 32), 32); // 320=2^6·5 → 32 (÷10)
        assert_eq!(core_split(64, 32), 32);
        assert_eq!(core_split(320, 9), 8); // not ÷9; largest divisor ≤9 is 8
        assert_eq!(core_split(48, 32), 24); // not ÷32; 48=16·3, largest ≤32 is 24
        assert_eq!(core_split(7, 32), 7); // prime ≤ max
        assert_eq!(core_split(13, 32), 13);
    }

    #[test]
    fn distribute_beats_single_core() {
        // The bmm case: out=384, mb=384 (outputs), in=64 (reduction).
        let dims = vec![
            ItDim {
                name: "mb",
                size: 384,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
            ItDim {
                name: "out",
                size: 384,
                is_reduction: false,
                is_stick: true,
                df: Df::Fp16,
            },
            ItDim {
                name: "in",
                size: 64,
                is_reduction: true,
                is_stick: true,
                df: Df::Fp16,
            },
        ];
        let splits = distribute_cores(&dims, MAX_CORES);
        let cores: u32 = splits.values().product::<u32>().max(1);
        // Must use all 32 cores — the entire point (sengraph auto-split wastes 31).
        assert_eq!(
            cores, 32,
            "work-division must fill all 32 cores: {splits:?}"
        );
        // And the validated WorkPlan accepts it (≤32 by type).
        let plan = WorkPlan::divide(&dims, MaxCores::<MAX_CORES>, distribute_cores).unwrap();
        assert_eq!(plan.cores_used().get(), 32);
    }

    #[test]
    fn bundle_layout_roles_and_packing() {
        // GLOBAL layout (task #55): weight source → seg1, activation source → seg0,
        // result → seg4 (logits), all 128 B aligned. A two-op chain produces one
        // seg3 intermediate; with a third op consuming nothing of the first, the
        // freed slot is REUSED (lifetime coloring).
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        let tensors = vec![
            TensorShape { rows: 4, cols: 8 },  // t0 = activation source
            TensorShape { rows: 8, cols: 16 }, // t1 = weight source
            TensorShape { rows: 4, cols: 16 }, // t2 = result (logits)
        ];
        let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
            tensor: TensorId::from_index(t as usize),
            region: ts[t].whole(),
        };
        let nodes = vec![SubtileNode {
            id: SubtileId::from_index(0),
            op: SubOp::MatmulTile {
                weight: GemmWeight::Dense,
            },
            inputs: vec![whole(0, &tensors), whole(1, &tensors)],
            output: whole(2, &tensors),
        }];
        let ir: SubtileIR = SubtileIR {
            tensors,
            num_sources: 2,
            nodes,
            result: TensorId::from_index(2),
            // Hand-authored fixture: there is no source op list to be the
            // provenance of, so the map is empty.
            op_output: Vec::new(),
        };
        let weight_ids: std::collections::HashSet<u32> = [1u32].into_iter().collect();
        let layout = compute_bundle_layout(&ir, &weight_ids, false)
            .expect("a layout for a plain matmul bundle");

        // ⭐ ROLES AND PACKING, ASSERTED AS THE RULES — not as magic numbers. This test pinned
        // `(Activation, seg 0, off 0, 64 B)` and both of those had since changed BY DESIGN, silently,
        // because the target would not compile:
        //   · seg 0 → 3: the dated `Activation = 3` / `Intermediate = 0` diagnostic swap (see `SegRole`),
        //     so a segment INDEX literal here restates a decision that already moved once.
        //   · 64 B → 4096 B: `nbytes` reserves the DEVICE footprint, and
        //     `bump_sticks_to_splittable` pads a 1-stick width up to 8 sticks so the work division can
        //     split it — 4 rows x 512 elems x 2 B.
        // So the segment comes from the role itself, and the size is checked against the invariants the
        // rule guarantees (at least the flat footprint, a whole number of sticks) rather than one
        // padding policy's current output. A role mix-up or an aliasing regression still fails; a
        // deliberate re-tune of the padding no longer reports a defect that is not there.
        let flat = |rows: u64, cols: u64| rows * cols * 2;
        for (tid, role, rows, cols) in [
            (0u32, SegRole::Activation, 4u64, 8u64),
            (1, SegRole::Weight, 8, 16),
            (2, SegRole::Logits, 4, 16),
        ] {
            let p = layout.placements[&tid];
            assert_eq!(p.role, role, "t{tid}'s role");
            assert_eq!(
                p.segment,
                role.segment(),
                "t{tid} lands in ITS ROLE's segment"
            );
            assert_eq!(
                p.offset, 0,
                "t{tid} is the only tensor in that segment, so it packs at 0"
            );
            assert!(
                p.size >= flat(rows, cols),
                "t{tid}'s reservation ({}) covers its flat footprint ({})",
                p.size,
                flat(rows, cols)
            );
            assert_eq!(
                p.size % (64 * 2),
                0,
                "t{tid}'s reservation is whole 64-elem fp16 sticks"
            );
            assert_eq!(
                layout.segment_bytes[p.segment],
                align128(p.offset + p.size),
                "segment {} is sized to its packed contents, 128 B aligned",
                p.segment
            );
        }
        // No intermediates in a single-op bundle — whichever segment that role currently owns.
        assert_eq!(layout.segment_bytes[SegRole::Intermediate.segment()], 0);

        // No two placements in the SAME segment overlap (build-time safety twin).
        let mut by_seg: std::collections::BTreeMap<usize, Vec<(u64, u64)>> = Default::default();
        for p in layout.placements.values() {
            by_seg
                .entry(p.segment)
                .or_default()
                .push((p.offset, p.size));
        }
        for ranges in by_seg.values() {
            for (i, &(o1, s1)) in ranges.iter().enumerate() {
                for &(o2, s2) in &ranges[i + 1..] {
                    assert!(
                        o1 + s1 <= o2 || o2 + s2 <= o1,
                        "segment alias {o1}+{s1} vs {o2}+{s2}"
                    );
                }
            }
        }
    }

    #[test]
    fn emit_sdsc_matmul_minimal_fields() {
        // The matmul OpSpec lowers to the frontend-minimal field set.
        let op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").unwrap();
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let sdsc = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
        let j = serde_json::to_string(&sdsc).unwrap();
        // Frontend-minimal memOrg: hbm+lx only, NO register file.
        assert!(j.contains("\"memOrg_\":{\"hbm\":{\"isPresent\":1},\"lx\":{\"isPresent\":1}}"));
        assert!(!j.contains("pelrf") && !j.contains("ptxrf"));
        // startAddr data keys carry SPACES "[c, 0, 0]".
        assert!(j.contains("[0, 0, 0]"));
        // Dropped scheduler fields are absent.
        assert!(!j.contains("gtrIdsUsed_") && !j.contains("pdsRelation_"));
        assert!(!j.contains("hbmStartAddress_") && !j.contains("lxBufferSize_"));
        assert!(!j.contains("stickRepl_") && !j.contains("unpadN_"));
        // exUnit is pt (sealed from OpFunc), and the fold factors agree.
        assert_eq!(sdsc.dscs_[0]["MatMul_0"].computeOp_[0].exUnit, "pt");
        assert_eq!(sdsc.coreFoldProp_.factor_, 32);
        assert_eq!(sdsc.coreletFoldProp_.factor_, 1);
    }

    #[test]
    fn emit_sdsc_matmul_fused_epilogue_broadcast_batch_scale_correct() {
        // The bug this test exists to pin: the pmask fusion's FIRST landing marked only "mb"
        // broadcast and forgot "y" (the GQA-group batch axis a batched score matmul ALSO needs
        // broadcast for a head-independent mask) — `set_scale_for_dim` silently no-op'd on the
        // missing marker at the time, so the emitter built a plausible-looking WRONG SdscOp with no
        // signal, and garbled generation on the card was the first anyone noticed. `broadcast_batch`
        // exists so the CALLER never has to name "y" (or re-derive whether it exists) at all — this
        // asserts a batched (`batch>1`, so `batch_dim_name()` returns `Some("y")`) matmul's fused
        // epilogue operand shows RedNonStick on BOTH "mb" and "y", Active on "out".
        let mut op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").unwrap();
        assert_eq!(op.time(), 1);
        assert_eq!(
            op.batch_dim_name(),
            Some("y"),
            "a batch=16 matmul must carry a real y dim"
        );
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs::One(
            scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[("mb", scratchy_subtile::superdsc_opspec::Scale::RedNonStick)],
                broadcast_batch: true,
            },
        ));
        // The epilogue operand is the one inserted BEFORE the real output (see
        // `attach_fused_epilogue`'s own doc) — for this 3-arg-input matmul (a,w,o) that is index 2.
        let epi_idx = 2;
        let view = op.args[epi_idx].view();
        // ⭐ ASSERTED BY DIM NAME, NOT BY POSITION. This test asserted `layout == ["mb","out","y"]` and
        // a scale array indexed against it; the emitter's order is `["mb","y","out"]` (the
        // on-hardware-proven batch-inner walk `batched_decode_walk_order.rs` pins), so the positional
        // form went stale the moment the walk work landed — and, because this whole test target failed
        // to COMPILE, said nothing about it for as long as it was wrong. The SUBJECT is which dims are
        // broadcast, which does not depend on their order, so it is now stated that way.
        let scale_of = |d: &str| {
            view.layout
                .iter()
                .position(|&l| l == d)
                .map(|i| view.scale[i])
                .unwrap_or_else(|| panic!("epilogue operand has no `{d}` dim: {:?}", view.layout))
        };
        assert_eq!(scale_of("mb"), Scale::RedNonStick, "mb must be broadcast");
        assert_eq!(
            scale_of("y"),
            Scale::RedNonStick,
            "the GQA-group batch axis must be broadcast too"
        );
        assert_eq!(scale_of("out"), Scale::Active, "out must stay Active");

        // And the FULL emit succeeds (would have panicked pre-fix if "y" were absent from this
        // shape's layout, or produced a silently-wrong scale_ if the marker were dropped).
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let sdsc = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
        let labeled = &sdsc.dscs_[0]["MatMul_0"].labeledDs_;
        let mask_lds = labeled
            .iter()
            .find(|l| l.dsName_ == format!("Tensor{epi_idx}"))
            .expect("mask labeledDs_ entry");
        // Wire `scale_` is parallel to the operand's own layout, so it is read the same way: -1 for a
        // broadcast dim, 1 for an active one. Two broadcast dims and one active, whatever the order.
        let wire = |d: &str| {
            view.layout
                .iter()
                .position(|&l| l == d)
                .map(|i| mask_lds.scale_[i])
                .expect("dim")
        };
        assert_eq!(wire("mb"), -1, "wire scale_ mb: {:?}", mask_lds.scale_);
        assert_eq!(wire("y"), -1, "wire scale_ y: {:?}", mask_lds.scale_);
        assert_eq!(wire("out"), 1, "wire scale_ out: {:?}", mask_lds.scale_);
    }

    #[test]
    fn emit_sdsc_matmul_fused_epilogue_broadcast_batch_is_noop_when_unbatched() {
        // The other half of the SAME fix: `broadcast_batch=true` on an UNBATCHED (batch==1) matmul
        // must be a genuine no-op, never a panic — `matmul_dims` omits "y" entirely at batch==1, and
        // `batch_dim_name()` reporting `None` there is exactly what lets `attach_fused_epilogue` skip
        // marking it instead of reaching for a dim that does not exist (the caller-side bug class
        // this mechanism replaces: attn.rs no longer computes "is this op batched" itself to decide
        // whether "y" is safe to name).
        let mut op = matmul_opspec(384, 384, 64, 1, "Tensor0", "Tensor1", "Tensor2").unwrap();
        assert_eq!(
            op.batch_dim_name(),
            None,
            "a batch=1 matmul must carry no y dim at all"
        );
        // broadcast_batch: true must NOT panic despite there being no "y" to mark.
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs::One(
            scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[("mb", scratchy_subtile::superdsc_opspec::Scale::RedNonStick)],
                broadcast_batch: true,
            },
        ));
        let epi_idx = 2;
        let view = op.args[epi_idx].view();
        assert_eq!(
            view.layout,
            ["mb", "out"],
            "unbatched output stays rank-2: {:?}",
            view.layout
        );
        assert_eq!(view.scale, [Scale::RedNonStick, Scale::Active]);
    }

    #[test]
    fn emit_sdsc_matmul_fused_epilogue_matches_golden_shape() {
        // Mirrors `ddc/ddl_templates/test/sdsc_bmm_lxopt.json`'s `MatMul_122`: computeOp_ is a
        // 2-element array — the matmul (unchanged inputLabeledDs, no mention of the epilogue operand),
        // then a second entry whose inputLabeledDs/outputLabeledDs both alias the MATMUL'S OWN output
        // by name (in place), with the epilogue's extra tensor as its second input.
        let mut op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2").unwrap();
        assert_eq!(
            op.time(),
            1,
            "test assumes an untiled matmul (attach_fused_epilogue's precondition)"
        );
        op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs::One(
            scratchy_subtile::superdsc_opspec::EpilogueSpec {
                operand_name: "mask".to_string(),
                offset_elems: 0,
                op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc::StridedAdd,
                broadcast_dims: &[],
                broadcast_batch: false,
            },
        ));
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let sdsc = emit_sdsc("MatMul_0", &op, &folds, None).unwrap();
        let ops = &sdsc.dscs_[0]["MatMul_0"].computeOp_;
        assert_eq!(
            ops.len(),
            2,
            "fused epilogue must add exactly one computeOp_ entry: {ops:?}"
        );

        // computeOp_[0]: the matmul itself, UNCHANGED — no trace of the mask operand.
        assert_eq!(ops[0].opFuncName, "batchmatmul");
        assert_eq!(ops[0].inputLabeledDs, vec!["Tensor0-idx0", "Tensor1-idx1"]);
        assert_eq!(ops[0].outputLabeledDs, vec!["Tensor3-idx3"]);

        // computeOp_[1]: the epilogue, reading+writing the MATMUL'S OWN output in place, plus the
        // mask as its second input — exactly the golden's `biasadd` shape.
        assert_eq!(ops[1].exUnit, "sfp");
        assert_eq!(ops[1].opFuncName, "stridedadd");
        assert_eq!(ops[1].inputLabeledDs, vec!["Tensor3-idx3", "Tensor2-idx2"]);
        assert_eq!(ops[1].outputLabeledDs, vec!["Tensor3-idx3"]);

        // The mask tensor still gets its OWN labeledDs_ entry (the on-card walk must address it), typed
        // OUTPUT to match `bmm.ddl`'s bias/bnA/bnB/resadd convention (same layout bucket as the real
        // output, not INPUT).
        let labeled = &sdsc.dscs_[0]["MatMul_0"].labeledDs_;
        assert_eq!(
            labeled.len(),
            4,
            "activation, kernel, mask, output: {labeled:?}"
        );
        let mask_lds = labeled
            .iter()
            .find(|l| l.dsName_ == "Tensor2")
            .expect("mask labeledDs_ entry");
        assert_eq!(mask_lds.dsType_, "OUTPUT");
    }

    #[test]
    fn sub_stick_matmul_is_rejected() {
        // N=65 is not a multiple of the 64-fp16 stick → builder Err (witness a).
        assert!(matmul_opspec(384, 65, 64, 1, "a", "w", "o").is_err());
    }

    #[test]
    fn prefill_m_gt_1_lm_head_folds_to_m1_decode_unchanged() {
        // The mq>1 (prefill) bundle CANNOT run the vocab-wide lm_head at m>1 (it time-tiles, and
        // per-row time-tiling is design-risk-4). It runs it at m=1 over the LAST prompt row instead,
        // which is what lets prefill produce the first generated token's logits itself. This guards
        // both halves of `lower_one_node`'s fold: the per-stick extraction copies, and the m=1
        // re-lowering. The m=1 DECODE bundle must stay a single bare matmul — no copies, no extra ops.
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        // hidden[m, H] @ W_lmhead[H, vocab] -> logits[m, vocab] (t2 = the result). Stick-aligned
        // H=128, vocab=256 so the m=1 path is a clean single matmul (no time-tile).
        let (h, vocab) = (128u32, 256u32);
        let build = |m: u32| {
            let tensors = vec![
                TensorShape { rows: m, cols: h }, // t0 = hidden (activation source)
                TensorShape {
                    rows: h,
                    cols: vocab,
                }, // t1 = lm_head weight source
                TensorShape {
                    rows: m,
                    cols: vocab,
                }, // t2 = logits (result)
            ];
            let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
                tensor: TensorId::from_index(t as usize),
                region: ts[t].whole(),
            };
            let node = SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    weight: GemmWeight::Dense,
                },
                inputs: vec![whole(0, &tensors), whole(1, &tensors)],
                output: whole(2, &tensors),
            };
            let ir: SubtileIR = SubtileIR {
                tensors,
                num_sources: 2,
                nodes: vec![node.clone()],
                result: TensorId::from_index(2),
                // Hand-authored fixture: there is no source op list to be the
                // provenance of, so the map is empty.
                op_output: Vec::new(),
            };
            (node, ir)
        };
        let lower = |m: u32| {
            let (node, ir) = build(m);
            let mut sym = 0i64;
            let mut fp8q = std::collections::HashSet::new();
            match lower_one_node(
                &node,
                &ir,
                ActiveCap::FULL,
                false,
                &mut sym,
                None,
                &mut fp8q,
            ) {
                NodeLowering::Ops(v) => v.into_iter().map(|e| e.op_name).collect::<Vec<_>>(),
                NodeLowering::Unhandled(e) => {
                    panic!("lm_head at m={m} unexpectedly Unhandled: {e}")
                }
                NodeLowering::HostRouted(_) => panic!("lm_head at m={m} unexpectedly HostRouted"),
            }
        };
        // m>1 (prefill): H/64 extraction copies, THEN the matmul re-lowered at m=1.
        let mq = 8u32;
        let names = lower(mq);
        let copies = (h / Fp16::ELEMS_PER_STICK) as usize;
        assert_eq!(
            names.len(),
            copies + 1,
            "prefill (m>1) lm_head must fold to {copies} extraction copies + 1 matmul, got {names:?}"
        );
        for (j, name) in names.iter().take(copies).enumerate() {
            assert_eq!(
                name,
                &format!("lmlast{j}_o2"),
                "copy {j} misnamed in {names:?}"
            );
        }
        assert_eq!(
            names[copies], "matmul_o2",
            "the folded tail must end in the lm_head matmul"
        );
        // m==1 (decode): the SAME node lowers to exactly one bare matmul — the fold never fires, so
        // the decode bundle is byte-identical to the pre-fold emitter.
        assert_eq!(lower(1), vec!["matmul_o2".to_string()]);
    }

    #[test]
    fn fp8_shared_activation_quantizes_once() {
        // granite decode emits q/k/v = gemm(normed, ·) — THREE arity-3 fp8 matmuls reading the SAME
        // activation (and gate/up = gemm(normed2, ·) — two more). The per-token activation quantize
        // (square→amax→scale→clamp→qfp8ch) is a PURE function of the activation, independent of the weight,
        // so it must be emitted ONCE and shared — not re-run per matmul. Lock that: two arity-3 fp8 matmuls
        // sharing t0 emit exactly ONE `fq_afp8_op` (qfp8ch) yet still TWO `fq_mm` (per-matmul matmulfp8).
        use scratchy_subtile::subtile_ir::{
            SubOp, SubtileIR, SubtileId, SubtileNode, TensorId, TensorRegion, TensorShape,
        };
        // ⛔ `n` WAS 64, WHICH IS SUB-STICK FOR fp8. An fp8 stick is 128 elems (fp16's is 64), and the
        // dxp scheduler rejects a sub-stick tile — a guard this crate enforces by construction, so the
        // lowering `Err`s before it can emit anything and this test asserted nothing about its actual
        // subject. That guard landed after the test was written, and the dead target hid it. The subject
        // — ONE shared activation quantize across two matmuls — does not depend on `n`, so `n` becomes a
        // legal fp8 width and the test measures what it is named for.
        let (k, n) = (128u32, 128u32);
        let tensors = vec![
            TensorShape { rows: 1, cols: k }, // t0 = activation (m=1 decode)
            TensorShape { rows: k, cols: n }, // t1 = W1 (fp8)
            TensorShape { rows: 1, cols: n }, // t2 = w_scale1
            TensorShape { rows: k, cols: n }, // t3 = W2 (fp8)
            TensorShape { rows: 1, cols: n }, // t4 = w_scale2
            TensorShape { rows: 1, cols: n }, // t5 = out1
            TensorShape { rows: 1, cols: n }, // t6 = out2
        ];
        let whole = |t: usize, ts: &[TensorShape]| TensorRegion {
            tensor: TensorId::from_index(t as usize),
            region: ts[t].whole(),
        };
        let nodes = vec![
            SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::MatmulTile {
                    weight: GemmWeight::Fp8Dynamic,
                },
                inputs: vec![whole(0, &tensors), whole(1, &tensors), whole(2, &tensors)],
                output: whole(5, &tensors),
            },
            SubtileNode {
                id: SubtileId::from_index(1),
                op: SubOp::MatmulTile {
                    weight: GemmWeight::Fp8Dynamic,
                },
                // SAME activation t0, DIFFERENT weight/scale/out → the quantize of t0 must be reused.
                inputs: vec![whole(0, &tensors), whole(3, &tensors), whole(4, &tensors)],
                output: whole(6, &tensors),
            },
        ];
        let ir: SubtileIR = SubtileIR {
            tensors,
            num_sources: 5, // t0 activation + t1..t4 (weights + w_scales)
            nodes,
            result: TensorId::from_index(6),
            // Hand-authored fixture: there is no source op list to be the
            // provenance of, so the map is empty.
            op_output: Vec::new(),
        };
        let weight_ids: std::collections::HashSet<u32> = [1u32, 3u32].into_iter().collect();
        let (ops, _layout) = lower_graph_to_superdsc(&ir, &weight_ids, ActiveCap::FULL, false)
            .expect("two-fp8-matmul lowering");
        let quantizes = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_afp8_op"))
            .count();
        assert_eq!(
            quantizes, 1,
            "two matmuls sharing an activation must quantize it ONCE (shared), got {quantizes}"
        );
        let matmuls = ops.iter().filter(|o| o.op_name.ends_with("fq_mm")).count();
        assert_eq!(
            matmuls, 2,
            "each fp8 matmul still emits its OWN matmulfp8 (weight differs), got {matmuls}"
        );
        // The FIRST chain op is likewise shared: one, not two. It is `abs` (`fq_absx_op`), not the
        // `square` (`fq_sq_op`) this test named — the quantize chain became abs→max, and no op by the
        // old name has existed for as long as this target failed to compile, so the assert was looking
        // for zero of something and would have passed only by finding nothing.
        let first_chain_op = ops
            .iter()
            .filter(|o| o.op_name.ends_with("fq_absx_op"))
            .count();
        assert_eq!(
            first_chain_op, 1,
            "the activation |x| must be shared too, got {first_chain_op}"
        );
    }

    #[test]
    fn matmul_cost_split_fills_cores() {
        // sdsc_bmm_autoBuffer.json: M=384, N=384, K=64, batch=16 → must use 32 cores.
        let s = matmul_cost_split(16, 384, 384, 64, 32);
        assert_eq!(s.cores(), 32, "matmul split must fill 32 cores: {s:?}");
        // and the iteration space maps M→mb, N→out, K→in, batch→x.
        let it = matmul_iter_space(384, 384, 64, 16);
        assert_eq!((it.mb_, it.out_, it.in_, it.x_), (384, 384, 64, 16));
    }

    #[test]
    fn assemble_matmul_serializes_and_fills_cores() {
        // bmm 384×384×64 batch16 fits LX (576 KiB < 1.6 MiB) → time=1 EmittedOp.
        let emitted = assemble_matmul(
            "MatMul_0",
            384,
            384,
            64,
            16,
            &rb("act", 384, 64),
            &Stk::<KernelTag>::kernel(64, 384, "wt"),
            &rb("out", 384, 384),
            None,
        );
        assert_eq!(emitted.time, 1, "bmm must NOT time-tile (fits LX)");
        let op = &emitted.op;
        // numWkSlices product = 32 cores.
        let prod: u32 = op.numWkSlicesPerDim_.values().product();
        assert_eq!(
            prod, 32,
            "matmul must use 32 cores: {:?}",
            op.numWkSlicesPerDim_
        );
        // serializes to JSON with the mandatory trailing-underscore keys.
        let j = serde_json::to_string(op).expect("SuperDSC serializes");
        assert!(j.contains("\"coreFoldProp_\""), "coreFoldProp_ present");
        assert!(j.contains("\"numWkSlicesPerDim_\""));
        assert!(j.contains("\"batchmatmul\""), "batch>1 → batchmatmul");
        assert!(j.contains("\"SEN169_FP16\""));
        // time=1 op is NOT symbolic — no isStartAddrSymbolic_, addresses concrete.
        assert!(
            !j.contains("isStartAddrSymbolic_"),
            "time=1 op stays concrete-addr"
        );
        // per-core stage dims = full / split.
        let dsc = &op.dscs_[0]["MatMul_0"];
        assert_eq!(dsc.numCoresUsed_, 32);
        assert_eq!(dsc.computeOp_[0].exUnit, "pt");
    }

    #[test]
    fn bundle_mlir_emits_execute() {
        // The flat (all-time=1) bundle.mlir is byte-identical to the historical form.
        let b = bundle_mlir(&["sdsc_0.json".to_string()]);
        assert!(b.contains("func.func @sdsc_bundle()"));
        assert!(b.contains("sdscbundle.sdsc_execute () {sdsc_filename=\"sdsc_0.json\"}"));
        // A time=1 EmittedOp routes through emit_bundle_mlir → the SAME flat body.
        let emitted = assemble_matmul(
            "MatMul_0",
            384,
            384,
            64,
            16,
            &rb("act", 384, 64),
            &Stk::<KernelTag>::kernel(64, 384, "wt"),
            &rb("out", 384, 384),
            None,
        );
        let via_emitted = emit_bundle_mlir(&[emitted]);
        assert_eq!(
            via_emitted,
            bundle_mlir(&["sdsc_0.json".to_string()]),
            "an all-time=1 bundle.mlir must be byte-identical to the historical flat form"
        );
        assert!(!via_emitted.contains("scf.for"));
    }

    #[test]
    fn tiled_matmul_concrete_unrolls() {
        // 64×16384×2048 batch1 overflows LX (2.42 MiB > 1.68 MiB) → time-tiled.
        // The A term is only 256 KiB so out-tiling brings it under LX (a wide-K
        // shape would Err instead); the cost split fills 32 cores via out×32.
        let emitted = assemble_matmul(
            "matmul_o7",
            64,
            16384,
            2048,
            1,
            &rb("a", 64, 2048),
            &Stk::<KernelTag>::kernel(2048, 16384, "w"),
            &rb("o", 64, 16384),
            None,
        );
        let n = emitted.time;
        assert!(n > 1, "this matmul must time-tile, got time={n}");
        // (ii) the divided per-time `out` shows up in N_ / ss_ (per-core out_per_time).
        let dsc = &emitted.op.dscs_[0]["matmul_o7"];
        let split_out = emitted
            .op
            .numWkSlicesPerDim_
            .get("out")
            .copied()
            .unwrap_or(1);
        let per_core_out_per_time = (dsc.N_.out_ as u32) / split_out;
        assert_eq!(
            per_core_out_per_time % 64,
            0,
            "per-time per-core out must be 64-aligned"
        );
        // (iv) CONCRETE-UNROLL: bundle.mlir has N flat executes, NO scf.for / symbols;
        //      the SdscOp JSON is concrete (NOT isStartAddrSymbolic_), and trips differ.
        let mlir = emit_bundle_mlir(&[emitted.shallow_copy()]);
        assert!(
            !mlir.contains("scf.for"),
            "concrete-unroll has no scf.for:\n{mlir}"
        );
        assert!(
            !mlir.contains("affine.apply"),
            "concrete-unroll has no affine.apply"
        );
        assert!(
            !mlir.contains("symbol_ids"),
            "concrete-unroll has no symbol_ids"
        );
        assert_eq!(
            mlir.matches("sdscbundle.sdsc_execute").count() as u32,
            n,
            "one flat execute per trip"
        );
        let trips = concrete_trips(&emitted);
        assert_eq!(trips.len() as u32, n);
        let j0 = serde_json::to_string(&trips[0]).unwrap();
        assert!(
            !j0.contains("isStartAddrSymbolic_"),
            "trip json is concrete, not symbolic"
        );
        assert!(
            j0.contains("{\"factor_\":1,\"label_\":\"time\"}"),
            "sdscFoldProps_ time stays 1"
        );
        assert_ne!(
            j0,
            serde_json::to_string(&trips[1]).unwrap(),
            "trips differ (bumped addrs)"
        );
    }

    #[test]
    fn stick_count_ceils() {
        assert_eq!(stick_count(64), 1);
        assert_eq!(stick_count(65), 2);
        assert_eq!(stick_count(384), 6);
    }

    // ── RUNG-2 LOCK: the matmul work-division is df-aware. An fp8 (128-lane) matmul
    // MUST split its N/K by the 128-stick basis, NOT fp16's 64 — a 64-granular split
    // hands a core a sub-128 slice, the exact `L3DlOpsScheduler:1070 multiple-of-stick`
    // DtException the fp8 bake used to hit. This guard is fail-first: reverting
    // `stick_basis`/`matmul_split_map` to a hardcoded 64 makes it RED. ──
    #[test]
    fn matmul_split_is_df_aware_fp8_128() {
        // N=512: fp16 ⇒ 512/64 = 8 sticks (can split ≤8 ways); fp8 ⇒ 512/128 = 4 sticks.
        // The fp8 split must therefore be COARSER (≤4), never the fp16 8. The `::<Fp8>` type
        // param — not a runtime flag — is what sources the 128 basis onto the dims.
        let dims_f16 = matmul_dims::<Fp16>(
            1,
            &StickExtent::<Fp16>::new(512).unwrap(),
            &StickExtent::<Fp16>::new(256).unwrap(),
            1,
        );
        let dims_f8 = matmul_dims::<Fp8>(
            1,
            &StickExtent::<Fp8>::new(512).unwrap(),
            &StickExtent::<Fp8>::new(256).unwrap(),
            1,
        );
        let s_f16 = matmul_split_map(&dims_f16, MAX_CORES);
        let s_f8 = matmul_split_map(&dims_f8, MAX_CORES);
        let out16 = s_f16.get("out").copied().unwrap_or(1);
        let out8 = s_f8.get("out").copied().unwrap_or(1);
        assert!(
            out16 <= 8,
            "fp16 out split bounded by 8 sticks, got {out16}"
        );
        assert!(
            out8 <= 4,
            "fp8 out split MUST be bounded by 4 (128-)sticks, got {out8}"
        );
        // The fp8 per-core `out` extent is a whole 128-stick multiple (never sub-stick).
        assert_eq!(
            512u32 / out8.max(1) % 128,
            0,
            "fp8 per-core out must be 128-aligned"
        );
    }

    #[test]
    fn workplan_rejects_substick_fp8_split() {
        // A hand-crafted over-split of an fp8 stick dim (2 sticks, split 4 ways) must be a
        // typed `Err` at emit (WorkPlan::divide stick clause, stick_basis=128), NOT an
        // on-card DtException. The SAME split of a fp16 dim (more 64-sticks) is legal.
        let over = |name: &'static str, _: u32| {
            let mut m = std::collections::BTreeMap::new();
            m.insert(name, 4u32);
            m
        };
        let fp8_dim = vec![ItDim {
            name: "out",
            size: 256,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp8,
        }];
        let err = WorkPlan::divide(&fp8_dim, MaxCores::<MAX_CORES>, |d, c| over(d[0].name, c));
        assert!(
            err.is_err(),
            "fp8 256 (=2×128 sticks) split 4 ways must be Err (sub-stick), got {err:?}"
        );
        // fp16 256 = 4×64 sticks ⇒ split 4 ways is exactly 1 stick/core ⇒ Ok.
        let fp16_dim = vec![ItDim {
            name: "out",
            size: 256,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        }];
        let ok = WorkPlan::divide(&fp16_dim, MaxCores::<MAX_CORES>, |d, c| over(d[0].name, c));
        assert!(
            ok.is_ok(),
            "fp16 256 (=4×64 sticks) split 4 ways is 1 stick/core, must be Ok, got {ok:?}"
        );
    }

    #[test]
    fn matmul_split_fp16_byte_identical_to_stick_count() {
        // The dense (fp16) path must be UNCHANGED by the df-aware refactor: for any N the
        // `out` split equals the pre-refactor `stick_count`(÷64)-based split. (Inert-at-fp16
        // is the rung invariant — only fp8 emission changes.)
        for &n in &[64u32, 128, 256, 384, 512, 2048, 5504] {
            let dims = matmul_dims::<Fp16>(
                1,
                &StickExtent::<Fp16>::new(n).unwrap(),
                &StickExtent::<Fp16>::new(64).unwrap(),
                1,
            );
            let split = matmul_split_map(&dims, MAX_CORES)
                .get("out")
                .copied()
                .unwrap_or(1);
            let expected = core_split(stick_count(n), MAX_CORES);
            assert_eq!(
                split, expected,
                "fp16 out split for N={n} must match stick_count-based split"
            );
        }
    }

    // SEN169_FP16 (1-6-9, bias 31) encode — the device's NATIVE fp16, NOT IEEE (1-5-10). Feeding IEEE
    // bits is silently mis-read by the device (1.0→IEEE 0x3C00→SEN169 0.5; 1/576→SEN169 ≈1e-6) — the
    // ~14× rmsnorm scale bug. Anchored to the SFP const table's ground truth (plus1=0x3E00,
    // minus1=0xBE00) + the exp-field/bias/packing points that were wrong in that bug. Concrete
    // machine-check (the float encoder is the ALU leaf; Kani/CBMC over-approximates its libm log2).
    #[test]
    fn sen169_encode_anchors() {
        assert_eq!(super::sen169_bits(1.0), 0x3E00); // 2^0 ⇒ exp field 31 (bias-31), mantissa 0
        assert_eq!(super::sen169_bits(-1.0), 0xBE00); // sign bit + plus1
        assert_eq!(super::sen169_bits(2.0), 0x4000); // 2^1 ⇒ exp field 32
        assert_eq!(super::sen169_bits(0.5), 0x3C00); // 2^-1 ⇒ exp field 30
        assert_eq!(super::sen169_bits(0.0), 0); // zero
        assert_ne!(super::sen169_bits(1.0), 0x3C00); // NOT IEEE-f16 1.0 (0x3C00) — the mismatch that WAS the bug
    }
}
