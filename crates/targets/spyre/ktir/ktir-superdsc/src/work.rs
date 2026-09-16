// SPDX-License-Identifier: Apache-2.0
//! THE WORK DIVISION — how one op's iteration space is cut across the card's cores.
//!
//! ⭐ ONE MODULE, WHICH IS THE POINT. These items sat at eight scattered sites inside the
//! 10,284-line `lower_subtile_tape_to_superdsc.rs` with 85 in-file call sites, which is exactly why
//! an earlier pass deferred them: cutting eight holes in a file that was itself mid-move creates
//! merge surface for code about to move again. They are pure arithmetic over types this crate
//! already owns — [`ItDim`], [`WorkPlan`], [`MAX_CORES`] — so they are one module here and nothing
//! else changed.
//!
//! ⛔ THE SPYRE MODEL, NOT THE KTIR EMITTER'S. 256 MB/core span and a 64-element fp16 stick; the
//! `pick_k`/`n_block` formulas in the KTIR emitter are H100's and are the wrong hardware.

use crate::superdsc_opspec::{ItDim, MAX_CORES, WorkPlan};
use std::collections::BTreeMap;

pub const CORELETS_PER_CORE: u32 = 2;
pub const STICK_BYTES: u32 = 128;
pub const FP16_ELEMS_PER_STICK: u32 = 64; // 128 B / 2 B
pub const MAX_SPAN_BYTES: u64 = 256 * 1024 * 1024;

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
) -> BTreeMap<String, BTreeMap<&'static str, crate::superdsc_opspec::SliceIndex>> {
    use crate::superdsc_opspec::SliceIndex;
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
pub fn matmul_split_plan(
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
