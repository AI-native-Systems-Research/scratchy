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

/// ⭐⭐⭐ THE ONE-ROW MATMUL'S CORE FILL — `(out, in)` cores for a matmul whose `mb` is one row, both in
/// WHOLE STICKS of their own axis: the most cores that fit `max_cores`, and among those the widest
/// `out` (so the reduction fan-in is the smallest that fills the machine).
///
/// ⛔ WHY AN OUT-ONLY SPLIT IS NOT ENOUGH. At one row the matmul is a pure weight stream, and the card
/// streams at its ~149 GB/s HBM read ceiling ONLY when all 32 cores read. MEASURED on the card, one
/// granite-3.1-8b fp16 projection per launch, device exec cycles off the response blocks:
///
/// | weight `[k, n]`        | out-only split | GB/s  | this fill      | GB/s  |
/// |------------------------|----------------|-------|----------------|-------|
/// | gate/up `[4096,12800]` | out25 (25 c)   | 126.0 | in4·out8 (32)  | 147.2 |
/// | k/v `[4096,1024]`      | out16 (16 c)   |  89.2 | in2·out16 (32) | 145.9 |
/// | down `[12800,4096]`    | out32          | 149.0 | out32 (same)   | 149.0 |
///
/// and the rate does not scale linearly with the cores reading: 8 and 16 cores both plateau near
/// 88 GB/s, and only all 32 reach the ~149 GB/s ceiling (`T = 4.7 µs + bytes / 148.1 GB/s` over
/// 18 32-core streams). So the fill aims at all 32. 200 sticks (12800) has no divisor in 26..=32 and
/// 16 sticks (1024) none above 16, which is the whole of why those two are slow: the cost is the
/// stick count's arithmetic, not the shape.
///
/// ⭐ THE PARTITION STAYS EXACT. `out` divides the output stick count and `in` divides the reduction
/// stick count, so each axis is cut into equal whole-stick slices that tile it with no remainder; the
/// output is DISJOINT over `out` and SHARED over `in`, which is the reduction split `emit_sdsc_tiled`
/// already admits (its disjoint-write guard is relaxed exactly for a split `in` the output does not
/// carry), dxp PSUM-accumulating the partials. Kani: `kani_proofs::the_one_row_fill_*`.
///
/// When `out` alone already fills `max_cores` (or nothing larger exists) this returns `(out, 1)` —
/// the out-only split, byte-identical to what one row emitted before the fill.
pub fn one_row_core_fill(out_sticks: u32, in_sticks: u32, max_cores: u32) -> (u32, u32) {
    let mut best = (core_split(out_sticks, max_cores), 1u32);
    let mut o = 1u32;
    while o <= out_sticks.min(max_cores) {
        if out_sticks.is_multiple_of(o) {
            let mut i = 1u32;
            while i <= in_sticks.min(max_cores / o) {
                if in_sticks.is_multiple_of(i) && (o * i, o) > (best.0 * best.1, best.0) {
                    best = (o, i);
                }
                i += 1;
            }
        }
        o += 1;
    }
    best
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
    //
    // ⭐ AND ONLY WHERE THE UNPADDED SPLIT LEAVES MOST OF THE MACHINE IDLE. The pad is weight bytes, and
    // at decode every weight byte is streamed every step: it buys cores, and cores are only worth bytes
    // while the gemm strands on a fraction of the machine. The lm_head's prime 769 sticks split onto ONE
    // core (8 after the floor pad) — 4% more bytes for 4× the cores. granite-3.1-8b's gate/up (12800 =
    // 200 sticks) already split onto 25 of 32: the pad was +12% of the model's largest weights, streamed
    // every token, to buy the last 7 cores. Below half the machine the pad pays; at or above it the
    // unpadded split stands.
    //
    // MEASURED, and the bytes were the smaller half of it: a keyed GROUP_TIME split of one 10k-context
    // 8b decode step put the whole 49 ms the pad cost in the MLP group. 224 sticks on 32 cores is 7
    // sticks a core — a prime — so each of gate and up was time-tiled into 7 one-stick trips, where
    // 200 sticks on 25 cores is 8 a core and runs 4 two-stick trips.
    let full = sticks.next_multiple_of(MAX_CORES);
    if cur * 2 < MAX_CORES && (full - sticks) * 8 <= sticks {
        return full * FP16_ELEMS_PER_STICK;
    }
    if cur >= UTIL_FLOOR_CORES {
        n64
    } else {
        sticks.next_multiple_of(8) * FP16_ELEMS_PER_STICK
    }
}

/// The contraction-K width of an fp8 W8A8 matmul's ACTIVATION and WEIGHT, rounded to a whole
/// 128-elem SEN143_FP8 stick. ⭐ THE K-AXIS TWIN OF [`DeviceWidth`]'s N law — and the FIRST fp8
/// model with a non-legal K is what forced it: gemma-4-26b's dense GeGLU DOWN gemm contracts
/// `k = 2112 = 16.5 sticks`, which `qfp8ch`'s convert guard refuses (`assert_df_stick_multiple`:
/// a sub-stick tile is rejected by the dxp scheduler, L3DlOpsScheduler:1040). Every earlier fp8
/// model was legal by luck (granite 2048/11008, g8b 4096/12800, g12b 3840/15360 — all multiples
/// of 128).
///
/// THE CONTRACT, SAME AS `DeviceWidth`'s: the fp8 quantize chain (`cl`/`afp8`), the packed
/// RetileDescriptor (`[n/64, k/2, 64, 2]`), the layout's weight `nbytes`, and the worker's
/// staged buffer ALL take their K from this one function, so the shared contraction k cannot
/// drift between activation and weight. A no-op for every 128-aligned width — granite and the
/// 12b are byte-identical under it.
///
/// ⛔ THE PAD LANES ARE ZERO BY EMISSION, NOT BY SEGMENT INIT. The chain that reaches the
/// contraction writes `cl`'s `[m, k]` logical window and the pad window is zero-FILLED by the
/// chain's own emission (a `sub(448,448)` broadcast) — see `matmul_fp8_descriptors`. Segments
/// zero-init once at prepare, but intermediates RE-USE bytes across steps: a stale pad lane
/// (last step's activation, or a NaN from a prior consumer) reaching the clamp would pass
/// through `maximum/minimum` UNCHANGED (NaN-propagation), then `0 · NaN = NaN` in the
/// contraction — the zero-fill is a correctness requirement, not belt-and-braces.
pub fn fp8_k_pad(k: u32) -> u32 {
    k.next_multiple_of(crate::superdsc_opspec::Df::Fp8.elems_per_stick())
}

/// ⭐ THE PACKED fp8 KERNEL'S OWN STICK GRANULARITY, PER AXIS. A packed W8A8 weight does NOT stage
/// as a flat 128-elem-squared fp8 tensor: its RetileDescriptor is `[N/64, K/2, 64, 2]` — one
/// 128-byte stick carries **64 N-columns × 2 K-bytes** (`gen_fp8_kernel_out_fold`'s 8×8 N-subtile,
/// `gen_fp8_kernel_in_fold`'s 2-pack, `fp8_kernel_stage_off`'s `(n/64)·(K·64) + (n%64) + (i/2)·128
/// + (i%2)·64` address law). So the kernel's OUT axis is legal at any 64-multiple and its K axis
/// at any even count — NOT the flat [`StickExtent::<Fp8>`] 128-law.
///
/// ⛔ THIS IS THE LAW THE 26b N-WALL EXPOSED. gemma-4-26b's dense GeGLU GATE/UP projection has
/// `n = 2112 = 33 × 64` — LEGAL to the retile's own `N%64==0` guard and to every addressing
/// consumer, but refused by `per_core_addr`'s validation, which modeled every fp8 operand as a
/// flat 128-stick tensor. Inert on every earlier fp8 model (granite/g8b/g12b n all 128-aligned
/// by luck). This const pair is the ONE statement of the real law; the retile guard and the
/// per-core validation both take their granularity from here so they cannot diverge again.
pub const FP8_PACKED_KERNEL_OUT_GRAN: u32 = 64;
/// The packed kernel's K (in) axis granularity — 2-packed fp8 per fp16-width slot.
pub const FP8_PACKED_KERNEL_IN_GRAN: u32 = 2;

// ⛔ THE EXPERT SLAB CUT IS GONE — `expert_slab_chunks`/`expert_slab_subpages` (the LX-fit
// chunk and sub-page sizing of the MoE weight gather) existed only for the hand-written
// per-op MoE card lowering, which is deleted: the MoE lowers only from `kernels/moe.py`
// through KTIR. The chunk-major bank law they stated lives on where the token-plane
// descriptor states it (`superdsc_opspec`'s token-plane form doc).

/// ⭐ THE UTIL-FLOOR CORE COUNT — the minimum cores a FLOP-heavy matmul (`macs ≥ 2^20`) may be
/// split onto. One constant for every reader: `DeviceWidth::for_matmul`'s windowed drop test,
/// guard #11's refusal in the emitter, and `out_width_the_weight_holds`' floor test in the
/// bridge. Previously three hard-coded `8`s that had to agree by hand (issue 201 item 5).
pub const UTIL_FLOOR_CORES: u32 = 8;

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
    /// The device width of a WHOLE-FUNCTION matmul output — the ONE decision both the
    /// intermediate's reservation and the emitter's `n_dev` take (issue 201 item 4: two sites
    /// computed the same fact differently and over-reserved 8× on granite tiled_k).
    ///
    /// `windowed = true` is the whole-function door's minted-intermediate arm: the padded weight
    /// columns would be read from the CALLER'S parameter windows (`for_output`'s staged-buffer
    /// contract does not exist there), so the util-floor bump is dropped when the LOGICAL width
    /// already meets the floor. ⛔ ONLY THE FLOOR BUP IS DROPPED — the rounding to a whole
    /// 64-stick STAYS (issue 201 item 5): an unaligned `n` with a ≥8-core split still emits at
    /// `n.next_multiple_of(64)`, so it cannot fall into the per-core stick guard that a raw
    /// `n_dev = n` would hit. `UTIL_FLOOR_CORES` is the floor both the drop test and guard #11
    /// use — named once, not three hard-coded `8`s.
    pub fn for_matmul(m: u32, logical_n: u32, k: u32, windowed: bool) -> DeviceWidth {
        if windowed {
            let n64 = logical_n.next_multiple_of(FP16_ELEMS_PER_STICK);
            if n64 > logical_n && CoreSplit::plan(m, logical_n).ncores() >= UTIL_FLOOR_CORES {
                // The logical width already meets the util floor the bump was buying — and its
                // stick rounding is kept by returning `n64`, not `logical_n`.
                return DeviceWidth(n64);
            }
        }
        DeviceWidth::for_output(m, logical_n, k)
    }

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

/// ⭐ A REDUCE's REDUCED STICK AXIS ACROSS THE CORES, every other dim whole — the largest divisor of
/// the reduced axis's stick count within the core budget, so every core's run is whole sticks. Reached
/// only through `reduce::ReduceDivision::ReducedAxis`, whose op's template combines the per-core
/// partials. A reduction with no reduced stick axis takes no split.
pub fn distribute_cores_reduced_axis(
    dims: &[ItDim],
    max_cores: u32,
) -> BTreeMap<&'static str, u32> {
    let mut splits = BTreeMap::new();
    if let Some(d) = dims.iter().find(|d| d.is_reduction && d.is_stick) {
        let sticks = d.size.div_ceil(d.df.elems_per_stick()).max(1);
        let split = core_split(sticks, max_cores);
        if split > 1 {
            splits.insert(d.name, split);
        }
    }
    splits
}

/// ⭐⭐⭐ HOW AN ELEMENTWISE OP's CORES ARE SPENT — rows first or sticks first, decided by the op's operand
/// layout and nothing else. See [`EwDivision::of`].
///
/// [`distribute_cores`] spends its cores on `mb` first, and for a FLAT row-major operand that is a
/// correctness law (its own note: an out-split core mis-walks the rows, measured `nz=3968`). A STICK-MAJOR
/// rank-2 operand is the other way round: its bytes are `[out/64][mb][64]`, so a core holding ALL rows of
/// a run of sticks reads one CONTIGUOUS block, while a core holding a run of rows reads `cols/64` strided
/// one-row bursts. MEASURED in isolation (#549's sweep, the real `mulsilu_o461` descriptor and placement,
/// one CB, 32 cores): a `[256, 8192]` mul ran 249.8 µs at 50.4 GB/s rows-first and 121.8 µs at 103.3 GB/s
/// sticks-first, flat across width at the copy band; `[256, 2048]` 50.0 → 30.3 µs. `out:8 × mb:4` lands on
/// the `out:32` line, so it is the stick axis taking its share first that matters, not `mb` going unsplit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EwDivision {
    /// THE LAW: whichever of the three below the op's operands call for ([`Self::resolve`]). Every
    /// elementwise door passes this, except for the ops below that state their division outright.
    Law,
    /// [`distribute_cores`]: `mb` first, then the other output dims largest-first. Stated outright only by
    /// an op whose own addressing is keyed on `mb` being the split axis (the gathered-row leg, whose index
    /// enumerates `mb` in whole entries).
    RowsFirst,
    /// [`distribute_cores_sticks_first`]: the stick dim's whole-stick split first, then `mb`. Stated
    /// outright by a decode batch chunk's `esub` and `exp`. Their shared core-local `cesub` is sized under
    /// it, so both ops keep it (`attn`'s `ChunkBufs::sm_division_of`).
    SticksFirst,
    /// ⭐⭐⭐ [`distribute_cores_sprays_first`]: the LEADING `y` (the spray axis) first, then the ordinary
    /// law over the cores it leaves. The ROW-SPLIT (y-major) mode's own — its tensors present
    /// `["y", "mb", "out"]`, so `y` is the outermost dim of the flat layout and a core's slab is
    /// CONTIGUOUS exactly when the split is over `y`. Stated by the y-major emission itself
    /// ([`crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile`]'s row-split arm),
    /// which refuses every other division for that mode by name — see the splitter's own doc for the
    /// card measurement that made this a law rather than a preference.
    SpraysFirst,
}

impl EwDivision {
    /// The concrete division. [`Self::Law`] resolves on the operands: `stick_major` is the builder's own
    /// rank-2 stick-major presentation decision — the SAME value that picks the operands' layout, so the
    /// layout and the division cannot disagree — `core_local` is whether any operand is a core-local
    /// LX pin (such a tensor's division is its PRODUCER's, held by `emit_sdsc_tiled`'s read law and sized
    /// by [`crate::emit::pointwise_untiled_slice_bytes`], so an op touching one under the law keeps the
    /// rows-first division those were laid out under; a pin laid out under a stated division is sized
    /// under that division instead, [`crate::emit::pointwise_untiled_slice_bytes_divided`], and its ops
    /// state it), and `sprays` is the row-split (y-major) presentation decision, the same kind of fact:
    /// the op's tensors walk `["y", "mb", "out"]`, and only a split over the leading `y` gives a core a
    /// contiguous slab in that layout. The concrete divisions resolve to themselves.
    ///
    /// ⭐ `sprays` outranks `stick_major` by construction, not by choice: the row-split frame law holds
    /// the op to ONE stick of `out` while the stick-major presentation requires wider than one, so the
    /// two facts are disjoint and their order in the match cannot matter.
    pub const fn resolve(self, stick_major: bool, core_local: bool, sprays: bool) -> EwDivision {
        match self {
            EwDivision::Law if sprays => EwDivision::SpraysFirst,
            EwDivision::Law if stick_major && !core_local => EwDivision::SticksFirst,
            EwDivision::Law => EwDivision::RowsFirst,
            concrete => concrete,
        }
    }

    /// The splitter this division names, in the shape [`WorkPlan::divide`] takes. [`Self::Law`] is not a
    /// splitter until it is resolved, so it names the rows-first one — the law's answer when there is
    /// nothing to resolve it on — and every door resolves before asking.
    pub fn splitter(self) -> fn(&[ItDim], u32) -> BTreeMap<&'static str, u32> {
        match self {
            EwDivision::Law | EwDivision::RowsFirst => distribute_cores,
            EwDivision::SticksFirst => distribute_cores_sticks_first,
            EwDivision::SpraysFirst => distribute_cores_sprays_first,
        }
    }
}

/// ⭐⭐⭐ A GATHERED OP'S WORK DIVISION — the ENTRY axis split over WHOLE ENTRIES and nothing
/// else, minted by exactly one constructor.
///
/// ⛔⛔ ONLY THE ENTRY AXIS (`mb`) MAY CARRY A SPLIT, AND THIS IS THE ONE NUMBER A GATHER'S
/// CORRECTNESS HANGS ON. dxp derives `skip_addr` as `page × the per-core extents of the
/// UNPINNED dims` (`getBufferCapacityForNodePerDim`, reproduced by
/// `zz_skip_addr_computed_from_our_own_emission.rs`), so a split on any other axis DIVIDES what
/// one index entry covers: every index step then lands a fraction of a row short, from a clean
/// bake. MEASURED through the gathered pointwise door (the prefill attention kernel's leg t6):
/// 8 entries on 32 cores left 24 cores for `distribute_cores` to fill, and it split `out`
/// 4 ways — fluent text assembled from two different rows. dbo also reduces the value tensor's
/// per-core starts to a single `min` before deriving any address (`computeGatherMetadata`), so
/// a core owning a COLUMN SLICE of an entry loses that slice and reads the entry from its
/// start.
///
/// ⭐ AND A GATHER LAWFULLY RUNS ON `min(entries, cores)` CORES — fewer cores than the machine
/// is the CARD-PROVEN shape of both existing gathered doors (`gather_copy_cores`' KV copy and
/// the gathered matmul arm's re-division), and the vendor's own one-core gather
/// (`dxp/test/test_gather_1core`) is the `entries == 1` case. The cores the entries cannot
/// take are LEFT IDLE, never filled by splitting another axis — that filler split is the
/// wrong-byte shape above, and it is the only way this law can be broken, so the mint refuses
/// nothing about the core count. The split is [`core_split`]'s largest divisor of `entries`
/// within the budget — exactly `min(entries, cores)` at every `entries ≤ cores` — so every
/// core owns one or more WHOLE entries whatever the count. (Ruling 10-20: `entries ≥ cores`
/// was this door's fill POLICY, not a byte law; the producer's batch-width move stays the
/// perf move, but a short gather is no longer unconstructible.)
///
/// This is the gathered POINTWISE leg's door (the attention legs). The KV copy door
/// (`gather_copy_cores`) states the mb-only half of this law privately — its entries are
/// `nkvh·nb·mq` (≥ 8) and a copy may lawfully run on the cores its entries can take; the
/// gathered matmul arm states it inline over its own re-division. Both cite the same
/// `skip_addr` derivation; this type is where the law lives for the door that hit it.
pub struct GatherSplit {
    splits: BTreeMap<&'static str, u32>,
}

impl GatherSplit {
    /// THE ONLY CONSTRUCTOR — the cores distributed over the gather's ENTRIES. `entry_page` is
    /// the `mb` positions ONE index entry covers; the division is over entries, so no core owns
    /// a fraction of one. The mint runs on `min(entries, max_cores)` cores (the largest divisor
    /// of `entries` within the budget — the same number at every `entries ≤ max_cores`) and
    /// splits NOTHING but `mb`. `Err` names the law that failed: a plan with no `mb` dim for
    /// the entries to enumerate, or an `mb` that does not divide into whole entries.
    pub fn over_entries(dims: &[ItDim], max_cores: u32, entry_page: u32) -> Result<Self, String> {
        let Some(mb) = dims
            .iter()
            .find(|d| d.name == "mb" && !d.is_reduction && !d.is_stick)
        else {
            return Err(
                "the op's work plan has no `mb` dim for the entries to enumerate — a gathered \
                 op's core split is the entry axis alone, and there is none"
                    .to_string(),
            );
        };
        if entry_page == 0 || mb.size % entry_page != 0 {
            return Err(format!(
                "the gather pages `mb` by {entry_page} position(s) and the axis is {} \
                 positions — an entry would cover a fraction of the axis, and a partial entry \
                 is an address",
                mb.size
            ));
        }
        let entries = mb.size / entry_page;
        let mut splits = BTreeMap::new();
        let split = core_split(entries, max_cores);
        if split > 1 {
            splits.insert(mb.name, split);
        }
        Ok(GatherSplit { splits })
    }

    /// The minted split — `mb` alone. The tiler consumes this verbatim; there is no door that
    /// hands a gathered op any other division.
    pub fn splits(&self) -> &BTreeMap<&'static str, u32> {
        &self.splits
    }
}

/// How a pointwise op's cores are divided: the elementwise law ([`EwDivision`]), or a gathered
/// op's own [`GatherSplit`]. The two are mutually exclusive BY TYPE — a gathered op's division
/// is not resolved from its operand layout (the elementwise law's input) but minted from its
/// entries, because only the entry axis may carry a split (see [`GatherSplit`]'s law).
pub enum CoreDivision {
    Ew(EwDivision),
    Gathered(GatherSplit),
}

#[cfg(test)]
mod gather_split_tests {
    use super::{CoreDivision, GatherSplit, ItDim};
    use crate::superdsc_opspec::Df;
    use std::collections::BTreeMap;

    /// A tile's `[mb, out, y]` dims — the spelling every pointwise gather states.
    fn dims(mb: u32, out: u32) -> Vec<ItDim> {
        let d = |name: &'static str, size: u32, is_stick: bool| ItDim {
            name,
            size,
            is_reduction: false,
            is_stick,
            df: Df::Fp16,
        };
        vec![d("mb", mb, false), d("out", out, true), d("y", 1, false)]
    }

    /// ⭐ THE SPLIT IS THE ENTRY AXIS ALONE — at the prefill kernel's own current form (32
    /// entries on 32 cores) the mint is `{mb: 32}` and `out` is carried WHOLE however many
    /// sticks it spans, because dxp derives `skip_addr` from the per-core extents of the
    /// unpinned dims and a split `out` shrinks what one index entry covers.
    #[test]
    fn the_split_is_the_entry_axis_alone() {
        let g = GatherSplit::over_entries(&dims(32, 256), 32, 1).expect("32 entries on 32 cores");
        assert_eq!(
            g.splits(),
            &BTreeMap::from([("mb", 32u32)]),
            "`mb` takes all the cores; `out` stays whole however many sticks it spans"
        );
    }

    /// ⭐ THE RULING (10-20): A GATHER LAWFULLY RUNS ON FEWER CORES THAN THE MACHINE — the
    /// prefill kernel's earlier 8-entry form is CONSTRUCTIBLE: the mint is `{mb: 8}`, the op
    /// runs on 8 cores with 24 idle, and `out` is still carried whole. What that form measured
    /// (`out` split 4 ways) was the FILLER split `distribute_cores` made to spend the idle
    /// cores — the wrong-byte shape — and `min(entries, cores)` over `mb` alone is the shape
    /// that keeps `skip_addr` correct at any core count. The batch width stays the perf move;
    /// it is no longer a correctness refusal.
    #[test]
    fn entries_fewer_than_cores_run_on_their_entries() {
        let g = GatherSplit::over_entries(&dims(8, 256), 32, 1).expect("8 entries on 32 cores");
        assert_eq!(
            g.splits(),
            &BTreeMap::from([("mb", 8u32)]),
            "8 whole entries, one per core, `out` never the filler"
        );
    }

    /// The vendor's own one-core gather (`dxp/test/test_gather_1core`) is the `entries == 1`
    /// case: `mb` is UNSPLIT (a split of 1 is no split) and the op runs on one core.
    #[test]
    fn one_entry_is_the_vendors_own_one_core_gather() {
        let g = GatherSplit::over_entries(&dims(1, 256), 32, 1).expect("1 entry on 32 cores");
        assert!(
            g.splits().is_empty(),
            "a single entry leaves `mb` unsplit — one core, one whole entry"
        );
    }

    /// At MORE entries than cores the split is [`core_split`]'s largest divisor — every core
    /// owns one or more WHOLE entries, and `out` still never splits.
    #[test]
    fn more_entries_than_cores_split_at_the_largest_divisor() {
        let g = GatherSplit::over_entries(&dims(64, 256), 32, 1).expect("64 entries on 32 cores");
        assert_eq!(g.splits(), &BTreeMap::from([("mb", 32u32)]));
        let g = GatherSplit::over_entries(&dims(48, 256), 32, 1)
            .expect("48 entries: 24 whole entries per core pair, no other axis touched");
        assert_eq!(
            g.splits(),
            &BTreeMap::from([("mb", 24u32)]),
            "the largest divisor within the budget — 48 has no 32, and `out` is not the filler"
        );
    }

    /// ⛔ A PARTIAL ENTRY IS AN ADDRESS — an `mb` the page does not divide would leave the last
    /// index entry covering a fraction of the axis, refused by name.
    #[test]
    fn a_partial_entry_is_refused() {
        let Err(m) = GatherSplit::over_entries(&dims(33, 256), 32, 2) else {
            panic!("33 positions at 2 per entry is a partial entry");
        };
        assert!(
            m.contains("fraction of the axis"),
            "names the partial entry: {m}"
        );
    }

    /// ⛔ NO `mb`, NO GATHER — a plan with no entry axis for the index to enumerate cannot be
    /// divided at this door at all.
    #[test]
    fn a_plan_with_no_mb_dim_is_refused() {
        let d = ItDim {
            name: "out",
            size: 256,
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        };
        let Err(m) = GatherSplit::over_entries(&[d], 32, 1) else {
            panic!("no `mb` dim means no entries to enumerate");
        };
        assert!(
            m.contains("no `mb` dim for the entries to enumerate"),
            "names the missing axis: {m}"
        );
    }

    /// THE PAGE DIVIDES POSITIONS INTO ENTRIES — the copy door's own law: at 64 positions and
    /// 2 per entry the division is over the 32 ENTRIES, so no core owns a fraction of one.
    #[test]
    fn the_page_divides_positions_into_entries() {
        let g = GatherSplit::over_entries(&dims(64, 64), 32, 2).expect("32 whole entries");
        assert_eq!(
            g.splits(),
            &BTreeMap::from([("mb", 32u32)]),
            "the split is over entries, not positions"
        );
    }

    /// The enum is the door's statement: an ungathered op carries the elementwise law, a
    /// gathered op carries its minted split — and the two spellings are the only ones.
    #[test]
    fn the_enum_carries_either_the_law_or_the_mint() {
        let g = GatherSplit::over_entries(&dims(32, 256), 32, 1).expect("the mint");
        match CoreDivision::Gathered(g) {
            CoreDivision::Gathered(g) => assert_eq!(g.splits(), &BTreeMap::from([("mb", 32u32)])),
            CoreDivision::Ew(_) => panic!("the minted split round-trips"),
        }
    }
}

/// [`distribute_cores`] with the STICK dim served FIRST — for a stick-major rank-2 elementwise op
/// ([`EwDivision::SticksFirst`]). Reductions are never split, for [`distribute_cores`]' own reason.
///
/// The stick dim takes the LARGEST whole-stick split that, with `mb` splitting the cores it leaves, fills
/// the most cores; then any other output dim splits what remains, largest first. Each core holds whole
/// sticks (the basis is the stick COUNT at the dim's own format, as there).
///
/// ⛔ NOT "the stick dim's largest split, then `mb`". That greedy order strands cores whenever the stick
/// count has no divisor near `max_cores`: granite-3.1-8b's second MLP column chunk is 72 sticks, whose
/// largest divisor ≤ 32 is 24 — and `32 / 24` leaves `mb` one core, so the op would run on 24 cores where
/// rows-first ran 32. Searching the stick split for the most cores gives `out:8 × mb:4`, which is on the
/// `out:32` line in #549's sweep. It is also why this can never use FEWER cores than [`distribute_cores`]:
/// that plan's stick split is one of the candidates, and `mb` given at least as many cores splits at
/// least as far.
pub fn distribute_cores_sticks_first(
    dims: &[ItDim],
    max_cores: u32,
) -> BTreeMap<&'static str, u32> {
    let mut splits: BTreeMap<&'static str, u32> = BTreeMap::new();
    let outs: Vec<&ItDim> = dims.iter().filter(|d| !d.is_reduction).collect();
    let stick = outs.iter().copied().find(|d| d.is_stick);
    let mb = outs.iter().copied().find(|d| d.name == "mb" && !d.is_stick);
    let mut remaining = max_cores;
    if let Some(s) = stick {
        let sticks = s.size.div_ceil(s.df.elems_per_stick());
        let mb_size = mb.map_or(1, |d| d.size);
        // The stick split filling the most cores with `mb`; ON A TIE, THE WIDER STICK SPLIT — the most
        // contiguous sticks per core is what #549's (e) cells measured as the win, so among equal core
        // counts the plan leans to `out`. 1 is always a candidate (it divides everything), so the search is
        // never empty. Pinned against the whole candidate set by the exhaustive test.
        let (ss, ms) = (1..=sticks.min(max_cores))
            .filter(|c| sticks.is_multiple_of(*c))
            .map(|c| (c, core_split(mb_size, max_cores / c)))
            .max_by_key(|&(c, m)| (c * m, c))
            .unwrap_or((1, 1));
        if ss > 1 {
            splits.insert(s.name, ss);
        }
        if ms > 1
            && let Some(d) = mb
        {
            splits.insert(d.name, ms);
        }
        remaining = max_cores / (ss * ms);
    }
    // Then the other output dims (and `mb`, when there is no stick dim), largest first.
    let mut rest: Vec<&ItDim> = outs
        .into_iter()
        .filter(|d| {
            !splits.contains_key(d.name) && !d.is_stick && !(stick.is_some() && d.name == "mb")
        })
        .collect();
    rest.sort_by_key(|d| std::cmp::Reverse(d.size));
    for d in rest {
        if remaining <= 1 {
            break;
        }
        let split = core_split(d.size, remaining);
        if split > 1 {
            splits.insert(d.name, split);
            remaining /= split;
        }
    }
    splits
}

/// ⭐⭐⭐⭐⭐ THE ROW-SPLIT (Y-MAJOR) OP'S OWN SPLITTER — the leading `y` (spray) axis first, then the
/// ordinary law over the cores it leaves.
///
/// ## Why this is a correctness law, not a preference (MEASURED on card, granite-3.1-2b prefill)
/// A row-split op presents its tensors `["y", "mb", "out"]` — `y` is the OUTERMOST dim of the flat
/// layout, so a core's slab is CONTIGUOUS exactly when the split is over `y`: core `y0` owns
/// `[mb, out]` at flat stick `y0·mb`, and the stride every level of the addressing derives (the
/// per-core start `per_core_addr`, the on-card walk `build_coordinates`) is computed from the
/// PER-CORE extents — which are the FULL extents for every dim the split does not touch.
///
/// Splitting any INNER dim instead hands every core a STRIDED stick set, and the per-core-extent
/// stride derivation then mis-walks it. MEASURED on the prefill mask add (`11_add_o1417`, the
/// `[y=32 qh, mb=21 t, out=64]` add of the causal mask): the rows-first law split `mb` 21 ways, so
/// each core's per-core `mb` extent was 1 row — its within-slab `y` stride came out ONE STICK
/// instead of 21, and its start address (the flat corner of `mb=c`) was one stick past the last.
/// Read straight off the emitted descriptor's `startAddressCoreCorelet_`: every core's base advanced
/// 128 B (ONE stick) while each core's slab was 32 sticks — the 21 cores covered sticks 0..51 of the
/// 672-stick score tile (identity reads, `t=20` stuttering onto `t=19`), and 620 sticks were NEVER
/// WRITTEN, read downstream as pure zero. That destroyed 29 of 32 heads' scores outright and
/// mis-cut the causal mask for the rest — the attention-dilution defect this splitter exists to
/// make unconstructible.
///
/// This is the y-major TWIN of [`distribute_cores`]' own mb-first note ("splitting `mb` first makes
/// each core own WHOLE rows … so the flat start alone places the row and there is no mb-stride to
/// mis-walk"): splitting `y` first makes each core own WHOLE SPRAYS, so the start alone places the
/// spray and there is no y-stride to mis-walk.
///
/// The remainder is spent by the ordinary law over the dims the spray axis leaves (an `mb` split
/// WITHIN one spray is contiguous — `mb` is the slab's own outer dim — so a short spray count loses
/// no cores: the J-slot merge at `y=4, mb=64` divides `{y:4, mb:8}` over 32 cores).
pub fn distribute_cores_sprays_first(
    dims: &[ItDim],
    max_cores: u32,
) -> BTreeMap<&'static str, u32> {
    let mut splits: BTreeMap<&'static str, u32> = BTreeMap::new();
    let mut remaining = max_cores;
    if let Some(y) = dims
        .iter()
        .find(|d| d.name == "y" && !d.is_reduction && !d.is_stick)
    {
        let split = core_split(y.size, remaining);
        if split > 1 {
            splits.insert(y.name, split);
            remaining /= split;
        }
    }
    // The cores the spray axis leaves, spent by the ordinary law over the dims it leaves — one
    // budget, one call, so the spray split and the remainder can never disagree about the count.
    let rest: Vec<ItDim> = dims.iter().filter(|d| d.name != "y").cloned().collect();
    let mut remainder = distribute_cores(&rest, remaining);
    splits.append(&mut remainder);
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

#[cfg(test)]
mod one_row_fill_tests {
    use super::{DeviceWidth, one_row_core_fill};

    /// granite-3.1-8b's decode projections, the shapes the table on [`one_row_core_fill`] measured —
    /// stick counts at the fp16 64-lane basis (`n/64`, `k/64`).
    #[test]
    fn the_one_row_fill_reaches_every_core_on_granite_8b() {
        // gate/up [4096 -> 12800]: 200 out-sticks have no divisor in 26..=32.
        assert_eq!(one_row_core_fill(200, 64, 32), (8, 4));
        // k/v [4096 -> 1024]: 16 out-sticks.
        assert_eq!(one_row_core_fill(16, 64, 32), (16, 2));
        // q/o [4096 -> 4096] and down [12800 -> 4096]: 64 out-sticks already fill 32.
        assert_eq!(one_row_core_fill(64, 64, 32), (32, 1));
        assert_eq!(one_row_core_fill(64, 200, 32), (32, 1));
        // lm_head, padded to 800 sticks (32 * 25).
        assert_eq!(one_row_core_fill(800, 64, 32), (32, 1));
    }

    /// granite-3.1-2b: k/v is 512 wide (8 sticks) — out8 alone is a quarter of the machine.
    #[test]
    fn the_one_row_fill_reaches_every_core_on_granite_2b() {
        assert_eq!(one_row_core_fill(8, 32, 32), (8, 4));
        assert_eq!(one_row_core_fill(32, 32, 32), (32, 1));
        assert_eq!(one_row_core_fill(128, 32, 32), (32, 1));
    }

    /// A one-stick reduction has nothing to give: the out-only split stands exactly as it was.
    #[test]
    fn a_one_stick_reduction_keeps_the_out_only_split() {
        assert_eq!(one_row_core_fill(16, 1, 32), (16, 1));
        assert_eq!(one_row_core_fill(32, 1, 32), (32, 1));
    }

    /// The Kani harness's property — the assertions `cargo test` can run on every build.
    fn assert_exact_fill(os: u32, is: u32) {
        let (o, i) = one_row_core_fill(os, is, 32);
        assert!(o >= 1 && i >= 1, "({os},{is})");
        assert_eq!(
            os % o,
            0,
            "({os},{is}): out factor {o} divides the output sticks"
        );
        assert_eq!(
            is % i,
            0,
            "({os},{is}): in factor {i} divides the reduction sticks"
        );
        assert!(o * i <= 32, "({os},{is}): {o}x{i} within the core budget");
        assert!(
            o * i >= super::core_split(os, 32),
            "({os},{is}): never fewer cores than the out-only split"
        );
    }

    /// Every dense one-row projection's `(out_sticks, in_sticks)` in every checked-in model config —
    /// q/k/v/o, gate/up/down (dense and MoE expert widths) and the lm_head at its device width —
    /// at the fp16 64-lane basis and the fp8 128-lane reduction basis. Read from the configs, so a new
    /// model's widths are covered the moment its config lands.
    fn config_projection_sticks() -> Vec<(u32, u32)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../models/arch/configs");
        let mut pairs = std::collections::BTreeSet::new();
        let mut files = 0usize;
        for arch in std::fs::read_dir(&root).expect("the model configs directory") {
            let arch = arch.expect("a config dir entry").path();
            if !arch.is_dir() {
                continue;
            }
            for f in std::fs::read_dir(&arch).expect("a config arch dir") {
                let f = f.expect("a config file entry").path();
                if f.extension().is_none_or(|e| e != "json") {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(
                    &std::fs::read_to_string(&f).expect("a readable config"),
                ) else {
                    continue;
                };
                // A multimodal config nests its language model; read both levels.
                for c in [&v, &v["text_config"]] {
                    let g = |k: &str| c[k].as_u64().map(|x| x as u32);
                    let Some(h) = g("hidden_size") else { continue };
                    files += 1;
                    let heads = g("num_attention_heads").unwrap_or(1).max(1);
                    let hd = g("head_dim").unwrap_or(h / heads);
                    let kvh = g("num_key_value_heads").unwrap_or(heads);
                    let mut outs_from_h = vec![heads * hd, kvh * hd];
                    let mut ffn = Vec::new();
                    for k in [
                        "intermediate_size",
                        "moe_intermediate_size",
                        "shared_expert_intermediate_size",
                    ] {
                        if let Some(x) = g(k) {
                            ffn.push(x);
                        }
                    }
                    outs_from_h.extend(ffn.iter().copied());
                    if let Some(vocab) = g("vocab_size") {
                        outs_from_h.push(DeviceWidth::for_output(1, vocab, h).get());
                    }
                    let mut shapes: Vec<(u32, u32)> = outs_from_h.iter().map(|&n| (n, h)).collect(); // [h -> n]
                    shapes.push((h, heads * hd)); // o_proj [heads*hd -> h]
                    shapes.extend(ffn.iter().map(|&f| (h, f))); // down [ffn -> h]
                    for (n, k) in shapes {
                        if n == 0 || k == 0 {
                            continue;
                        }
                        let os = n.div_ceil(super::FP16_ELEMS_PER_STICK);
                        pairs.insert((os, k.div_ceil(64)));
                        pairs.insert((os, k.div_ceil(128)));
                    }
                }
            }
        }
        assert!(
            files > 100,
            "read {files} configs — the configs directory moved?"
        );
        pairs.into_iter().collect()
    }

    /// The fill's partition law over EVERY real one-row projection shape in the checked-in configs,
    /// and over every stick-count pair up to the largest one those configs reach — so the bound is
    /// the configs' own, not a hand-picked range.
    #[test]
    fn the_one_row_fill_partitions_both_axes_exhaustively() {
        let pairs = config_projection_sticks();
        for &(os, is) in &pairs {
            assert_exact_fill(os, is);
        }
        // And the dense sweep around them: every output stick count up to the largest a config reaches
        // (gemma's 262,144-wide vocab is 4096 sticks) against every reduction count up to 64 — the
        // law reads `in_sticks` only through its divisors <= 32, so 1..=64 holds every distinct
        // divisor set below the core budget, and the config pairs above hold the real large ones.
        let max_os = pairs.iter().map(|p| p.0).max().unwrap_or(1);
        assert!(
            max_os >= 4096,
            "largest output axis {max_os} sticks — the vocab widths are missing"
        );
        for os in 1..=max_os {
            for is in 1..=64 {
                assert_exact_fill(os, is);
            }
        }
    }

    /// MOST CORES FIRST, then the widest `out`: a prime reduction whose only divisor past 1 is itself
    /// still beats a half-empty machine (31 cores reading against 16).
    #[test]
    fn most_cores_wins_before_the_widest_out() {
        assert_eq!(one_row_core_fill(16, 31, 32), (1, 31));
        // Ties on cores go to the wider `out` — the smaller reduction fan-in.
        assert_eq!(one_row_core_fill(32, 64, 32), (32, 1));
        assert_eq!(one_row_core_fill(8, 64, 32), (8, 4));
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::one_row_core_fill;

    /// ⭐ THE FILL IS AN EXACT PARTITION OF BOTH AXES, AND NEVER WORSE THAN THE OUT-ONLY SPLIT.
    ///
    /// For every output and reduction stick count up to the bound: each factor divides its own stick
    /// count (so the equal whole-stick slices tile the axis with no remainder — disjoint and covering on
    /// `out`, covering on `in`), the product fits the core budget, and the fill uses at least as many
    /// cores as the out-only split it replaces.
    #[kani::proof]
    #[kani::unwind(34)]
    fn the_one_row_fill_partitions_both_axes() {
        let os: u32 = kani::any();
        let is: u32 = kani::any();
        kani::assume((1..=32).contains(&os));
        kani::assume((1..=32).contains(&is));
        let (o, i) = one_row_core_fill(os, is, 32);
        assert!(o >= 1 && i >= 1);
        assert!(os % o == 0, "out factor divides the output stick count");
        assert!(is % i == 0, "in factor divides the reduction stick count");
        assert!(o * i <= 32, "within the core budget");
        assert!(
            o * i >= super::core_split(os, 32),
            "never fewer cores than out-only"
        );
    }
}
