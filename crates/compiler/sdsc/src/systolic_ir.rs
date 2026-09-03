// SPDX-License-Identifier: Apache-2.0
//! # SystolicIR — the typed layout layer that makes a wrong device layout UNEXPRESSIBLE
//!
//! **Purpose (read this first).** A tensor's device layout used to be reconstructed imperatively at
//! *every* op via the free `dev_off(dims, stick_idx, idx)` — each call site passing its own dims and
//! stick index. Nothing tied a tensor to ONE layout, so a producer and a consumer of the same tensor
//! could silently disagree. `SystolicIR` gives every tensor exactly ONE [`StickLayout`] that OWNS its
//! device address AND its iteration sweeps, so producer and consumer derive from the same source and
//! CANNOT diverge.
//!
//! ## The bug class it kills
//! Spyre is a SYSTOLIC array: a contraction STREAMS one axis (the [`AxisRole::Reduction`], e.g. matmul
//! K) through the PE array and ACCUMULATES, while CARRYING the free axes ([`AxisRole::FreeM`] rows/
//! tokens across the PT rows, [`AxisRole::FreeN`] output sticks). HBM tensors live as 64-lane fp16
//! STICKS, not flat row-major. Every prefill bug this project fought was ONE class: *something assumed
//! a layout that only coincides with the real one at `m==1`*. **At `m==1` (decode) flat / stick-blocked
//! / mb-major all produce byte-identical addresses** — so decode validates NOTHING about layout, and
//! the disagreement only scrambles values at `m>1` (prefill), with no crash. Conflating `Reduction`
//! with `FreeM` (a per-row reduce that strides M as if it were part of the reduction, mixing rows) is
//! the archetype.
//!
//! ## The design
//! - [`AxisRole`] `{Reduction, FreeM, FreeN, Lane}` — the distinction the PE array itself makes.
//! - [`StickKind`] `{RowBlocked, Kernel, RowScalar, Flat}` — the physical stick residencies; each picks
//!   a `dev_off` formula and a sweep. EXTENSIBLE without touching the address algebra — a scale-bearing
//!   residency (W8A16 / turboquant: the per-channel scale rides `FreeN` while the weight streams
//!   `Reduction`), an expert-indexed residency (MoE), a windowed-KV kernel (SWA).
//! - [`StickLayout`] owns `dev_off(r,c)`, the sweeps [`StickLayout::reduce_over_feature`] /
//!   [`StickLayout::broadcast_rowscalar`], and the general-rank classifier [`StickLayout::for_view`] /
//!   [`StickLayout::off_view`]. `m` (row count) is a RUNTIME field, NOT a type parameter — so symbolic
//!   continuous-batching stays expressible and decode is literally the prefill code at `m==1` (no
//!   separate path to drift).
//! - [`Stk`]`<K>` — a typed handle carrying the ONE layout with its KIND in the type: handing a
//!   `RowScalar` reduce-output to a matmul-A (which needs `RowBlocked`) does not compile (`E0308` — see
//!   the `compile_fail` doctests on [`Stk`]).
//!
//! ## Wired into the LIVE emitter (byte-identical, proven)
//! EVERY device-address computation in the SuperDSC emitter now flows through `StickLayout` — ZERO raw
//! `dev_off` calls remain in `lower_subtile_tape_to_superdsc.rs`: the K/V-cache offsets
//! (`kcache_kt_write_offset` / `vcache_write_offset` → `StickLayout::kernel`) and `per_core_addr`
//! (EVERY op's baked per-core device address → `for_view(dims,stick).off_view(dims,corner)`). This
//! changed ZERO emitted bytes (the "migration bridge"): `StickLayout::dev_off` is proven byte-identical
//! to the free `dev_off`, which survives ONLY as the primitive the type delegates to.
//!
//! ## Why it lives where it does — and the trap it avoids
//! The vocabulary is defined in `scratchy_subtile::sdsc_abstract` (next to `dev_off`), because the live
//! emitter (in `scratchy-subtile`) cannot depend on this tower crate (circular); this module re-uses
//! it. And `FlatIR`/`SystolicIR` are SINGLE-ASSIGNMENT, so keying a layout by tensor id is SOUND here —
//! unlike a tid-keyed registry wired into the RE-ROLLED monolith tape, which FALSE-conflicts because
//! the reroll REUSES tid slots (buffer reuse: one tid is an 8192-wide MLP intermediate at one point and
//! a 512-wide KV tensor at another). **Do not wire a tid registry into the monolith reroll.**
//!
//! ## Guarantees (Kani harnesses in `kani_layout`, + the `#[test]`s)
//! `dev_off_method_equals_free_fn` (decode-safety lock, every `StickKind`); `for_view_off_equals_dev_
//! off_rank2/3/4` (classifier byte-identity, every emitter rank); `off_view_injective_rank2/3/4` (no
//! #50 aliasing); `off_view_in_footprint_rank2/3` (no OOB); `kv_cache_offsets_locked_to_sticklayout`
//! (the attention path cannot revert to hand-rolled stick math); `layout_reduce_reads_producer_written_
//! element`; `normalize_broadcast_pairs_own_row`; `free_extents_never_split_reduction`;
//! `layout_ctor_kind_guard`; and the FAIL-FIRST RED `layout_disagreement_is_rejected` (models the flat-
//! vs-stick divergence at `m>1`). Plus a non-vacuous defect-counter test. Verified end-to-end by REAL
//! `scr` output (granite-3.3-2b sequential prefill coherent, unchanged).
//!
//! ## Extending it (the discipline)
//! Migrate a call site to `StickLayout` BYTE-IDENTICALLY first, locked by a Kani `*_equals_dev_off`-style
//! proof BEFORE wiring; only THEN make a behavior change (e.g. deriving a batched `m>1` sweep from the
//! layout so a wrong stride becomes unexpressible), and judge it by REAL `scr` output — never a proxy.
//! The batched (`m>1`) prefill bug is the natural next target for this now-live layout.

use crate::flat_ir::{FlatIR, FlatNode, FlatOp, Shape};
use crate::macro_ir::seed_src;
use crate::tiled_ir::CoreSplit;
// Re-export the systolic vocabulary (defined in subtile next to `dev_off` so the live emitter can
// reach it) so `systolic_ir::*` consumers — and the `Stk` doctests — get `StickLayout` etc. here.
use scratchy_subtile::sdsc_abstract::dev_off;
pub use scratchy_subtile::sdsc_abstract::{AxisRole, DeviceExtents, Span, StickKind, StickLayout};
#[cfg(kani)]
use scratchy_subtile::sdsc_abstract::{kcache_kt_write_offset, vcache_write_offset};
use scratchy_subtile::subtile_ir::{RopeForm, SubtileIR, eval_dag};

// ── The typed tensor handle `Stk<K>` now lives in subtile (next to `StickLayout`) so the LIVE emitter
//    can thread typed handles; re-exported here. A producer/consumer KIND mismatch is a compile error. ──
pub use scratchy_subtile::sdsc_abstract::{
    FlatTag, KernelTag, KindTag, LayoutError, RowBlockedTag, RowScalarTag, Stk,
};

// ── The IR: FlatIR's op-graph + one StickLayout per tensor + per-op AxisRoles ──

/// Per-op systolic role assignment (parallel to `FlatIR::nodes`). `in_kinds` is the kind each input
/// is REQUIRED to have by the op's semantics; a `layouts[in]` that disagrees is a defect.
#[derive(Clone, Debug)]
pub struct OpRoles {
    pub out: usize,
    pub out_kind: StickKind,
    pub out_row: AxisRole,
    pub out_col: AxisRole,
    pub in_kinds: Vec<StickKind>,
    /// The contracted extent (matmul K / reduce feature), else `None`.
    pub reduction_len: Option<u32>,
}

/// A distinct IR: the `FlatIR` op-graph with ONE `StickLayout` per tensor (`layouts`, parallel to
/// `tensors`) + per-op `OpRoles` (`roles`, parallel to nodes). Arithmetic is `FlatIR`'s (reused for
/// `eval`); this island layers the single-source layout on top.
#[derive(Clone, Debug)]
pub struct SystolicIR {
    pub tensors: Vec<Shape>,
    pub num_sources: u32,
    pub layouts: Vec<StickLayout>,
    pub roles: Vec<OpRoles>,
    pub result: usize,
    flat: FlatIR,
}

/// Build the `StickLayout` for `kind` from a 2-D `Shape` (rows/cols carried from FlatIR).
fn layout_for(kind: StickKind, sh: Shape) -> StickLayout {
    let (r, c) = (sh.rows as usize, sh.cols as usize);
    match kind {
        StickKind::RowBlocked => StickLayout::row_blocked(r, c),
        StickKind::Kernel => StickLayout::kernel(r, c),
        StickKind::RowScalar => StickLayout::row_scalar(r),
        StickKind::Flat => StickLayout::flat(r, c),
    }
}

/// The per-`FlatOp` systolic role assignment. Output is post-contraction (`out_row=FreeM`, `out_col`
/// ∈ {`FreeN`, `Lane`}) so it never carries a `Reduction` axis; inputs state the kind they require.
/// At the FlatIR level every tensor is 2-D `RowBlocked` except matmul weights (`Kernel`) and per-row
/// reduce outputs (`RowScalar`) — no head-major `Flat` here (that is an emission-island concern), so
/// attention I/O stays `RowBlocked` and cannot false-conflict with its `RowBlocked` producers.
pub fn op_roles(node: &FlatNode, tensors: &[Shape]) -> OpRoles {
    let out = node.out;
    let nin = node.ins.len();
    let rb = || vec![StickKind::RowBlocked; nin];
    let (out_kind, out_row, out_col, in_kinds, reduction_len): (
        StickKind,
        AxisRole,
        AxisRole,
        Vec<StickKind>,
        Option<u32>,
    ) = match &node.op {
        FlatOp::Matmul => {
            let k = tensors[node.ins[0]].cols; // A is [m, k]
            (
                StickKind::RowBlocked,
                AxisRole::FreeM,
                AxisRole::FreeN,
                vec![StickKind::RowBlocked, StickKind::Kernel],
                Some(k),
            )
        }
        FlatOp::RowSum | FlatOp::RowMax => {
            let c = tensors[node.ins[0]].cols; // reduce over the feature axis
            (
                StickKind::RowScalar,
                AxisRole::FreeM,
                AxisRole::Lane,
                vec![StickKind::RowBlocked],
                Some(c),
            )
        }
        FlatOp::RsqrtMeanEps { .. } => (
            StickKind::RowScalar,
            AxisRole::FreeM,
            AxisRole::Lane,
            vec![StickKind::RowScalar],
            None,
        ),
        FlatOp::MulRowBcast | FlatOp::SubRowBcast | FlatOp::DivRowBcast => (
            StickKind::RowBlocked,
            AxisRole::FreeM,
            AxisRole::FreeN,
            vec![StickKind::RowBlocked, StickKind::RowScalar],
            None,
        ),
        // Attention: 2-D at this level; a head-dim/seq reduction, coarse. All I/O RowBlocked.
        FlatOp::Score { head_dim, .. } => (
            StickKind::RowBlocked,
            AxisRole::FreeM,
            AxisRole::FreeN,
            rb(),
            Some(*head_dim),
        ),
        FlatOp::WeightedValue { .. } => (
            StickKind::RowBlocked,
            AxisRole::FreeM,
            AxisRole::FreeN,
            rb(),
            None,
        ),
        // Every other op (elementwise / broadcast-by-col-or-head / rope / scale / sum / silu / exp /
        // square) produces same-shape RowBlocked from RowBlocked inputs — a pure carry, no reduction.
        _ => (
            StickKind::RowBlocked,
            AxisRole::FreeM,
            AxisRole::FreeN,
            rb(),
            None,
        ),
    };
    OpRoles {
        out,
        out_kind,
        out_row,
        out_col,
        in_kinds,
        reduction_len,
    }
}

/// THE PASS: `FlatIR → SystolicIR`. Assign one `StickLayout` per tensor. Producers set their output's
/// layout from `op_roles`; matmul-weight SOURCES are set to `Kernel` (the sole non-default source
/// residency). Produced tensors are never overwritten by a consumer's requirement — so a genuine
/// producer/consumer disagreement survives as a defect (`eval`) rather than being papered over.
pub fn lower_flat_to_systolic(f: &FlatIR) -> SystolicIR {
    // Default every tensor RowBlocked from its 2-D shape.
    let mut layouts: Vec<StickLayout> = f
        .tensors
        .iter()
        .map(|sh| StickLayout::row_blocked(sh.rows as usize, sh.cols as usize))
        .collect();
    let mut roles = Vec::with_capacity(f.nodes.len());
    for node in &f.nodes {
        let r = op_roles(node, &f.tensors);
        // Producer sets the output layout.
        layouts[node.out] = layout_for(r.out_kind, f.tensors[node.out]);
        // A matmul weight is a SOURCE with no producer — give it its Kernel residency. (Only sources
        // are assigned from `in_kinds`; produced inputs keep their producer's layout so disagreements
        // are visible, never clobbered.)
        for (i, &inp) in node.ins.iter().enumerate() {
            if inp < f.num_sources as usize && r.in_kinds[i] != StickKind::RowBlocked {
                layouts[inp] = layout_for(r.in_kinds[i], f.tensors[inp]);
            }
        }
        roles.push(r);
    }
    SystolicIR {
        tensors: f.tensors.clone(),
        num_sources: f.num_sources,
        layouts,
        roles,
        result: f.result,
        flat: f.clone(),
    }
}

/// The `(row_cores_extent, stick_cores_extent)` the splitter may divide: a `FreeM` row axis
/// contributes its full extent, a `FreeN` col axis its full extent, and ANY OTHER role (`Reduction`,
/// `Lane`) contributes `1` — i.e. is NEVER split. This is the whole systolic guarantee, isolated as
/// pure enum/int logic (Kani `free_extents_never_split_reduction`, closes <1s — no `plan` loop).
pub fn free_extents(roles: &OpRoles, out: Shape) -> (u32, u32) {
    let free_m = if roles.out_row == AxisRole::FreeM {
        out.rows
    } else {
        1
    };
    let free_n = if roles.out_col == AxisRole::FreeN {
        out.cols
    } else {
        1
    };
    (free_m, free_n)
}

/// The 32-core split DERIVED from an op's free axes via [`free_extents`]: split only `FreeM`/`FreeN`,
/// never a `Reduction` axis. Because every FlatOp output is post-contraction, this equals today's
/// `CoreSplit::plan(rows, cols)` at all current shapes (runtime `#[test] derive_core_split_matches_plan`)
/// — the value is that splitting a reduction axis is now STRUCTURALLY refused, not merely absent.
pub fn derive_core_split(roles: &OpRoles, out: Shape) -> CoreSplit {
    let (free_m, free_n) = free_extents(roles, out);
    CoreSplit::plan(free_m, free_n)
}

impl SystolicIR {
    /// Run the layout-annotated execution. Values are `FlatIR::eval` (arithmetic UNCHANGED). The
    /// second return is `layout_defects`: the number of op-input edges whose recorded `layouts[in]`
    /// disagrees with the consumer's required `in_kinds[i]`, PLUS any element where the layout's
    /// `dev_off` diverges from the free `dev_off` (the runtime mirror of the decode-safety lock;
    /// always 0 by proof #1). `0` ⇒ every producer/consumer handoff shares one layout.
    pub fn eval(&self, sources: &[&[f32]]) -> (Vec<Vec<f32>>, u32) {
        let values = self.flat.eval(sources);
        let mut defects = 0u32;
        for (k, node) in self.flat.nodes.iter().enumerate() {
            let r = &self.roles[k];
            for (i, &inp) in node.ins.iter().enumerate() {
                if self.layouts[inp].kind != r.in_kinds[i] {
                    defects += 1;
                }
            }
        }
        // dev_off spot-check: the layout's address equals the free dev_off on a corner of each tensor.
        for (t, lay) in self.layouts.iter().enumerate() {
            let (rows, cols) = (self.tensors[t].rows as usize, self.tensors[t].cols as usize);
            if rows == 0 || cols == 0 {
                continue;
            }
            let (r, c) = (rows - 1, cols - 1);
            let stick_idx = if lay.kind == StickKind::Flat {
                usize::MAX
            } else {
                1
            };
            if lay.dev_off(r, c) != dev_off(&[lay.rows, lay.cols], stick_idx, &[r, c]) {
                defects += 1;
            }
        }
        (values, defects)
    }
}

/// VERIFY the `FlatIR → SystolicIR` bridge against the golden `eval_dag`: values must match per tensor
/// (relative-L2 ≤ 1e-2) AND `layout_defects == 0` (every producer/consumer handoff shares one
/// layout). Panics (un-catchable `cargo build`/test failure) on divergence — the island's obligation.
pub fn verify_against_eval_dag<F: RopeForm>(layout: &SystolicIR, subtile: &SubtileIR<F>) {
    const RELL2_TOL: f64 = 1e-2;
    const EPS: f64 = 1e-9;
    assert_eq!(
        layout.num_sources, subtile.num_sources,
        "SystolicIR bridge: source count {} != SubtileIR {}",
        layout.num_sources, subtile.num_sources
    );
    let src: Vec<Vec<f32>> = (0..subtile.num_sources as usize)
        .map(|s| {
            let sh = subtile.tensors[s];
            (0..(sh.rows * sh.cols) as usize)
                .map(|j| seed_src(s, j))
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
    let golden = eval_dag(subtile, &refs);
    let (got, defects) = layout.eval(&refs);
    assert_eq!(
        defects, 0,
        "SystolicIR bridge: {defects} producer/consumer layout disagreement(s) — a tensor's producer \
         and a consumer declared different StickLayouts (the bug#1 class). Reconcile the assignment."
    );

    let n_orig = subtile.tensors.len();
    let (mut worst_rell2, mut worst_tid) = (0f64, 0usize);
    for tid in 0..n_orig {
        let (g, o) = (&golden[tid], &got[tid]);
        assert_eq!(
            g.len(),
            o.len(),
            "SystolicIR bridge: tensor t{tid} size changed"
        );
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..g.len() {
            let (gv, ov) = (g[i] as f64, o[i] as f64);
            num += (gv - ov) * (gv - ov);
            den += gv * gv;
        }
        let rell2 = num.sqrt() / den.sqrt().max(EPS);
        if rell2 > worst_rell2 {
            worst_rell2 = rell2;
            worst_tid = tid;
        }
    }
    assert!(
        worst_rell2 <= RELL2_TOL,
        "SystolicIR bridge diverges at t{worst_tid}: relative-L2 {worst_rell2:.2e} > {RELL2_TOL}. The \
         layout annotation changed the math — it must not (eval reuses FlatIR::eval)."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat_ir::lower_sm_to_flat;
    use crate::prim_ir::{PrimIR, PrimNode, PrimOp, Shape as PShape};
    use crate::sm_ir::lower_prim_to_sm;

    /// The FlatIR→SystolicIR bridge on a softmax block: values identical to FlatIR + zero layout defects.
    #[test]
    fn layout_bridge_values_match_and_no_defects() {
        let (r, c) = (4usize, 7usize);
        let tensors = vec![
            PShape {
                rows: r as u32,
                cols: c as u32,
            },
            PShape {
                rows: r as u32,
                cols: c as u32,
            },
        ];
        let nodes = vec![PrimNode {
            op: PrimOp::Softmax,
            ins: vec![0],
            out: 1,
        }];
        let pir = PrimIR {
            tensors,
            num_sources: 1,
            nodes,
            result: 1,
        };
        let src: Vec<Vec<f32>> = vec![(0..(r * c)).map(|j| seed_src(0, j)).collect()];
        let refs: Vec<&[f32]> = src.iter().map(|v| v.as_slice()).collect();
        let flat = lower_sm_to_flat(&lower_prim_to_sm(&pir));
        let gf = flat.eval(&refs);
        let layout = lower_flat_to_systolic(&flat);
        let (gl, defects) = layout.eval(&refs);
        assert_eq!(
            defects, 0,
            "softmax block should have no layout disagreement"
        );
        for i in 0..(r * c) {
            assert!((gf[1][i] - gl[1][i]).abs() < 1e-6, "value drift at {i}");
        }
    }

    /// `derive_core_split` reproduces the current `CoreSplit::plan(rows,cols)` bridge on granite shapes.
    #[test]
    fn derive_core_split_matches_plan() {
        for &(r, c) in &[(1u32, 4096u32), (1, 1024), (256, 128), (8, 2048)] {
            let roles = OpRoles {
                out: 0,
                out_kind: StickKind::RowBlocked,
                out_row: AxisRole::FreeM,
                out_col: AxisRole::FreeN,
                in_kinds: vec![],
                reduction_len: None,
            };
            assert_eq!(
                derive_core_split(&roles, Shape { rows: r, cols: c }),
                CoreSplit::plan(r, c)
            );
        }
        // Reduce output [R,1] (out_col = Lane) splits free_n = 1 ⇒ plan(R,1).
        let reduce_roles = OpRoles {
            out: 0,
            out_kind: StickKind::RowScalar,
            out_row: AxisRole::FreeM,
            out_col: AxisRole::Lane,
            in_kinds: vec![],
            reduction_len: None,
        };
        assert_eq!(
            derive_core_split(&reduce_roles, Shape { rows: 8, cols: 1 }),
            CoreSplit::plan(8, 1)
        );
    }

    /// The defect counter is NON-VACUOUS: a valid graph reports 0, but a CORRUPTED producer layout
    /// (a weight declared RowBlocked where the matmul consumes it as a Kernel) is FLAGGED. Proves the
    /// "producer/consumer disagreement is a defect" guard actually has teeth (fail-first for the guard).
    #[test]
    fn eval_defect_counter_catches_wrong_producer_kind() {
        // A[2,4] @ W[4,3] -> O[2,3]; sources A=0, W=1; out O=2.
        let flat = FlatIR {
            tensors: vec![
                Shape { rows: 2, cols: 4 },
                Shape { rows: 4, cols: 3 },
                Shape { rows: 2, cols: 3 },
            ],
            num_sources: 2,
            nodes: vec![FlatNode {
                op: FlatOp::Matmul,
                ins: vec![0, 1],
                out: 2,
            }],
            result: 2,
        };
        let src_a = [0f32; 8];
        let src_w = [0f32; 12];
        let refs: [&[f32]; 2] = [&src_a, &src_w];
        let mut sys = lower_flat_to_systolic(&flat);
        // The weight source is classified Kernel ⇒ agrees with the matmul's in_kinds[1] ⇒ 0 defects.
        assert_eq!(
            sys.eval(&refs).1,
            0,
            "valid graph must have no layout defect"
        );
        // CORRUPT the weight's declared layout to RowBlocked — a producer/consumer kind disagreement.
        sys.layouts[1] = StickLayout::row_blocked(4, 3);
        assert!(
            sys.eval(&refs).1 > 0,
            "eval must FLAG the corrupted weight kind (Kernel consumer vs RowBlocked producer)"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  Kani proofs — the systolic-layout invariants (run `cargo kani -p scratchy-sdsc`). Follows the
//  seg_layout in-file `#[cfg(kani)] mod` pattern. Proof #3 is FAIL-FIRST (RED) — it models the bug.
// ══════════════════════════════════════════════════════════════════════════════════════════════
#[cfg(kani)]
mod kani_layout {
    use super::*;

    /// GREEN — DECODE-SAFETY LOCK: `StickLayout::dev_off` is BYTE-IDENTICAL to the free `dev_off` for
    /// every kind, so routing any call site through the layout changes ZERO emitted bytes.
    #[kani::proof]
    fn dev_off_method_equals_free_fn() {
        let m: usize = kani::any();
        let feat: usize = kani::any();
        let r: usize = kani::any();
        let c: usize = kani::any();
        kani::assume(m >= 1 && m <= 8);
        kani::assume(feat >= 64 && feat <= 512 && feat % 64 == 0);
        kani::assume(r < m && c < feat);
        let rb = StickLayout::row_blocked(m, feat);
        assert!(rb.dev_off(r, c) == dev_off(&[m, feat], 1, &[r, c]));
        let n = 64 * (1 + (m % 4));
        let kn = StickLayout::kernel(feat, n);
        let rr = r % feat.max(1);
        kani::assume(c < n);
        assert!(kn.dev_off(rr, c) == dev_off(&[feat, n], 1, &[rr, c]));
        let rs = StickLayout::row_scalar(m);
        assert!(rs.dev_off(r, 0) == dev_off(&[m, 64], 1, &[r, 0]));
        assert!(rs.dev_off(r, 0) == r * 64);
        let fl = StickLayout::flat(m, feat);
        assert!(fl.dev_off(r, c) == dev_off(&[m, feat], 0, &[r, c]));
        assert!(fl.dev_off(r, c) == r * feat + c);
    }

    /// GREEN — THE FIX IS CORRECT: a per-row feature reduction driven by `ReduceNest` reads EXACTLY
    /// the `RowBlocked` producer's write address for `(r, t*lane+l)`, for ANY m.
    #[kani::proof]
    fn layout_reduce_reads_producer_written_element() {
        let m: usize = kani::any();
        let feat: usize = kani::any();
        kani::assume(m >= 1 && m <= 8);
        kani::assume(feat >= 64 && feat <= 512 && feat % 64 == 0);
        let layout = StickLayout::row_blocked(m, feat);
        let nest = layout.reduce_over_feature();
        let r: usize = kani::any();
        let t: usize = kani::any();
        let l: usize = kani::any();
        kani::assume(r < m && t < nest.red_tiles && l < nest.lane);
        assert!(nest.elem_off(r, t, l) == layout.dev_off(r, t * nest.lane + l));
    }

    /// FAIL-FIRST (RED at m>1) — THE BUG, MODELED: a reduce that reads element `(r,k)` at the FLAT
    /// address `r*feat+k` (the pre-bug#1 assumption) must equal the STICK-BLOCKED producer's write.
    /// Kani finds `m>1, k>=64` where they differ (GREEN only at `m==1`, where flat==stick — why the
    /// bug hid in decode). The FIX routes the reduce through `ReduceNest` (proof #2), making this
    /// divergence unexpressible.
    #[kani::proof]
    fn layout_disagreement_is_rejected() {
        let m: usize = kani::any();
        let feat: usize = kani::any();
        kani::assume(m >= 1 && m <= 8);
        kani::assume(feat >= 128 && feat <= 512 && feat % 64 == 0);
        let r: usize = kani::any();
        let k: usize = kani::any();
        kani::assume(r < m && k < feat);
        let producer = StickLayout::row_blocked(m, feat).dev_off(r, k);
        let flat_read = r * feat + k;
        assert!(flat_read == producer);
    }

    /// GREEN — THE NORMALIZE FIX IS CORRECT: the per-row recip broadcast pairs element `(r,c)` with
    /// ROW r's lane-0 scalar for EVERY c and m — recip[r] cannot leak into row r'.
    #[kani::proof]
    fn normalize_broadcast_pairs_own_row() {
        let m: usize = kani::any();
        let feat: usize = kani::any();
        kani::assume(m >= 1 && m <= 8);
        kani::assume(feat >= 64 && feat <= 512 && feat % 64 == 0);
        let nest = StickLayout::row_blocked(m, feat).broadcast_rowscalar();
        let r: usize = kani::any();
        let c: usize = kani::any();
        kani::assume(r < m && c < feat);
        assert!(nest.scalar_off(r, c) == r * 64);
        assert!(nest.scalar_off(r, c) == StickLayout::row_scalar(m).dev_off(r, 0));
        assert!(nest.data_off(r, c) == StickLayout::row_blocked(m, feat).dev_off(r, c));
    }

    /// GREEN — Stk<K> CTOR GUARD (the runtime belt behind the compile-time KIND type guard): minting
    /// a handle whose runtime `StickKind` != the tag's kind returns `Err` (never a silent mis-tag).
    #[kani::proof]
    fn layout_ctor_kind_guard() {
        let m: usize = kani::any();
        let feat: usize = kani::any();
        kani::assume(m >= 1 && m <= 8 && feat >= 64 && feat <= 512 && feat % 64 == 0);
        assert!(Stk::<RowBlockedTag>::new("t", StickLayout::row_blocked(m, feat)).is_ok());
        assert!(Stk::<RowScalarTag>::new("t", StickLayout::row_scalar(m)).is_ok());
        assert!(Stk::<RowBlockedTag>::new("t", StickLayout::row_scalar(m)).is_err());
        assert!(Stk::<RowScalarTag>::new("t", StickLayout::row_blocked(m, feat)).is_err());
    }

    /// GREEN — THE SYSTOLIC SPLIT GUARANTEE, proven directly on `free_extents` (no `plan` divisor loop
    /// ⇒ closes <1s). For SYMBOLIC axis roles and shape: the extent handed to the splitter is the axis's
    /// full extent IFF it is `FreeM`/`FreeN`, and `1` for EVERY other role — so a `Reduction` (or `Lane`)
    /// axis is NEVER split. This is the invariant that makes the M>1 "split the reduction as if it were
    /// FreeM" scramble unexpressible; the numeric equality to today's `plan(rows,cols)` is the runtime
    /// `#[test] derive_core_split_matches_plan`.
    #[kani::proof]
    fn free_extents_never_split_reduction() {
        fn role_of(sel: u8) -> AxisRole {
            match sel % 4 {
                0 => AxisRole::Reduction,
                1 => AxisRole::FreeM,
                2 => AxisRole::FreeN,
                _ => AxisRole::Lane,
            }
        }
        let rows: u32 = kani::any();
        let cols: u32 = kani::any();
        kani::assume(rows >= 1 && rows <= 512 && cols >= 1 && cols <= 8192);
        let out_row = role_of(kani::any());
        let out_col = role_of(kani::any());
        let roles = OpRoles {
            out: 0,
            out_kind: StickKind::RowBlocked,
            out_row,
            out_col,
            in_kinds: Vec::new(),
            reduction_len: None,
        };
        let (fm, fnn) = free_extents(&roles, Shape { rows, cols });
        // A row axis is split by its extent IFF FreeM; every other role (incl. Reduction) ⇒ 1.
        if out_row == AxisRole::FreeM {
            assert!(fm == rows);
        } else {
            assert!(fm == 1);
        }
        if out_col == AxisRole::FreeN {
            assert!(fnn == cols);
        } else {
            assert!(fnn == 1);
        }
        // THE guarantee, stated explicitly: a Reduction axis's extent is NEVER the split count.
        if out_row == AxisRole::Reduction {
            assert!(fm == 1);
        }
        if out_col == AxisRole::Reduction {
            assert!(fnn == 1);
        }
    }

    /// GREEN — per_core_addr MIGRATION LOCK (rank-2): the typed `StickLayout::for_view(dims,stick)
    /// .off_view(dims,corner)` is BYTE-IDENTICAL to the free `dev_off(dims,stick,corner)` for every
    /// rank-2 view and every stick_idx. So routing `per_core_addr` (every op's baked address) through
    /// the systolic layout changes ZERO emitted bytes.
    #[kani::proof]
    fn for_view_off_equals_dev_off_rank2() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let stick: usize = kani::any();
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(a >= 1 && a <= 8);
        kani::assume(b >= 64 && b <= 512 && b % 64 == 0);
        kani::assume(i < a && j < b);
        kani::assume(stick <= 2); // covers stick==1 (stick-blocked) and stick!=1 (flat)
        let dims = [a, b];
        let corner = [i, j];
        let sl = StickLayout::for_view(&dims, stick);
        assert!(sl.off_view(&ext(&dims), &corner) == dev_off(&dims, stick, &corner));
    }

    /// GREEN — per_core_addr MIGRATION LOCK (rank-3): same byte-identity for rank-3 views (the
    /// head-major / trailing-singleton activation shapes) — all classify to `Flat` and fold row-major,
    /// exactly the free `dev_off` else-branch.
    #[kani::proof]
    fn for_view_off_equals_dev_off_rank3() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let c: usize = kani::any();
        let stick: usize = kani::any();
        let i: usize = kani::any();
        let j: usize = kani::any();
        let k: usize = kani::any();
        kani::assume(a >= 1 && a <= 8 && b >= 1 && b <= 64 && c >= 1 && c <= 64);
        kani::assume(i < a && j < b && k < c);
        kani::assume(stick <= 3);
        let dims = [a, b, c];
        let corner = [i, j, k];
        let sl = StickLayout::for_view(&dims, stick);
        assert!(sl.off_view(&ext(&dims), &corner) == dev_off(&dims, stick, &corner));
    }

    fn pad64(c: usize) -> usize {
        c.div_ceil(64) * 64
    }

    /// The extent vector these byte-identity / injectivity harnesses fold over: the plain iteration
    /// shape, NOTHING declared, every dim ranged — [`DeviceExtents::of_view`]'s identity case, so
    /// `off_view` sees exactly `dims` and the locks below still compare against the free `dev_off`.
    fn ext(dims: &[usize]) -> DeviceExtents {
        let n = dims.len();
        DeviceExtents::of_view(
            dims,
            &[None; 4][..n],
            usize::MAX,
            &[true; 4][..n],
            Span::Swept,
        )
    }

    /// 🛑 FAIL-FIRST (RED before the `DeviceExtents` commit) — **A DECLARED PHYSICAL EXTENT SETS THE
    /// ADDRESS STRIDE.**
    ///
    /// A batched decode gives each request its OWN KV pages, so an attention bmm must ITERATE a 64-slot
    /// window of the resident cache while ADDRESSING the whole allocation: the per-`y` (per-request)
    /// stride is a property of the POOL, not of how much of it one launch touches.
    /// `TensorArg::with_device_extent` is the only way to say that, and it was inert on every emitted
    /// address — the per-core addresser hand-rolled its extents from the iteration space while the
    /// arrangement classifier read the declaration, so request `y` landed INSIDE request 0's slab.
    /// MEASURED before the fix: a rank-3 `[y,in,out]` kernel emitted a per-`y` start stride of `in·out`
    /// whether or not `in`/`out` was declared larger, byte for byte.
    #[kani::proof]
    fn declared_physical_extent_sets_the_address_stride() {
        let swept_in: usize = kani::any();
        let out: usize = kani::any();
        let phys_in: usize = kani::any();
        let y: usize = kani::any();
        kani::assume(swept_in >= 1 && swept_in <= 4);
        kani::assume(out >= 1 && out <= 4);
        // The only direction the declaration means anything: the view is a WINDOW into a LARGER
        // allocation (`matmul_opspec_batched_off` refuses `phys < swept` at `cargo build`).
        kani::assume(phys_in >= swept_in && phys_in <= 8);
        kani::assume(y >= 1 && y <= 3);
        let swept = [4usize, swept_in, out];
        let declared = [None, Some(phys_in), None];
        let active = [true, true, true];
        // Stick is `out` — the last dim of a rank-3 view, so this is the row-major (`Flat`) fold, the
        // residency `matmul_opspec_batched_off`'s 3-D per-batch KERNEL actually gets.
        let e = DeviceExtents::of_view(&swept, &declared, 2, &active, Span::Materialized);
        let sl = StickLayout::for_view(e.dims(), 2);
        let step = sl.off_view(&e, &[y, 0, 0]) - sl.off_view(&e, &[0, 0, 0]);
        // THE LAW: one step along `y` crosses the DECLARED slab, not the swept window.
        assert!(step == y * phys_in * out);
        // AND THEREFORE the aliasing this makes unconstructable: request `y`'s corner is at or past the
        // end of request 0's window, and STRICTLY past it whenever the allocation really is larger.
        // Without the declaration `step == y * swept_in * out`, which is inside the window — the RED.
        assert!(step >= y * swept_in * out);
        assert!(phys_in == swept_in || step > y * swept_in * out);
    }

    /// GREEN — OPT-IN INERTNESS (the byte-identity lock for wiring both call sites through the type):
    /// with NOTHING declared, [`DeviceExtents::of_view`] reproduces exactly the extent vector each site
    /// hand-rolled before. `Span::Swept` is the identity on the iteration shape; `Span::Materialized`
    /// collapses a non-ranged NON-STICK dim to one row and touches nothing else. So no baked address
    /// moves until a call site opts in.
    #[kani::proof]
    fn undeclared_extents_reproduce_the_hand_rolled_vectors() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let c: usize = kani::any();
        kani::assume(a >= 1 && a <= 8 && b >= 1 && b <= 8 && c >= 1 && c <= 8);
        let stick: usize = kani::any();
        kani::assume(stick < 3);
        let act: [bool; 3] = kani::any();
        let swept = [a, b, c];
        let declared = [None, None, None];
        // Swept: every dim at its iteration extent, regardless of which dims are ranged — what
        // `view_stick_layout` folded by hand.
        assert!(
            DeviceExtents::of_view(&swept, &declared, stick, &act, Span::Swept).dims() == swept
        );
        // Materialized: exactly `if d != stick && !active { 1 } else { swept }` — what `per_core_addr`
        // folded by hand.
        let m = DeviceExtents::of_view(&swept, &declared, stick, &act, Span::Materialized);
        for i in 0..3 {
            let want = if i != stick && !act[i] { 1 } else { swept[i] };
            assert!(m.dims()[i] == want);
        }
    }

    /// GREEN — WIRED-ADDRESS INJECTIVITY (the #50 aliasing class): for any rank-2 view + stick_idx,
    /// DISTINCT logical corners map to DISTINCT device offsets through `off_view`. Self-contained on
    /// `StickLayout` (not via the free dev_off) ⇒ a regression in `for_view`/`off_view` is caught here.
    /// So no two cores addressing this tensor can ever write the same cell.
    #[kani::proof]
    fn off_view_injective_rank2() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let stick: usize = kani::any();
        kani::assume(a >= 1 && a <= 6 && b >= 1 && b <= 130);
        kani::assume(stick <= 2);
        let (i1, j1): (usize, usize) = (kani::any(), kani::any());
        let (i2, j2): (usize, usize) = (kani::any(), kani::any());
        kani::assume(i1 < a && i2 < a && j1 < b && j2 < b);
        kani::assume(!(i1 == i2 && j1 == j2));
        let dims = [a, b];
        let sl = StickLayout::for_view(&dims, stick);
        assert!(sl.off_view(&ext(&dims), &[i1, j1]) != sl.off_view(&ext(&dims), &[i2, j2]));
    }

    /// GREEN — WIRED-ADDRESS INJECTIVITY, rank-3 (head-major / trailing-singleton views ⇒ Flat fold).
    #[kani::proof]
    fn off_view_injective_rank3() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let c: usize = kani::any();
        let stick: usize = kani::any();
        kani::assume(a >= 1 && a <= 4 && b >= 1 && b <= 8 && c >= 1 && c <= 8);
        kani::assume(stick <= 3);
        let (i1, j1, k1): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
        let (i2, j2, k2): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
        kani::assume(i1 < a && i2 < a && j1 < b && j2 < b && k1 < c && k2 < c);
        kani::assume(!(i1 == i2 && j1 == j2 && k1 == k2));
        let dims = [a, b, c];
        let sl = StickLayout::for_view(&dims, stick);
        assert!(sl.off_view(&ext(&dims), &[i1, j1, k1]) != sl.off_view(&ext(&dims), &[i2, j2, k2]));
    }

    /// GREEN — WIRED-ADDRESS FOOTPRINT BOUND (the OOB-write class): every `off_view` for a rank-2 view
    /// lands within `rows * pad64(cols)` — the device bytes SegLayout allocates per tensor — so the
    /// baked address never runs past the tensor's footprint into a neighbor. (`a*pad64(b)` upper-bounds
    /// both the stick-blocked and the flat fold `i*b+j < a*b <= a*pad64(b)`.)
    #[kani::proof]
    fn off_view_in_footprint_rank2() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let stick: usize = kani::any();
        kani::assume(a >= 1 && a <= 6 && b >= 1 && b <= 130);
        kani::assume(stick <= 2);
        let (i, j): (usize, usize) = (kani::any(), kani::any());
        kani::assume(i < a && j < b);
        let dims = [a, b];
        let sl = StickLayout::for_view(&dims, stick);
        assert!(sl.off_view(&ext(&dims), &[i, j]) < a * pad64(b));
    }

    /// GREEN — WIRED-ADDRESS FOOTPRINT BOUND, rank-3 (Flat fold < prod(dims)).
    #[kani::proof]
    fn off_view_in_footprint_rank3() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let c: usize = kani::any();
        let stick: usize = kani::any();
        kani::assume(a >= 1 && a <= 4 && b >= 1 && b <= 8 && c >= 1 && c <= 8);
        kani::assume(stick <= 3);
        let (i, j, k): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
        kani::assume(i < a && j < b && k < c);
        let dims = [a, b, c];
        let sl = StickLayout::for_view(&dims, stick);
        assert!(sl.off_view(&ext(&dims), &[i, j, k]) < a * b * c);
    }

    /// GREEN — CLASSIFIER TOTALITY (rank-4): `per_core_addr` is general-rank, so prove `for_view`
    /// /`off_view` is byte-identical to the free `dev_off` for rank-4 views too (paged-KV / any future
    /// rank-4 tensor that reaches the central addresser) — no rank silently mis-classifies.
    #[kani::proof]
    fn for_view_off_equals_dev_off_rank4() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let c: usize = kani::any();
        let e: usize = kani::any();
        let stick: usize = kani::any();
        let (i, j, k, l): (usize, usize, usize, usize) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        kani::assume(a >= 1 && a <= 3 && b >= 1 && b <= 4 && c >= 1 && c <= 4 && e >= 1 && e <= 4);
        kani::assume(i < a && j < b && k < c && l < e);
        kani::assume(stick <= 4);
        let dims = [a, b, c, e];
        let corner = [i, j, k, l];
        let sl = StickLayout::for_view(&dims, stick);
        assert!(sl.off_view(&ext(&dims), &corner) == dev_off(&dims, stick, &corner));
    }

    /// GREEN — WIRED-ADDRESS INJECTIVITY (rank-4): distinct corners → distinct offsets for rank-4 too.
    #[kani::proof]
    fn off_view_injective_rank4() {
        let a: usize = kani::any();
        let b: usize = kani::any();
        let c: usize = kani::any();
        let e: usize = kani::any();
        let stick: usize = kani::any();
        kani::assume(a >= 1 && a <= 2 && b >= 1 && b <= 3 && c >= 1 && c <= 3 && e >= 1 && e <= 3);
        kani::assume(stick <= 4);
        let (i1, j1, k1, l1): (usize, usize, usize, usize) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        let (i2, j2, k2, l2): (usize, usize, usize, usize) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        kani::assume(i1 < a && i2 < a && j1 < b && j2 < b && k1 < c && k2 < c && l1 < e && l2 < e);
        kani::assume(!(i1 == i2 && j1 == j2 && k1 == k2 && l1 == l2));
        let dims = [a, b, c, e];
        let sl = StickLayout::for_view(&dims, stick);
        assert!(
            sl.off_view(&ext(&dims), &[i1, j1, k1, l1])
                != sl.off_view(&ext(&dims), &[i2, j2, k2, l2])
        );
    }

    /// GREEN — K/V-WIRING REGRESSION LOCK: the correctness-critical attention cache offsets ARE the
    /// systolic `StickLayout::kernel` addresses. `kcache_kt_write_offset` == `kernel(hd,cap).dev_off
    /// (d,slot)` (Kᵀ `[hd,cap]` sticked on cap); `vcache_write_offset` == `kernel(cap,hd).dev_off
    /// (slot,d)` (V `[cap,hd]` sticked on hd). Fails the build if the K/V path ever reverts to
    /// hand-rolled stick math instead of the layout.
    #[kani::proof]
    fn kv_cache_offsets_locked_to_sticklayout() {
        let hd: usize = 64; // the deployed PATH B (hd == stick)
        let cap: usize = kani::any();
        kani::assume(cap >= 64 && cap <= 512 && cap % 64 == 0);
        let slot: usize = kani::any();
        let d: usize = kani::any();
        kani::assume(slot < cap && d < hd);
        assert!(
            kcache_kt_write_offset(slot, d, hd, cap, 64)
                == StickLayout::kernel(hd, cap).dev_off(d, slot)
        );
        assert!(
            vcache_write_offset(slot, d, hd, cap, 64)
                == StickLayout::kernel(cap, hd).dev_off(slot, d)
        );
    }

    /// GREEN — the infallible typed Kernel ctor (used by the emitter's K/V-cache guard) yields a
    /// `Kernel`-kind handle and addresses BYTE-IDENTICALLY to the untyped `StickLayout::kernel`. So
    /// routing the cache guard through `Stk::<KernelTag>::kernel(...).dev_off(...)` changes zero
    /// addresses while adding the compile-time `KernelTag`.
    #[kani::proof]
    fn stk_kernel_handle_addresses_as_kernel() {
        let k: usize = kani::any();
        let n: usize = kani::any();
        kani::assume(k >= 1 && k <= 512 && n >= 64 && n <= 512 && n % 64 == 0);
        let r: usize = kani::any();
        let c: usize = kani::any();
        kani::assume(r < k && c < n);
        let h = Stk::<KernelTag>::kernel(k, n, "t");
        assert!(h.layout().kind == StickKind::Kernel);
        assert!(h.dev_off(r, c) == StickLayout::kernel(k, n).dev_off(r, c));
    }

    /// FAIL-FIRST (RED at rows>1) — THE TIME-TILE ALIASING BUG, MODELED: when a STICK-BLOCKED output is
    /// time-tiled on its stick ("out") dim, the emitter's FLAT per-trip advance (`out_per_time * 1`,
    /// device_stride_out=1) must equal the true stick-blocked advance `StickLayout::dev_off(0,
    /// out_per_time)` (= out_per_time*rows for whole-stick trips). RED: Kani finds rows>1 where flat <
    /// stick ⇒ trips advance too little ⇒ ALIAS (the `tiled_trips_alias` refusal). GREEN only at rows==1
    /// (flat==stick — the proven lm_head case). The fix routes the advance through `StickLayout`.
    #[kani::proof]
    fn time_tile_flat_stride_aliases_at_multirow() {
        let rows: usize = kani::any();
        let out_per_time: usize = kani::any();
        let out_full: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 8);
        kani::assume(out_per_time >= 64 && out_per_time <= 256 && out_per_time % 64 == 0);
        kani::assume(out_full > out_per_time && out_full <= 512 && out_full % 64 == 0);
        let flat_advance = out_per_time; // current emitter: out_per_time * device_stride_out(=1)
        let stick_advance = StickLayout::row_blocked(rows, out_full).dev_off(0, out_per_time);
        assert!(flat_advance == stick_advance);
    }

    /// GREEN — THE FIX INVARIANT: the `StickLayout`-derived per-trip advance EXACTLY tiles the trips —
    /// the advance `dev_off(0, out_per_time)` equals the trip's own footprint (`rows * out_per_time`
    /// elements) for ANY rows ⇒ consecutive trips are DISJOINT and gapless ⇒ no `tiled_trips_alias`.
    #[kani::proof]
    fn time_tile_sticklayout_stride_tiles_disjoint() {
        let rows: usize = kani::any();
        let out_per_time: usize = kani::any();
        let out_full: usize = kani::any();
        kani::assume(rows >= 1 && rows <= 8);
        kani::assume(out_per_time >= 64 && out_per_time <= 256 && out_per_time % 64 == 0);
        kani::assume(out_full > out_per_time && out_full <= 512 && out_full % 64 == 0);
        let advance = StickLayout::row_blocked(rows, out_full).dev_off(0, out_per_time);
        let trip_footprint = rows * out_per_time; // rows × the trip's `out` slice width
        assert!(advance == trip_footprint);
    }
}
