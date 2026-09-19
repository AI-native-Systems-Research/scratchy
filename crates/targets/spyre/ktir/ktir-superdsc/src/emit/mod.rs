// SPDX-License-Identifier: Apache-2.0
//! THE SuperDSC EMITTER — the typed op family ([`OpSpec`]/[`TensorArg`]) lowered into the wire
//! [`SdscOp`], plus every `assemble_*` entry point the assemblers and the lowering call.
//!
//! ⭐ THE CONSUMER HALF OF THE PROVEN EMITTER, WITHOUT ITS PRODUCER HALF. Two contiguous ranges of
//! the on-card-proven `lower_subtile_tape_to_superdsc` moved here verbatim: `pointwise_iter_space`
//! through `op_func_from_str` (the pointwise/restickify/transpose assemblers, [`ArgBinding`],
//! [`EmittedOp`], [`emit_sdsc`]/[`emit_sdsc_tiled`], `resolve_seg_base`, [`pw1`]/[`pw2`]) and
//! [`bmm_site`] through [`check_pointwise_cols`] (the SFP constant table and the SEN169 encoder).
//! What stayed with the producer is what BUILDS a plan from a graph — `compute_bundle_layout`,
//! `audit_layout_addresses`, `bake_layout`, the tape walk and the bake queue — which is the seam the
//! whole extraction turns on. Line citations to `ibm/main` throughout are the port's provenance and
//! stay.
//!
//! ⛔ [`bmm_site::TapeLoweringSite::witness`] IS `pub(in crate::emit)`, AND THAT IS WHY THIS IS A
//! DIRECTORY MODULE. Its doc has always claimed "the attention emitter cannot mint one"; while the
//! emitter and `attn.rs` shared one crate that was FALSE — a `pub(crate)` mint is callable from
//! `ir::bridge::tiled_op_sdsc_op::attn`. `crate::emit` is an ancestor of both real callers
//! ([`lower_ktir_to_superdsc`] and [`ktir_matmul_fp8`]) and NOT of the attention emitter, so the
//! seal is finally the visibility it always claimed to be. Checked by negative control: a probe that
//! calls `witness()` from `attn.rs` fails with `E0624`, and compiled under the old `pub(crate)`.

use crate::ir::bridge::tiled_op_sdsc_op::{
    pointwise_broadcast_opspec_from_tile, pointwise_opspec_from_tile,
};
use crate::ktir_node::KtirNode;
use crate::placement::{BundleLayout, SegRole, align128};
use crate::sdsc_abstract::{FlatTag, KindTag, RowBlockedTag, StickKind, StickLayout, Stk};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::{
    Allocation, AnyTensorArg, ArgView, DataFormat, DeviceTileLayout, Df, Fp8, Fp16, Fp32, ItDim,
    MAX_CORES, MaxCores, OpFunc, OpInfo, OpSpec, Role, Scale, SdscFoldSet, StickExtent, TensorArg,
    WorkPlan, assert_df_stick_multiple,
};
use crate::wire::{
    AddrFold, AllocNode, ComputeAttrs, ComputeOp, Dsc, FoldFunc, FoldProp, IterSpace, LabeledDs,
    LayoutInfo, MemOrg, SdscFolds, SdscOp, StageParam, build_coordinates, core_dsc_schedule,
    gen_coord_info_value, segment_base,
};
use crate::work::{FP16_ELEMS_PER_STICK, core_to_wk_slice, distribute_cores};
use std::collections::BTreeMap;

/// main's `lower_matmul_node` arity-3 branch — the W8A8 fp8 matmul descriptors, reached from the
/// facts the KTIR states. See its own header.
pub mod ktir_matmul_fp8;
/// KTIR → SuperDSC: `ibm/main`'s eight `lower_*_node` bodies with their input door changed. The two
/// that cross a model-geometry const door ([`lower_ktir_to_superdsc::attn_at`],
/// [`lower_ktir_to_superdsc::rope_at`]) are public generic entry points the CALLER instantiates.
pub mod lower_ktir_to_superdsc;

/// ONE FUNCTION, MANY OPS — the door for a producer whose KTIR function is a whole kernel, which is
/// the shape IBM's C++ reference producer emits (3 `linalg.matmul` in one `func.func` for a SwiGLU
/// MLP, 12 for a decoder layer). Purely additive: it changes no entry point, no assembler and not
/// `split_out`. See the module docs.
pub mod whole_function;

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
) -> crate::sdsc_abstract::StickLayout {
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
    let extents = crate::sdsc_abstract::DeviceExtents::of_view(
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
        crate::sdsc_abstract::Span::Swept,
    );
    let host_size_us: Vec<usize> = extents.dims().to_vec();
    let stick_idx = view
        .layout
        .iter()
        .position(|&d| d == view.stick)
        .unwrap_or(usize::MAX);
    if view.row_blocked && !host_size_us.is_empty() {
        let feat: usize = host_size_us[1..].iter().product::<usize>().max(1);
        return crate::sdsc_abstract::StickLayout::row_blocked_df(host_size_us[0], feat, view.df);
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
    crate::sdsc_abstract::StickLayout::for_view_df(&host_size_us, stick_idx, view.df)
}

fn per_core_addr(
    seg_base: u64,
    view: &ArgView<'_>,
    // THE one device layout, handed in by `emit_sdsc` — NOT re-derived here — so the per-core START this
    // computes and the on-card WALK `build_coordinates` computes read the SAME `StickLayout` (task #11).
    sl: &crate::sdsc_abstract::StickLayout,
    plan: &WorkPlan,
    wk_slice: &BTreeMap<String, BTreeMap<&'static str, crate::superdsc_opspec::SliceIndex>>,
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
    let extents = crate::sdsc_abstract::DeviceExtents::of_view(
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
        crate::sdsc_abstract::Span::Materialized,
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
                .unwrap_or(crate::superdsc_opspec::SliceIndex::UNSPLIT);
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
/// [`crate::ir::island::tile_op::pointwise_lx_resident_generic`] (`TileOp`'s dispatch target for
/// `TileOpKind::PointwiseOrReduce`), Kani-proven equal to this formula at every symbolic extent. Kept
/// as a free function so the 6 existing `time_tile_for_lx(|p,opt| pointwise_lx_resident(…), …)` call
/// sites are unchanged; new call sites should build a `TileOp` instead.
fn pointwise_lx_resident(plan: &WorkPlan, n_operands: u64, out_per_time: u32, df: Df) -> u64 {
    crate::ir::island::tile_op::pointwise_lx_resident_generic(plan, n_operands, out_per_time, df)
}

/// Build the typed [`OpSpec`] for ONE shape-preserving ELEMENTWISE op over
/// `[rows, cols]`. All operands share the rank-3 `[mb,out,y]` OUTPUT layout
/// (`cols`/`out` is the stick axis); scales all active.
///
/// Thin wrapper: builds the `[mb,out,y]` `TileOp` (the TileIR declaration for this shape) and hands
/// it to [`pointwise_opspec_from_tile`] — the SAME two-step split every other caller of that function
/// goes through, whether their `TileOp` came from here or from
/// `subtile_tape_to_tile_ir::node_to_single_tile_op` (the live per-node path).
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
    let tile_op = crate::ir::island::tile_op::TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands },
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
pub(crate) fn pointwise_broadcast_opspec(
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
    let tile_op = crate::ir::island::tile_op::TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands },
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
/// emitter calls it `[mb=1, out=64-stick, y=1]` (this emitter's rank-3) or `[mb=64-stick]`
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
    // ONLY remaining byte-difference between this emitter's compiled rsqrt program and torch-
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
    rows: crate::sdsc_abstract::RowCount,
    cols: crate::sdsc_abstract::BlockCols,
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
        crate::addr::DevOff::ZERO,
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
    rows: crate::sdsc_abstract::RowCount,
    cols: crate::sdsc_abstract::BlockCols,
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
        crate::addr::DevOff::ZERO,
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
    rows: crate::sdsc_abstract::RowCount,
    cols: crate::sdsc_abstract::BlockCols,
    inputs: &[EwOperand<'_>],
    o: &Stk<O>,
    // TYPED, like the inputs' `In::sliced`. A write offset is a DEVICE address, so it must come from
    // a nest that knows the tensor's extents; the only constructor is `View::dev`. When this was a
    // bare `u32` a caller could pass a flat column index, and one did -- the split MLP intermediate
    // wrote its second column block at element 8192 instead of `(8192/64)*(rows*64)`, correct at one
    // row and wrong at every prefill. The reads were already typed and were already right.
    out_offset: crate::addr::DevOff,
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
    // The guard's op set is [`SFP_SPLIT_TRANSCENDENTALS`], shared with the tiled twin of this fn.
    //
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
    rows: crate::sdsc_abstract::RowCount,
    cols: crate::sdsc_abstract::BlockCols,
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
pub fn materialized_bytes(v: &crate::superdsc_opspec::ArgView, iter: &WorkPlan) -> u64 {
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
pub fn declared_axis_block_steps(v: &crate::superdsc_opspec::ArgView, iter: &WorkPlan) -> Vec<u64> {
    let stick_idx = v
        .layout
        .iter()
        .position(|&d| d == v.stick)
        .unwrap_or(usize::MAX);
    let ext = crate::sdsc_abstract::DeviceExtents::of_view(
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
        crate::sdsc_abstract::Span::Materialized,
    );
    let layout = crate::sdsc_abstract::StickLayout::for_view_df(ext.dims(), stick_idx, v.df);
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
pub fn touched_blocks(v: &crate::superdsc_opspec::ArgView, iter: &WorkPlan) -> TouchedBlocks {
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

#[derive(Clone)]
pub struct EmittedOp {
    pub op_name: String,
    /// The SuperDSC descriptor, when this op has one.
    ///
    /// ⛔ `None` FOR A KTIR OP, AND THAT IS THE ANSWER — not an empty descriptor. A KTIR op carries
    /// its PROGRAM; a descriptor is what dxp schedules from, and there is nothing to schedule.
    pub op: Option<SdscOp>,
    /// ⭐⭐⭐ THIS OP'S PROGRAM — the KTIR the lowering constructed for it, and what the launch runs.
    ///
    /// `None` only for an op the KTIR lowering declines; a bundle built from those would be short a
    /// program, which is why bundle creation refuses rather than skipping.
    pub ktir: Option<KtirNode>,
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
    /// persist. Classified `GroupKind::Slab` — a DISTINCT kind from `Slot` (its `slab_stride_bytes`
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
    pub kv_fold_rows: crate::sdsc_abstract::FoldRowRegime,
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
    /// `GroupKind::SlotSolo` → its OWN singleton group, launched with its baked slot doff
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
    /// The SuperDSC descriptor this op was built from.
    ///
    /// ⛔ THE SuperDSC PATH REQUIRES ONE, AND SAYS SO. Everything that reads a descriptor reads a
    /// fold, a core count or a baked address — the parts with no KTIR counterpart — so reaching one
    /// on a KTIR op is a bug in the caller, not a case to handle.
    pub fn dsc(&self) -> &SdscOp {
        self.op
            .as_ref()
            .unwrap_or_else(|| panic!("{}: a KTIR op has no SuperDSC descriptor", self.op_name))
    }

    /// An op that carries ONLY its program.
    ///
    /// ⛔ EVERY OTHER FIELD IS THE SuperDSC DESCRIPTOR'S. `arg_bindings`, `affine_strides`, the
    /// `slot_stride_bytes` / `kv_*` classification — all of it is baked addressing and folds, which
    /// have no KTIR counterpart. A KTIR op is its func and the tensors its parameters carry.
    pub fn bare(op_name: String) -> EmittedOp {
        EmittedOp {
            op_name,
            op: None,
            ktir: None,
            arg_bindings: Vec::new(),
            time: 1,
            affine_strides: Vec::new(),
            slot_stride_bytes: 0,
            slab_write: false,
            slab_stride_bytes: 0,
            slot_no_fuse: false,
            kv_page_fold: false,
            kv_batched_requests: false,
            kv_fold_rows: Default::default(),
            kv_page_slots: 0,
            kv_request: 0,
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

    /// A by-value copy, for building a trimmed op list (see `ops_for_grouping`). `EmittedOp` is not
    /// `Clone` by design — copies of a baked op are almost always a mistake — so this is explicit.
    pub fn shallow_copy(&self) -> EmittedOp {
        EmittedOp {
            op_name: self.op_name.clone(),
            op: self.op.clone(),
            ktir: self.ktir.clone(),
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
            op: Some(op),
            // Set by the arm that constructs the node's program.
            ktir: None,
            arg_bindings,
            time,
            affine_strides,
            slot_stride_bytes: 0,
            kv_page_fold: false,
            kv_batched_requests: false,
            kv_fold_rows: crate::sdsc_abstract::FoldRowRegime::WholeBatch,
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
            let advance_elems =
                crate::sdsc_abstract::StickLayout::row_blocked_df(full_rows, out_full, v.df)
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
        // 1. A real caller-bound buffer → its global placement. The id comes from the layout's own
        //    mint table, NOT from picking the digits back out of the spelling.
        if let Some(crate::place::PlaceId::Act(tid)) = l.id_of(name)
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
            // This lowering runs at BUILD time: every intermediate it invents is known when it is
            // invented, so a tensor that reaches its first ACCESS undeclared is a lowering that
            // forgot to say what it was making — not a case to serve.
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
            // ⚠️ These are STILL NOT COLOURED. A caller's slot-colouring pass runs over its own
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
                Some(crate::place::PlaceId::Act(tid)) => l.placements.get(&tid).map(|p| p.size),
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
                crate::sdsc_abstract::DeviceExtents::of_view(
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
                    crate::sdsc_abstract::Span::Swept,
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
        |arg_idx: usize, op_func: crate::superdsc_opspec::EpilogueOpFunc| {
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
    pub fn sliced(h: &'a Stk<K>, col_offset: crate::addr::DevOff) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: false,
            col_offset: col_offset.into_raw_elems(),
        }
    }
    /// [`col`] (per-row scalar, out-broadcast over the stick) PLUS a base `col_offset` — the per-head-block
    /// variant: a `[mqu,stk]` reduced scalar at `h·mqu·stk`, broadcast over the `[mqu,mqp]` score block.
    pub fn col_at(h: &'a Stk<K>, col_offset: crate::addr::DevOff) -> Self {
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
    pub fn mb_at(h: &'a Stk<K>, col_offset: crate::addr::DevOff) -> Self {
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
    rows: crate::sdsc_abstract::RowCount,
    cols: crate::sdsc_abstract::BlockCols,
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
        crate::addr::DevOff::ZERO,
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
    rows: crate::sdsc_abstract::RowCount,
    cols: crate::sdsc_abstract::BlockCols,
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
        crate::addr::DevOff::ZERO,
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
///
/// ⛔⛔⛔ BOTH EXTENTS ARE STICK WITNESSES, AND `mb` USED TO BE UNCHECKED. `out` is the INPUT's stick,
/// and it has always been routed through [`StickExtent`]. `mb` is the OTHER half of the OUTPUT's stick
/// — `emit_sdsc` overrides that role to the 8×8 inter-slice block `["out","mb"]`/`[8,8]`, so `mb` is a
/// stick extent of the output exactly as `out` is of the input, and it was the one extent this builder
/// accepted at any value. That is the omission the deleted generic op walk fell through: a
/// `linalg.transpose` of a per-step `[1, 64]` key is `mb = 1`, whose stick extent is 1, which the dxp
/// scheduler refuses (`L3DlOpsScheduler:1040`) — and the on-card ReStickify that was asked for that
/// shape anyway wrote the second Kᵀ stick wrong and garbled decode past 64 tokens. So `mb` is a
/// witness too, and a sub-stick transpose is a `cargo` `Err` rather than a shape the card sees.
///
/// ⚖️ 64, THOUGH THE `[8,8]` BLOCK ALONE WOULD ADMIT ANY MULTIPLE OF 8 — RECORDED, NOT APPLIED. The
/// only `mb` this tree has evidence for is a whole 64-fp16 stick: the shipped attention pads its row
/// count to a stick multiple and loops whole 64×64 tiles (`attn.rs`'s per-sub-block single-stick Kᵀ),
/// and every other extent in this emitter is a 64-stick. Whether `mb ∈ {8, 16, 24, 32, 40, 48, 56}`
/// bakes is answerable only against the golden `sdsc_interslicetranspose.json`, which is NOT in this
/// repo. Until it is, the guard is the extent we have evidence for, and widening it needs that fixture
/// — not this comment.
fn transpose_opspec(mb: u32, out: u32, in_name: &str, o_name: &str) -> Result<OpSpec, String> {
    let out_ext = StickExtent::<Fp16>::new(out)?;
    let mb_ext = StickExtent::<Fp16>::new(mb)?;
    let dims = vec![
        ItDim {
            name: "mb",
            size: mb_ext.elems(),
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
    let tile_op = crate::ir::island::tile_op::TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands: 2 },
        dims: dims.clone(),
        df: Df::Fp16,
    };
    // ⭐ THE TRANSPOSE'S OWN DIVIDER, NOT `distribute_cores`. This op's output stick is the 8×8
    // inter-slice block, so its per-core slice must be whole blocks on BOTH axes — a property
    // `distribute_cores` cannot serve, because it splits `mb` first and to the hilt for a MEASURED
    // reason of its own (see its doc: the flat row-major `(c+r)·eps` row-mixing, `nz=3968`) and every
    // reduce/pointwise/silu in the crate rides that order. `TileOp::tile` takes the divider as a
    // parameter, so pointing this ONE builder at
    // [`distribute_cores_transpose_blocks`](crate::work::distribute_cores_transpose_blocks) leaves
    // every other op's division — and bytes — untouched. The guard below re-checks the property on
    // whatever plan comes back, so the divider and the seal cannot drift apart.
    let tiled = tile_op
        .tile(
            MaxCores::<MAX_CORES>,
            crate::work::distribute_cores_transpose_blocks,
            "out",
        )
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    // ⛔⛔⛔ THE PER-CORE SLICE MUST BE WHOLE 8×8 OUTPUT BLOCKS — RE-CHECKED ON THE PLAN, NOT ASSUMED.
    //
    // The OUTPUT's stick is the 8×8 inter-slice block over (`out`, `mb`) — `emit_sdsc`'s
    // `is_transpose_out` arm, `stickSize_: [8, 8]`. A stick is the atomic unit a core moves, so if a
    // core's work slice ends part-way through a block, TWO cores own bytes of one output stick and
    // each writes a partial one. That is the failure class already on this tree's record (the second Kᵀ
    // stick written wrong; the shared-stick per-core init).
    //
    // MEASURED against the golden fixture at
    // `/Users/nickm/git/deeptools/ddc/ddl_templates/test/sdsc_interslicetranspose.json`, whose op is
    // this exact shape (`N_`: `mb_ 384`, `out_ 3072`, `y_ 1`, 32 cores):
    //
    //   fixture  `numWkSlicesPerDim_ {"out": 8, "mb": 4}`  ⇒ per-core (mb 96, out 384) — both 8-whole
    //   ours     `numWkSlicesPerDim_ {"mb": 32, "out": 1}` ⇒ per-core (mb 12, out 3072) — 12 is NOT
    //
    // `distribute_cores` splits `mb` FIRST and to the hilt (its own doc explains why, for the FLAT
    // row-major activations it was written for), which leaves `out` whole and hands each core a
    // fraction of an 8-row block for every `mb` that is not a multiple of 8·MAX_CORES. So the division
    // this op needs is the fixture's — both axes split at the block — and deriving that rule from ONE
    // fixture point is not something this builder guesses. It REFUSES instead, naming the extent and
    // the reference division, so the shape is a `cargo` error and not a descriptor whose cores collide.
    // Lifting it is a core-division change with that fixture as its oracle, not a widening here.
    const BLOCK: u32 = 8; // the `[8, 8]` output stick, `emit_sdsc`'s `is_transpose_out` arm
    let (per_mb, per_out) = (plan.per_core_extent("mb"), plan.per_core_extent("out"));
    if !per_mb.is_multiple_of(BLOCK) || !per_out.is_multiple_of(BLOCK) {
        return Err(format!(
            "transpose [{}, {}]: the core division gives each core (mb {per_mb}, out {per_out}) but \
             the output's stick is the {BLOCK}x{BLOCK} inter-slice block over (out, mb), so a \
             per-core extent that is not a whole multiple of {BLOCK} puts two cores inside one output \
             stick and each writes a partial one. This builder's `distribute_cores` splits `mb` alone \
             ({:?}); the golden `sdsc_interslicetranspose.json` splits BOTH axes at the block \
             (`numWkSlicesPerDim_ {{\"out\": 8, \"mb\": 4}}` for mb=384/out=3072 on 32 cores, i.e. \
             per-core mb 96 and out 384). Use an extent whose division lands on the block, or fix the \
             division against that fixture — this refuses rather than guessing a division law from one \
             fixture point.",
            mb_ext.elems(),
            out_ext.elems(),
            plan.splits(),
        ));
    }
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
    try_assemble_transpose(op_name, mb, out, in_name, o_name, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_transpose {op_name}: {e}"))
}

/// [`assemble_transpose`] AS A `Result` — the form a producer-facing entry point needs.
///
/// ⛔ THE PANIC IS RIGHT FOR THE ASSEMBLERS AND WRONG FOR THE DOOR. Every `assemble_*` here panics on a
/// refused opspec, and that is correct where they are called from: inside a `#[forward]` expansion a
/// panic IS the build error. But [`lower_ktir_to_superdsc::transpose`] is reached by a third-party KTIR
/// producer that has its own op to blame, and this builder makes TWO refusals a caller cannot
/// pre-check by inspecting extents — the stick law (which the lowering does re-state, so it can name
/// the padding alternative) and the CORE-DIVISION law, which depends on `distribute_cores`' answer for
/// the shape and so cannot be restated without duplicating the divider. So the fallible form exists and
/// the door uses it; `assemble_transpose` keeps its signature and its panic for every existing caller.
#[allow(clippy::too_many_arguments)]
pub fn try_assemble_transpose(
    op_name: &str,
    mb: u32,
    out: u32,
    in_name: &str,
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, String> {
    let op = transpose_opspec(mb, out, in_name, o_name)?;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout).map_err(|e| e.0)
}

/// ⭐⭐⭐ THE FIRST TESTS `OpFunc::Transpose` HAS EVER HAD, because until the [`Program::Transpose`]
/// entry point landed the assembler above had ZERO callers anywhere in the workspace — so
/// `emit_sdsc`'s 8×8 output-stick override (`is_transpose_out`) was code no test executed and no build
/// reached.
///
/// ⭐⭐⭐ THE ORACLE IS THE VENDOR'S OWN GOLDEN, AND IT IS ON DISK:
/// **`/Users/nickm/git/deeptools/ddc/ddl_templates/test/sdsc_interslicetranspose.json`** (off-repo, in
/// the `deeptools` checkout beside this one — the same tree `ddl_templates/` ships). Its op
/// (`"mul_78-Gelu"`) is one `interslicetranspose_fp16`, and every layout number asserted below is READ
/// OUT OF IT rather than out of our own emission:
///
/// ```text
/// "primaryDsInfo_" : { "INPUT"  : { layoutDimOrder_ ["mb","out","y"], stickDimOrder_ ["out"],       stickSize_ [64]  },
///                      "OUTPUT" : { layoutDimOrder_ ["mb","out","y"], stickDimOrder_ ["out","mb"],  stickSize_ [8,8] } }
/// "computeOp_" : [ { "exUnit" : "pt", "opFuncName" : "interslicetranspose_fp16", … } ]
/// "N_" : { "out_" : 3072, "mb_" : 384, "y_" : 1 }      "numCoresUsed_" : 32
/// "numWkSlicesPerDim_" : { "out" : 8, "mb" : 4, "y" : 1 }
/// ```
///
/// ⚖️ WHAT THE FIXTURE DOES **NOT** LICENSE. It carries `stickRepl_` (which this crate deliberately
/// never emits — see [`LayoutInfo`]) and `coreletFoldProp_ factor_ 2` against our 1, so this is NOT a
/// byte-parity comparison and no test below claims one. What it settles is the layout contract, the
/// wire name, the unit, and — the reason [`transpose_opspec`] now refuses its own fixture's shape — the
/// per-core division. Everything else here is a REFUSAL with a control.
#[cfg(test)]
mod transpose_tests {
    use super::*;
    use crate::ktir_node::{BufferId, KtirNode, Program};
    use ktir_core::arena::Arena;
    use ktir_core::attrkey::AttrKey;
    use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
    use ktir_core::irtype::IrType;
    use ktir_core::opkind::OpKind;

    /// A row extent these tests share. It USED to be load-bearing: while the transpose rode
    /// `distribute_cores` (which splits `mb` alone, to the hilt) only `mb` a multiple of `8 · MAX_CORES`
    /// landed on the 8×8 block, and 256 was the smallest such extent. The transpose now has its own
    /// block-aware divider ([`crate::work::distribute_cores_transpose_blocks`]), so any stick-aligned
    /// extent lands on the block and 256 is merely a convenient one — kept so these descriptor
    /// assertions stay byte-comparable with what they pinned before.
    const MB: u32 = 256;

    /// The `dscs_[0]` entry of an emitted op, by name.
    fn dsc(op: &EmittedOp, name: &str) -> Dsc {
        op.op
            .as_ref()
            .expect("a transpose emits a descriptor")
            .dscs_[0][name]
            .clone()
    }

    // ── (i) the OUTPUT override, which is the transpose ──────────────────────────────────────────

    /// ⭐⭐⭐ THE TRANSPOSE IS THE OUTPUT'S STICK ORDER, AND NOTHING ELSE SAYS SO — asserted against
    /// the golden's own `primaryDsInfo_`, quoted in this module's header.
    ///
    /// The fixture gives the OUTPUT role the 8×8 inter-slice block `["out","mb"]`/`[8,8]` and the INPUT
    /// the plain single stick `["out"]`/`[64]`, with the SAME `layoutDimOrder_` on both. That asymmetry
    /// IS the reorder, so if `emit_sdsc`'s `is_transpose_out` arm were dropped the descriptor would be
    /// an identity copy that still parsed, still baked, and computed the wrong thing. Until this test
    /// there was no caller and no coverage of that arm at all.
    #[test]
    fn the_transpose_layout_matches_the_golden_interslicetranspose_fixture() {
        let mut sid = 0i64;
        let op = assemble_transpose("tr", MB, 128, "t1", "t2", &mut sid, None);
        let d = dsc(&op, "tr");
        let inp = &d.primaryDsInfo_["INPUT"];
        let out = &d.primaryDsInfo_["OUTPUT"];
        // Verbatim from the fixture's OUTPUT entry.
        assert_eq!(out.stickDimOrder_, vec!["out", "mb"]);
        assert_eq!(out.stickSize_, vec![8, 8]);
        // Verbatim from the fixture's INPUT entry.
        assert_eq!(inp.stickDimOrder_, vec!["out"]);
        assert_eq!(inp.stickSize_, vec![64]);
        // The fixture states `["mb","out","y"]` for BOTH roles.
        assert_eq!(inp.layoutDimOrder_, vec!["mb", "out", "y"]);
        assert_eq!(out.layoutDimOrder_, vec!["mb", "out", "y"]);
    }

    /// `exUnit "pt"` + `opFuncName "interslicetranspose_fp16"` — verbatim from the fixture's
    /// `computeOp_[0]`, and independently from `OpFunc::name`/`OpFunc::ex_unit`. An `Unrecognized
    /// opFunc` DtException is the on-card failure this pins.
    #[test]
    fn the_emitted_op_is_interslicetranspose_fp16_on_the_pt_unit() {
        let mut sid = 0i64;
        let op = assemble_transpose("tr", MB, 128, "t1", "t2", &mut sid, None);
        let d = dsc(&op, "tr");
        let c = &d.computeOp_[0];
        assert_eq!(c.opFuncName, "interslicetranspose_fp16");
        assert_eq!(c.exUnit, "pt");
        // NOT fixture-backed: the golden's computeOp carries no `attributes_` at all. This is our
        // emitter's own field, which `ComputeOp`'s doc ties to torch-spyre's real frontend; it is
        // asserted as a regression pin on the fp16 marker, not as parity with the golden.
        assert_eq!(c.attributes_.dataFormat_, "SEN169_FP16");
    }

    /// The op's ITERATION SPACE is the fixture's, at the fixture's own extents: `mb_ 384`, `out_ 3072`,
    /// `y_ 1`. This used to be assertable only through the refusal message, because the fixture's own
    /// shape could not reach a descriptor at all; with the block-aware divider it builds, so the
    /// iteration space is now read off the real plan.
    #[test]
    fn the_iteration_space_matches_the_fixture_at_the_fixture_extents() {
        let spec = transpose_opspec(384, 3072, "t1", "t2")
            .expect("the fixture's own shape divides on the block now");
        assert_eq!(spec.iter.extent("mb"), 384, "the fixture's `mb_`");
        assert_eq!(spec.iter.extent("out"), 3072, "the fixture's `out_`");
        assert_eq!(spec.iter.extent("y"), 1, "the fixture's `y_`");
    }

    /// ⭐ EXHAUSTIVE: a transpose declares exactly TWO tensors, one per role, and in that order. The
    /// override is `primary.entry(role).or_insert`, so two roles that collapsed to one would silently
    /// skip it (the builder's own doc says so) — this is the assertion that they do not. The fixture
    /// likewise carries exactly `INPUT` and `OUTPUT`.
    #[test]
    fn a_transpose_declares_exactly_one_input_and_one_output_role() {
        let mut sid = 0i64;
        let op = assemble_transpose("tr", MB, 128, "t1", "t2", &mut sid, None);
        let d = dsc(&op, "tr");
        assert_eq!(
            d.primaryDsInfo_.keys().copied().collect::<Vec<_>>(),
            vec!["INPUT", "OUTPUT"],
            "exactly two roles, so the OUTPUT override is reached"
        );
        assert_eq!(d.labeledDs_.len(), 2, "one input tensor, one output tensor");
        assert_eq!(
            d.labeledDs_.iter().map(|l| l.dsType_).collect::<Vec<_>>(),
            vec!["INPUT", "OUTPUT"]
        );
    }

    /// ⭐⭐⭐ THE DIVERGENCE THE FIXTURE FOUND, NOW CLOSED — AND THIS IS THE PLACE ITS OWN DOC ASKED
    /// FOR THE NEW BEHAVIOUR TO BE RECORDED.
    ///
    /// This test used to pin a REFUSAL: at the golden's shape (`mb_ 384`, `out_ 3072`, 32 cores)
    /// `distribute_cores` split `mb` alone to 32, giving per-core (mb 12, out 3072), and 12 is not a
    /// multiple of 8 — two cores inside one 8×8 output stick. Its doc said "when the division is fixed
    /// against the fixture this test fails loudly and is the place to record the new behaviour". It did,
    /// and this is that record.
    ///
    /// ⛔ WHAT IS ASSERTED IS THE PROPERTY AND THE CORE COUNT, NOT THE FIXTURE'S PARTICULAR PAIR.
    /// Correctness is "every per-core extent is a whole 8-block"; SEVERAL divisions satisfy it at 32
    /// cores, and the fixture's `{mb: 4, out: 8}` (per-core 96×384) is one of them, not the only one. So
    /// this pins that we reach the fixture's CORE COUNT with the block property intact, and separately
    /// that the fixture's own pair is admissible under that property. Asserting our particular pair
    /// equals the fixture's would be fitting one point — the thing the guard's own text warns against.
    #[test]
    fn the_fixtures_own_shape_now_divides_on_the_block_and_uses_its_core_count() {
        let spec = transpose_opspec(384, 3072, "t1", "t2")
            .expect("the block-aware divider serves the golden's own shape");
        for dim in ["mb", "out"] {
            let per = spec.iter.per_core_extent(dim);
            assert!(
                per.is_multiple_of(8),
                "per-core {dim} = {per} is not a whole 8-block of the output's [8, 8] stick"
            );
        }
        assert_eq!(
            spec.iter.cores_used().get(),
            32,
            "the fixture uses all 32 cores at this shape and so must we — a block-valid division that \
             quietly dropped to 24 would pass the property and lose a quarter of the machine"
        );
        // AND the fixture's own pair is admissible under the same property — checked directly, so the
        // claim in the doc above is a measurement rather than an assertion.
        assert!(
            (384u32 / 4).is_multiple_of(8) && (3072u32 / 8).is_multiple_of(8),
            "the fixture's own {{mb: 4, out: 8}} must satisfy the property this builder enforces"
        );
    }

    /// ⛔ AND THE SHAPE THAT ONCE FAILED, AS THE REGRESSION IT NOW IS: `[64, 128]` is the decoder's
    /// `tl.dot(p, v)` transpose, and under `distribute_cores` it gave per-core (mb 2, out 128) and was
    /// refused. It is the reason this divider exists.
    ///
    /// ⭐ NOTE THE STICK AXIS CAPS THE CORE COUNT AT 16, NOT 32, and reasoning in elements gets this
    /// wrong. `out` is the stick dim, so it divides over whole 64-element sticks — `out = 128` is TWO
    /// sticks and cannot split beyond 2 however many cores are free (half a stick per core is dxp
    /// `L3DlOpsScheduler:1070`). With `mb` split 8 that is 16 cores, per-core (mb 8, out 64), both whole
    /// blocks. An "obvious" `{mb: 8, out: 4}` for 32 cores would hand each core 32 columns — half a
    /// stick.
    #[test]
    fn the_decoders_own_transpose_shape_divides_and_the_stick_axis_caps_the_cores() {
        let spec =
            transpose_opspec(64, 128, "t1", "t2").expect("the shape both decoders need");
        assert_eq!(spec.iter.per_core_extent("mb"), 8, "8 rows = one whole block");
        assert_eq!(
            spec.iter.per_core_extent("out"),
            64,
            "64 columns = one whole 64-element stick, which is 8 whole blocks"
        );
        assert_eq!(
            spec.iter.cores_used().get(),
            16,
            "16, not 32: the stick axis has only two sticks to give"
        );
    }

    /// EVERY shape this builder ACCEPTS has whole-block per-core extents. This is the invariant, and it
    /// is stated over the builder's output rather than over `distribute_cores` — so it holds however the
    /// division is later fixed, and it cannot be satisfied by a divider that got lucky on one shape.
    #[test]
    fn every_accepted_transpose_gives_each_core_whole_8x8_blocks() {
        let mut accepted = 0u32;
        for mb in (64..=2048u32).step_by(64) {
            for out in (64..=1024u32).step_by(64) {
                let Ok(spec) = transpose_opspec(mb, out, "t1", "t2") else {
                    continue;
                };
                accepted += 1;
                for dim in ["mb", "out"] {
                    let per = spec.iter.per_core_extent(dim);
                    assert!(
                        per.is_multiple_of(8),
                        "[{mb}, {out}] was accepted with per-core {dim} = {per}, which is not a whole \
                         8-row/column block of the output's [8, 8] stick"
                    );
                }
            }
        }
        assert!(
            accepted > 0,
            "the sweep must actually accept something, or it proves nothing"
        );
    }

    // ── (ii) the stick law, each refusal with its control ────────────────────────────────────────

    /// The INPUT stick extent (`out`) must be a whole 64-fp16 stick — the guard that has always been
    /// here, with the control that 64 itself still bakes.
    #[test]
    fn an_unaligned_column_extent_is_refused_and_64_still_succeeds() {
        let e =
            transpose_opspec(MB, 65, "t1", "t2").expect_err("65 is not a whole 64-element stick");
        assert!(
            e.contains("stick extent 65"),
            "the refusal names the extent, got: {e}"
        );
        assert!(e.contains("64-elem"), "and the 64 rule, got: {e}");
        assert!(
            transpose_opspec(MB, 64, "t1", "t2").is_ok(),
            "STILL SUCCEEDS: one whole stick is the smallest legal column extent"
        );
        assert!(
            transpose_opspec(MB, 128, "t1", "t2").is_ok(),
            "STILL SUCCEEDS: and so is two"
        );
    }

    /// ⛔⛔⛔ THE ROW EXTENT IS A STICK EXTENT TOO, AND IT USED TO BE UNCHECKED — so the ONE shape
    /// whose on-card failure is on this project's record, the per-step `[1, 64]` key, was accepted by
    /// this builder. The control is the same transpose with its rows padded up.
    #[test]
    fn the_historically_fatal_1x64_shape_is_refused_and_a_padded_one_still_succeeds() {
        let e = transpose_opspec(1, 64, "t1", "t2")
            .expect_err("a 1-row transpose has a stick extent of 1 on its output");
        assert!(
            e.contains("stick extent 1"),
            "the refusal names the sub-stick extent, got: {e}"
        );
        assert!(
            transpose_opspec(MB, 64, "t1", "t2").is_ok(),
            "STILL SUCCEEDS: padded to whole sticks, the same transpose bakes"
        );
        assert!(
            transpose_opspec(0, 64, "t1", "t2").is_err(),
            "and a zero row extent is not a transpose"
        );
    }

    // ── (ii) the same law at the ENTRY POINT, where a producer meets it ──────────────────────────

    /// One KTIR program shaped like a whole-tensor transpose: parameter 0 reads `[rows, cols]`,
    /// parameter 1 is stored through a `[cols, rows]` view. Built out of the same four op kinds
    /// `regions` reads (`construct_memory_view`, `construct_access_tile`, `arith.constant`, `store`),
    /// so the lowering sees a real program rather than a stub.
    fn transpose_program(in_shape: (i64, i64), out_shape: (i64, i64)) -> KtirNode {
        let a: &'static Arena = Arena::global();
        let shape = |op: Operation<'static>, s: (i64, i64)| {
            op.with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(vec![s.0, s.1])))
        };
        // %0 = input address, %1 = output address; %2 = the index constant 0 both corners use.
        let (p_in, p_out, zero) = (Ssa(0), Ssa(1), Ssa(2));
        let (v_in, t_in, v_out, t_out, val) = (Ssa(3), Ssa(4), Ssa(5), Ssa(6), Ssa(7));
        let ops = vec![
            Operation::new(a, Some(zero), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Int(0),
            ),
            shape(
                Operation::new(a, Some(v_in), OpKind::KtdpConstructMemoryView, &[p_in]),
                in_shape,
            ),
            shape(
                Operation::new(
                    a,
                    Some(t_in),
                    OpKind::KtdpConstructAccessTile,
                    &[v_in, zero, zero],
                ),
                in_shape,
            ),
            shape(
                Operation::new(a, Some(v_out), OpKind::KtdpConstructMemoryView, &[p_out]),
                out_shape,
            ),
            shape(
                Operation::new(
                    a,
                    Some(t_out),
                    OpKind::KtdpConstructAccessTile,
                    &[v_out, zero, zero],
                ),
                out_shape,
            ),
            Operation::new(a, Some(val), OpKind::KtdpLoad, &[t_in]),
            Operation::new(a, None, OpKind::KtdpStore, &[val, t_out]),
        ];
        KtirNode {
            func: IRFunction {
                name: "transpose_s0",
                arguments: a.args(vec![(p_in, IrType::Index), (p_out, IrType::Index)]),
                operations: a.ops(ops),
                grid: (1, 1, 1),
                return_type: None,
            },
            program: Program::Transpose,
            bindings: vec![BufferId::new(101), BufferId::new(102)],
            mask: None,
            node_out_tid: None,
        }
    }

    fn lower_transpose(k: &KtirNode) -> Result<Vec<EmittedOp>, lower_ktir_to_superdsc::Error> {
        let mut sid = 0i64;
        let r = lower_ktir_to_superdsc::regions(k)?;
        lower_ktir_to_superdsc::transpose(k.func.name, &r, &mut sid, None)
    }

    /// The refusal message, or a panic naming what was emitted instead. Hand-written because
    /// `EmittedOp` is deliberately not `Debug` (it carries a whole descriptor), so `expect_err` cannot
    /// be used — and a test that reads a refusal must fail loudly when there is none.
    fn refusal(k: &KtirNode) -> String {
        match lower_transpose(k) {
            Ok(ops) => panic!(
                "expected a refusal, got {} descriptor(s): {:?}",
                ops.len(),
                ops.iter().map(|o| o.op_name.clone()).collect::<Vec<_>>()
            ),
            Err(e) => e.message,
        }
    }

    /// ⛔⛔⛔ THE CASE A CONSUMER ASKED FOR IS THE CASE THIS CRATE REFUSES, AND THE REFUSAL HAS TO SAY
    /// SO IN FULL. A `[1, 64]` per-step key is the shape that deleted the former generic op walk and
    /// then wrote the second Kᵀ stick wrong on-card. It reaches the entry point as an `Err` that names
    /// the extent, the 64 rule, AND the alternative the shipped attention uses — a producer that gets
    /// only "unsupported" pads nothing and files a bug.
    #[test]
    fn the_entry_point_refuses_a_1x64_transpose_and_names_the_padding_alternative() {
        let e = refusal(&transpose_program((1, 64), (64, 1)));
        assert!(e.contains("[1, 64]"), "names the shape, got: {e}");
        assert!(
            e.contains("the row extent 1"),
            "names WHICH extent is wrong, got: {e}"
        );
        assert!(
            e.contains("multiples of the 64-element SEN169_FP16 stick"),
            "names the 64 rule, got: {e}"
        );
        assert!(
            e.contains("Pad the offending extent up to a multiple of 64")
                && e.contains("64x64 tiles"),
            "names the padding-plus-tiling alternative, got: {e}"
        );
        assert!(
            e.contains("garbled decode past 64 tokens"),
            "names the on-card consequence, got: {e}"
        );
    }

    /// THE CONTROL, and the proof that the entry point is not simply broken: the same transpose padded
    /// to whole sticks lowers, and it lowers to exactly ONE descriptor named after its output buffer.
    ///
    /// ⛔⛔⛔ AND THE PRIMITIVE IT LOWERS TO CHANGED, WHICH IS WHY THIS TEST MOVED. It used to assert
    /// `interslicetranspose_fp16` / `stickSize_ [8, 8]`, and that emission CANNOT BE TRANSLATED: DDC
    /// turns a relayout's output stick order into a dynamic mask with one offset per dim of the parent
    /// loop, and the DSC2→DataflowIR V3 translator accepts exactly one
    /// (`SNComputeLowering.cpp:72/74`) — measured as `dxp_standalone` exit 1 on both decoder
    /// configurations, at `sdsc_22`, at four shapes. The door now calls
    /// [`try_assemble_restickify_transpose_2d`], whose output stick is the single `y` axis.
    ///
    /// ⚖️ NOTHING HERE WEAKENS `OpFunc::Transpose`'S OWN COVERAGE. The four sibling tests that assert
    /// the `[8, 8]` block stick, the `interslicetranspose_fp16` wire name, the golden's layout and the
    /// whole-block core division all call [`assemble_transpose`] DIRECTLY and are untouched — that
    /// assembler still exists and is still pinned. This test is about the DOOR, and the door's answer
    /// is a different primitive now.
    #[test]
    fn the_entry_point_still_lowers_a_stick_aligned_transpose() {
        let ops = lower_transpose(&transpose_program((MB as i64, 128), (128, MB as i64)))
            .expect("a stick-aligned whole-tensor transpose is the case this supports");
        let [op] = &ops[..] else {
            panic!("a transpose is ONE descriptor, got {}", ops.len());
        };
        assert_eq!(
            op.op_name, "transpose_o102",
            "named after its output buffer"
        );
        let d = dsc(op, "transpose_o102");
        // ONE-dim output stick — the only property the V3 translator's limit is a function of.
        assert_eq!(
            d.primaryDsInfo_["OUTPUT"].stickDimOrder_,
            vec!["y"],
            "the door's output stick must be the single `y` axis; a two-dim stick is the emission \
             `SNComputeLowering.cpp:74` refuses"
        );
        assert_eq!(d.primaryDsInfo_["OUTPUT"].stickSize_, vec![64]);
        // AND the roles' dim orders are SWAPPED, which is what makes it a transposition rather than a
        // re-stick that moves the stick axis alone and computes the wrong thing.
        assert_eq!(
            (
                d.primaryDsInfo_["INPUT"].layoutDimOrder_.as_slice(),
                d.primaryDsInfo_["OUTPUT"].layoutDimOrder_.as_slice(),
            ),
            (["y", "out"].as_slice(), ["out", "y"].as_slice()),
        );
        assert_eq!(d.computeOp_[0].opFuncName, "ReStickifyOpHBM");
    }

    /// A column extent that is not a whole stick is refused at the entry point too, and the message
    /// blames the COLUMN — a refusal that named the wrong axis would send a producer padding the wrong
    /// side.
    #[test]
    fn the_entry_point_names_the_column_extent_when_that_is_the_unaligned_one() {
        let e = refusal(&transpose_program((MB as i64, 65), (65, MB as i64)));
        assert!(
            e.contains("the column extent 65"),
            "blames the column axis, got: {e}"
        );
    }

    /// An output view that is not the input's, swapped, is a RESHAPE — refused, because both extents
    /// can be perfectly stick-aligned and the descriptor would still write a shape the program never
    /// declared. (Stick-aligned on purpose: this refusal is unreachable through the one above.)
    #[test]
    fn a_transpose_whose_output_view_is_not_the_swapped_input_is_refused() {
        let e = refusal(&transpose_program((MB as i64, 128), (MB as i64, 128)));
        assert!(
            e.contains("swapped"),
            "the refusal names what the output view must be, got: {e}"
        );
    }

    /// ⭐⭐⭐ THE DECODER'S OWN TRANSPOSE LOWERS THROUGH THE ENTRY POINT — the regression this whole
    /// divider exists for. `[64, 128]` → `[128, 64]` is `tl.dot(p, v)`'s B operand in
    /// `decoder_block.py`, and it is the shape that used to trip the core-division guard with per-core
    /// (mb 2, out 128).
    ///
    /// ⛔ THIS TEST REPLACES ONE THAT PINNED THAT REFUSAL AS "AN `Error`, NOT A PANIC", and the intent
    /// behind it is preserved in a STRONGER form below rather than dropped. The concern was real:
    /// `assemble_transpose` `panic!`s on a refused opspec, so a shape passing the entry point's stick law
    /// but tripping the opspec's division guard would abort the build with a bare panic instead of a
    /// producer-reportable error. The answer is no longer "the error is plumbed through" but "there is no
    /// error left to plumb" — see the sweep in
    /// [`no_shape_the_entry_point_accepts_can_make_the_opspec_refuse`].
    #[test]
    fn the_decoders_transpose_shape_lowers_through_the_entry_point() {
        let ops = lower_transpose(&transpose_program((64, 128), (128, 64)))
            .expect("`[64, 128]` -> `[128, 64]` is the decoder's `tl.dot(p, v)` operand");
        assert_eq!(ops.len(), 1, "one `interslicetranspose_fp16`, not a decomposition");
    }

    /// ⛔⛔⛔ THE SEAL IS UNREACHABLE BY CONSTRUCTION, AND THAT IS WHAT IS ASSERTED — not that it was
    /// deleted, because it was not.
    ///
    /// `transpose_opspec`'s block guard still re-checks the property on every plan the divider returns;
    /// it is the seal that would fire if the divider, the block size, or `TileOp::tile` ever changed
    /// under it. What this pins is that no shape the ENTRY POINT accepts can trip it, so the `panic!`
    /// inside `assemble_transpose` has nothing to fire on. Both halves matter: keeping the guard means a
    /// divider regression is caught, and this sweep means a legal producer program can never meet it.
    ///
    /// The two guards compose because the entry point already demands both extents be whole 64-element
    /// sticks, and 64 is itself a whole number of 8-blocks — so even the one-core fallback division
    /// satisfies the block property. That is the reasoning; this is the measurement of it.
    #[test]
    fn no_shape_the_entry_point_accepts_can_make_the_opspec_refuse() {
        let mut checked = 0u32;
        for mb in (64..=1024u32).step_by(64) {
            for out in (64..=1024u32).step_by(64) {
                // The entry point accepts exactly the stick-aligned, view-swapped whole-buffer case.
                if lower_transpose(&transpose_program(
                    (mb as i64, out as i64),
                    (out as i64, mb as i64),
                ))
                .is_err()
                {
                    continue;
                }
                checked += 1;
                let spec = transpose_opspec(mb, out, "t1", "t2").unwrap_or_else(|e| {
                    panic!(
                        "[{mb}, {out}] passed the entry point but the opspec refused it: {e} — that                          reaches `assemble_transpose` as a bare `panic!`, which is the failure mode the                          test this replaced was written to prevent"
                    )
                });
                for dim in ["mb", "out"] {
                    let per = spec.iter.per_core_extent(dim);
                    assert!(
                        per.is_multiple_of(8),
                        "[{mb}, {out}]: per-core {dim} = {per} is not a whole 8-block"
                    );
                }
            }
        }
        assert_eq!(checked, 256, "the sweep must cover every stick-aligned shape in its range, or it \
             proves nothing about the composition of the two guards");
    }

    // ── (iii) the byte-identity control ──────────────────────────────────────────────────────────

    /// ⭐ NOTHING ELSE MOVED, AND THIS IS THE ASSERTION THAT COULD CATCH IT. The `is_transpose_out`
    /// arm is gated on `op.op == OpFunc::Transpose` **and** `Role::Output` — and every pointwise
    /// operand is at `Role::Output` (`pointwise_opspec_from_tile` puts input and output alike there,
    /// which is exactly why `transpose_opspec`'s doc has to say it uses `Role::Input` for its input).
    /// So a gate that had drifted to the role alone would rewrite the stick layout of every pointwise
    /// op in every bundle. It does not: a pointwise still declares its ONE role with the plain
    /// single-stick `["out"]`/`[64]`.
    ///
    /// That is the strongest byte-identity control available in-source, and the reason a stronger one
    /// is not needed: `assemble_transpose`/`transpose_opspec` had ZERO callers in the workspace before
    /// this change (grep-verified), so no descriptor that bakes today passes through either — the added
    /// row-extent witness narrows a function nothing called.
    #[test]
    fn a_non_transpose_op_still_sticks_one_64_element_stick_on_its_only_role() {
        let mut sid = 0i64;
        let op =
            assemble_pointwise_seeded("pw", "add", 64, 128, &["t1", "t2"], "t3", &mut sid, None);
        let d = dsc(&op, "pw");
        assert_eq!(
            d.primaryDsInfo_.keys().copied().collect::<Vec<_>>(),
            vec!["OUTPUT"],
            "a pointwise puts EVERY operand at Role::Output, so it shares the key the transpose \
             override is written under"
        );
        let p = &d.primaryDsInfo_["OUTPUT"];
        assert_eq!(
            p.stickDimOrder_,
            vec!["out"],
            "still the plain single-stick layout — the 8x8 override is Transpose-only"
        );
        assert_eq!(p.stickSize_, vec![64]);
        assert_eq!(
            d.computeOp_[0].opFuncName, "add",
            "and it is still an add on the sfp unit"
        );
        assert_eq!(d.computeOp_[0].exUnit, "sfp");
    }
}

/// ⭐⭐⭐ THE STICK LAW AS A PROPERTY, NOT AS FOUR EXAMPLES — the refusal side, EXHAUSTIVELY.
///
/// The tests above name the shapes history names (`[1, 64]`, `65`). This says the thing that has to be
/// true of every OTHER shape too: no `(mb, out)` pair that is not a pair of whole 64-fp16 sticks can
/// produce a descriptor at all. That is the direction that matters — an accepted unaligned extent is a
/// `DtException 1535` on the card, or, for the `mb = 1` case, the silently-wrong second Kᵀ stick and a
/// fluent wrong answer — whereas a refused aligned extent is a build error somebody reads.
///
/// ⚖️ A SWEEP RATHER THAN A KANI HARNESS, AND THAT IS NOT A PREFERENCE. The crate's other symbolic
/// properties are `#[kani::proof]`s, and this one was written as one first. **Kani cannot build this
/// crate today**: Kani 0.67.0 bundles rustc 1.93.0-nightly and `ktir-core` (the input language, a
/// mandatory dependency) declares `rust-version = "1.94"`, so `cargo kani -p ktir-superdsc` fails
/// before verification with "rustc 1.93.0-nightly is not supported". A harness nobody can run is how
/// "the proofs hold" turned out to be vacuous for all 11 superdsc proofs once already, so the property
/// is stated in the form that actually executes: over a BOUNDED domain, an exhaustive sweep IS the
/// proof for that domain — there is no symbolic gap to leave unverified. Restore the harness when the
/// toolchains meet.
#[cfg(test)]
mod transpose_stick_law {
    use super::*;

    /// NO `(mb, out)` in `0..=512²` that is not a pair of whole 64-element sticks is EVER accepted.
    ///
    /// ⭐ ONE DIRECTION, DELIBERATELY. The converse — "every whole-stick pair is accepted" — is FALSE
    /// and must not be asserted: the core-division guard refuses the whole-stick pairs whose per-core
    /// slice misses the output's 8×8 block, which today is most of them (see
    /// `the_fixtures_own_shape_is_refused_because_our_core_division_is_not_the_fixtures`). A sweep that
    /// demanded acceptance would have to encode `distribute_cores`' behaviour as an expectation, i.e.
    /// assert our own division as truth. The invariant that acceptance implies whole blocks is stated
    /// separately, over the builder's output, by
    /// `every_accepted_transpose_gives_each_core_whole_8x8_blocks`.
    ///
    /// 512 spans the extents a real transpose has (a chunk width against a head dim or feature width)
    /// and covers 8 stick multiples per axis, so the `% 64 != 0`, `% 8 == 0 but % 64 != 0` and zero
    /// cases are all inside it.
    #[test]
    fn no_unaligned_extent_can_produce_a_transpose_descriptor() {
        let stk = <Fp16 as DataFormat>::ELEMS_PER_STICK;
        let (mut checked, mut sub_stick_but_8_whole) = (0u32, 0u32);
        for mb in 0..=512u32 {
            for out in 0..=512u32 {
                if mb != 0 && out != 0 && mb % stk == 0 && out % stk == 0 {
                    continue; // the whole-stick pairs: the division guard decides those, not this law
                }
                checked += 1;
                if mb % 8 == 0 && out % 8 == 0 && mb != 0 && out != 0 {
                    sub_stick_but_8_whole += 1;
                }
                assert!(
                    transpose_opspec(mb, out, "t1", "t2").is_err(),
                    "[{mb}, {out}] is not a pair of whole {stk}-element sticks, so it must never \
                     reach a descriptor"
                );
            }
        }
        assert_eq!(
            checked,
            513 * 513 - 64,
            "every pair but the 64 whole-stick ones"
        );
        // THE SWEEP'S OWN NEGATIVE CONTROL: the domain really does contain the interesting case — a
        // pair that is a whole multiple of the 8×8 BLOCK but not of the 64-element STICK. If the guard
        // had been written at 8 instead of 64 these would be accepted, and the assertion above is only
        // meaningful because they are in it.
        assert_eq!(
            sub_stick_but_8_whole, 4032,
            "64 multiples-of-8 per axis, minus the 64 pairs that are also whole sticks"
        );
    }
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
/// [`KtTileSlots`](crate::sdsc_abstract::KtTileSlots)/[`KtTileFeats`](crate::sdsc_abstract::KtTileFeats) —
/// each door names which quantity fills it, so the pair cannot be handed over swapped.
/// `in_off`/`out_off` are the per-head element offsets into the natural cache (`head·cap·hd`) and
/// the Kᵀ scratch (`head·hd·cap`).
#[allow(clippy::too_many_arguments)]
pub fn assemble_restickify_kt_2d(
    op_name: &str,
    slots: crate::sdsc_abstract::KtTileSlots,
    feats: crate::sdsc_abstract::KtTileFeats,
    in_name: &str,
    in_off: crate::addr::DevOff,
    o_name: &str,
    out_off: crate::addr::DevOff,
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

/// ⭐⭐⭐ A WHOLE-TENSOR 2-D TRANSPOSE FOR A **KTIR** `linalg.transpose`, REALIZED BY THE STICK AXIS —
/// the same relayout [`restickify_kt_opspec_2d`] does for the paged Kᵀ cache, reached with PLAIN
/// extents instead of the paged-KV tile newtypes. `rows` is the axis that becomes the output's
/// COLUMNS (the restickify's `cap`/`y`); `cols` is the axis that becomes the output's ROWS (its
/// `hd`/`out`). `[rows, cols]` in, `[cols, rows]` out, both whole 64-element fp16 sticks.
///
/// # ⛔⛔⛔ WHY THIS EXISTS AND `OpFunc::Transpose` IS NOT USED FOR A KTIR TRANSPOSE
///
/// `interslicetranspose_fp16`'s output stick is the 8×8 inter-slice BLOCK over (`out`, `mb`) —
/// [`emit_sdsc`]'s `is_transpose_out` arm, `stickSize_: [8, 8]`. TWO measured consequences, and the
/// second one is the one that matters, because a bake gate cannot see it:
///
///   1. **IT CANNOT BE TRANSLATED.** `Ddc::transformForInterSliceRestickify`
///      (`ddc/ddc_transformation.cpp:2398`, called unconditionally from `Ddc::run_v1`) fires on any PT
///      compute node whose INPUT and OUTPUT stick orders differ — every relayout — and writes one mask
///      offset per dim of `getParentDimLoop(outputStickDimOrder[0])` (`:2383-2397`). For a 2-D
///      (`mb`, `out`) tile that parent is the FUSED `loop_dsX_dsY_out_mb`, two dims;
///      `SNComputeLowering::constructDynamicMasking` accepts exactly ONE
///      (`dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:74`). MEASURED at the DEFAULT
///      arch (`DEFAULT_ISA = RCUDD1A_ISA`, `sys-arch-spec/isa/isa.hpp:31`): the transpose door exits 1
///      with "Translator currently supports translating only 1-D dynamic masking" at FOUR shapes
///      ([64,64], [256,64], [64,512], [128,1024] — two of which split both axes across cores like the
///      vendor's own `sdsc_interslicetranspose.json` golden), so it is the stick DIMENSIONALITY and
///      not the shape or the core division. This door's output stick is the single `y` axis, DDC lands
///      on its one-dim `loop_ds2_ds3_y`, and the same masked-MACC path is ACCEPTED: exit 0, a 477-byte
///      all-concrete plan and a 3200-byte `init_binary.bin` whose md5 moves when either operand's
///      start address is perturbed. Pinned by `triton-ktir-superdsc/tests/transpose_shape_probe.rs`.
///
///   2. **AND THE 8×8 BLOCK IS NOT A KERNEL RESIDENCY, SO THE CONSUMER READ THE WRONG BYTES.** The
///      only reason a KTIR transpose exists is to make a `[k, n]` score kernel out of a COMPUTED value
///      (a RoPE'd K — see [`whole_function`]'s plain-B note, which records this exact mismatch being
///      measured once already for a transposed-B WEIGHT: `32_transpose_o41` wrote an 8×8-block-sticked
///      tile and `33_matmul_o42` then declared its KERNEL `["in","out"]` single-stick over it). The
///      matmul KERNEL slot reads ONE residency — `StickLayout::kernel(k_in, n_out)`, i.e.
///      `dev_off_stk`'s rank-2 stick-on-last law — and `OpFunc::Transpose` declares BOTH its operands
///      rank-3 `["mb","out","y"]`, which takes `DeviceTileLayout::device_size`'s FLAT branch for both,
///      carrying the reorder only in the hardware's in-flight 8×8 block move. This door's OUTPUT is a
///      rank-2 `["out","y"]` sticked on `y`, so `dev_off([hd,cap],1,[d,slot]) = d·cap + slot` — which
///      IS `dev_off([in,out],1,[i,o])` of the KERNEL the score matmul declares next. The two agree by
///      the same formula rather than by a hardware step in between.
///
/// # THE INPUT MUST BE SLAB-MAJOR, AND IT IS — BY THE SAME BRANCH, NOT BY LUCK
///
/// [`restickify_kt_opspec_2d`] reads its input at `dev_off([rows, cols], 1, ·)`, the sticked-kernel
/// tiling `device_size = [cols/stk, rows, stk]`. A KTIR pointwise producer (the RoPE `add`) declares
/// its output rank-2 `[mb, out]` sticked on `out` — `layout.len() == 2 && stick_idx == 1`, the SAME
/// `DeviceTileLayout::device_size` branch — so producer and consumer are the same formula at every
/// `cols` that is a whole stick, and no second relayout is needed to make the input slab-major. At
/// `cols == stk` the single slab degenerates to the flat `row·cols + col`, which is additionally
/// byte-identical to what the rank-3 transpose door was reading, so the substitution moves no input
/// byte at the shape it was measured on.
///
/// # ⛔ NO PAGED-KV NEWTYPE IS WIDENED TO REACH THIS
///
/// [`assemble_restickify_kt_2d`]'s extents are [`KtTileSlots`](crate::sdsc_abstract::KtTileSlots) /
/// [`KtTileFeats`](crate::sdsc_abstract::KtTileFeats), and `KtTileSlots` deliberately has NO door for
/// an arbitrary extent — only `of_row_window` and `of_page`, because its doc's whole point is that
/// the four 64s of the paged pool cannot cross-convert. A KTIR `linalg.transpose` is neither a row
/// window nor a page, so this takes plain extents and calls the SAME private opspec builder; the
/// newtypes and their two doors are untouched. The extent guard is not lost: `StickExtent::<Fp16>`
/// inside that builder refuses a sub-stick extent by name.
///
/// # ⛔ AND IT REFUSES A TILE THAT DOES NOT FIT LX, RATHER THAN EMITTING ONE THAT OVERFLOWS
///
/// [`transpose_opspec`] runs `TileOp::tile`, which time-tiles a tile too big for
/// [`USABLE_LX_BYTES`](crate::superdsc_opspec::USABLE_LX_BYTES). `restickify_kt_opspec_2d` sets
/// `time_tile: None` and cannot be time-tiled, and `TileOp`'s residency formula is unusable here
/// because `pointwise_lx_resident_generic` reads `per_core_extent("mb")` while this builder's dims are
/// named `y`/`out` — it would silently charge one row. So the fit is checked directly, on the plan the
/// builder actually produced (2 live operands, per-core `y × out` fp16), and a shape that does not fit
/// is a NAMED refusal. That is a narrowing of what this door accepts, never a descriptor that
/// addresses past LX.
#[allow(clippy::too_many_arguments)]
pub fn try_assemble_restickify_transpose_2d(
    op_name: &str,
    rows: u32,
    cols: u32,
    in_name: &str,
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, String> {
    // `cap` = the input's ROW axis (the restickify's `y`, the output's columns);
    // `hd`  = the input's COLUMN axis (its `out`, the output's rows). Whole-buffer, so no offsets.
    let op = restickify_kt_opspec_2d(rows, cols, in_name, o_name, 0, 0)?;
    // THE LX FIT, on the plan the builder chose — see this function's doc. Two live operands (the
    // input tile and the output tile), per-core extents, fp16 words.
    let (per_y, per_out) = (op.iter.per_core_extent("y"), op.iter.per_core_extent("out"));
    let resident = 2u64 * u64::from(per_y) * u64::from(per_out) * u64::from(Fp16::WORD_LENGTH);
    if resident > crate::superdsc_opspec::USABLE_LX_BYTES {
        return Err(format!(
            "restickify transpose [{rows}, {cols}]: the core division gives each core \
             (y {per_y}, out {per_out}), whose 2 live fp16 tiles are {resident} B against the \
             {} B LX scratchpad. This relayout's opspec carries `time_tile: None` — its output stick \
             IS the whole `y` axis, so there is no stick dim left to time-tile without splitting an \
             output stick — so an over-large tile is refused here rather than emitted as a descriptor \
             that addresses past LX. Transpose in whole tiles that fit, the way the shipped attention \
             does (one restickify per sub-block).",
            crate::superdsc_opspec::USABLE_LX_BYTES,
        ));
    }
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout).map_err(|e| e.0)
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

/// The SFP transcendentals that CRASH the dxp compile at multi-stick width with an out-broadcast
/// operand — see the guard in `assemble_pointwise_broadcast_off`, which explains the geometry and cites
/// the 2026-06-27 `map::at` observation.
///
/// ⛔ ONE COPY, BECAUSE THERE WERE TWO AND THEY HAD ALREADY DRIFTED. This list was a function-local
/// `const` duplicated verbatim in `assemble_pointwise_broadcast_off` here and in
/// `assemble_pointwise_broadcast_off_from_tile` under `ir::bridge::tiled_op_sdsc_op::pointwise` — so
/// every fix had to land twice, and `"mish"` shows what happens when it doesn't: it was MISSING from
/// both copies although this crate's own DDL audit lists it among the SFP transcendentals lowered
/// through that path, right beside `sigmoid`, `gelu` and `tanh`. It is added here.
///
/// ⚠️ IT IS A STRING TABLE AND CANNOT BE TYPED TODAY. `"relu"` and `"layernormscale"` have NO `OpFunc`
/// variant, so keying this on the enum would make them inexpressible — while
/// [`op_func_from_str`] PANICS on both. That is the live inconsistency: this table admits two names its
/// own converter refuses. Typing it means first deciding whether those two ops exist for this backend.
pub(crate) const SFP_SPLIT_TRANSCENDENTALS: &[&str] = &[
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
    // ⛔ ADDED: absent from BOTH former copies, though the crate's own DDL audit classes it a
    // transcendental. Its omission left the dxp-crash guard blind to a `mish` at multi-stick width.
    "mish",
];

/// Map a `&str` op-func (the `assemble_*` call convention) to the sealed [`OpFunc`].
///
/// PANICS on an unknown key — a build-time failure at proc-macro/test time — rather than defaulting,
/// because a typo'd op name is a WRONG OP, i.e. silently-wrong output. `"mul"` is an alias for
/// `"multiply"`, `"sub"` for `"subtract"`.
///
/// ⛔ THE PARAGRAPH ABOVE THIS ONE USED TO SAY "Unknown names default to `Add`", DIRECTLY ABOVE A BODY
/// THAT PANICS. Two doc paragraphs were stacked here, the first describing behaviour that had been
/// removed, and it was believed over the code during this crate's own review — the reader reported the
/// function as silently defaulting and had to retract it. A doc comment is not evidence; the arms are
/// twenty lines below.
///
/// ⚠️ THIS IS NOT THE WIRE SPELLING. The key here and [`OpFunc::name`] differ for four ops —
/// `"multiply"`→`mul`, `"subtract"`→`sub`, `"gelu"`→`gelufwd`, and `Transpose`/`Restickify` have no key
/// at all. One caller turns a key of this vocabulary into a DESCRIPTOR NAME
/// (`format!("{op_func}_o{tid}")` in `lower_ktir_to_superdsc`'s `elementwise`), and that name reaches
/// both the dxp-input top-level key and the bundle fingerprint. So substituting `OpFunc::name()` at
/// that site would change emitted bytes and invalidate every cached bake, with no predicate breaking to
/// say so.
pub fn op_func_from_str(s: &str) -> OpFunc {
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
        ///
        /// ⛔ `pub(in crate::emit)`, NOT `pub(crate)`, AND THAT IS WHAT MAKES THE DOC ABOVE TRUE.
        /// While this mint and `ir::bridge::tiled_op_sdsc_op::attn` shared one crate, `pub(crate)`
        /// let the attention emitter call it — the seal read as real and was nominal. `crate::emit`
        /// is an ancestor of both wrapper-site callers and of nothing under `ir::bridge`.
        pub(in crate::emit) fn witness() -> Self {
            TapeLoweringSite(())
        }
    }
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
/// ⚠️ `pub` ONLY BECAUSE ITS LAW TEST IS OUTSIDE THIS CRATE. `fp32_reduce_scale_const_is_ieee_fp32`
/// (a consumer's emitter tests — `scratchy-target-spyre` in this workspace) pins the IEEE-fp32 word
/// this returns; the test stayed
/// where it was so the roster did not move, and the crate boundary now sits between them. Its sibling
/// [`sen169_bits`] is `pub` for the same reason.
pub fn scaling_factor_const_fp32(bits: u32) -> serde_json::Value {
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
pub(crate) fn check_pointwise_cols(
    cols: u32,
    what: &str,
    out_id: u32,
) -> Result<(), SuperDscError> {
    if !cols.is_multiple_of(FP16_ELEMS_PER_STICK) {
        return Err(SuperDscError(format!(
            "{what} t{out_id}: cols={cols} not a multiple of the {FP16_ELEMS_PER_STICK}-fp16 \
             stick (assemble_pointwise emits no coordinateMasking_) — pad cols"
        )));
    }
    Ok(())
}

// ───────────────────────────────────────────────────────────────────────────
// The indirect-access (gathered HBM) baseline.
// ───────────────────────────────────────────────────────────────────────────

/// ⭐ THE BYTE-IDENTITY BASELINE FOR THE INDIRECT-ACCESS FIELD SET, PINNED BEFORE ANY GATHER EXISTS.
///
/// [`crate::wire`] declares the vendor's whole indirect-access field set — `indirectAllocType_`,
/// `relatedIndirectAccessAlloc_`, `indexTensorType_` and `isStartAddrSymbolic_` on [`AllocNode`],
/// `indirectAccessIndexLabeledDs` on [`ComputeOp`] — plus [`MemOrg::hbm_only`] beside them. Nothing
/// in this crate sets any of them to an indirect value: [`emit_sdsc`] hardcodes every alloc node to
/// `"no_indirection"` at its single construction site and leaves the other four `None`/empty, and
/// `MemOrg::hbm_only` has ZERO call sites. The capability is DECLARED AND UNREACHABLE.
///
/// Whatever eventually reaches it will rest on one claim: that switching it on changes NOTHING for
/// the ops that do not use it — that every already-shipped bundle stays BYTE-IDENTICAL, not merely
/// equivalent. That is a claim about a serializer, and a claim about a serializer is worth exactly
/// one diff. There was no diff. These tests are it, so a later change is measured against a recorded
/// baseline rather than against a reading of the emitter.
///
/// ⛔ TYPED-`None` AND SERIALIZED-ABSENT ARE TWO DIFFERENT CLAIMS, AND ONLY ONE OF THEM IS ABOUT THE
/// BYTES. A bare `Option::None` serializes as `null`; it is `skip_serializing_if = "Option::is_none"`
/// that omits the key entirely. That attribute is one line, droppable without breaking a single type
/// or a single typed assertion, and dropping it puts three new keys into every alloc node of every
/// bundle. So each field is asserted TWICE — once on the typed value, once on the emitted JSON — and
/// [`indirect_access_baseline::omission_is_the_serializer_not_the_spelling`] is the negative control
/// that proves the JSON half can fail at all.
#[cfg(test)]
mod indirect_access_baseline {
    use super::*;
    use crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec;

    /// Occurrences of `needle` in `hay`. Counting rather than `contains` is what makes an
    /// every-node claim non-vacuous: `contains` passes on one node out of three.
    fn count(hay: &str, needle: &str) -> usize {
        hay.match_indices(needle).count()
    }

    /// The op every assertion here is made about: an ordinary batched fp16 matmul over three
    /// distinct operands. It declares no gather — as does every op this crate can build today.
    /// Same shape the emitter's other field-set tests use, so a divergence is about the fields
    /// under test and not about a differently-shaped op.
    fn non_gather_sdsc() -> (SdscOp, String) {
        let op = matmul_opspec(384, 384, 64, 16, "Tensor0", "Tensor1", "Tensor2")
            .expect("a 384x384x64 batch-16 fp16 matmul is a legal OpSpec");
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let sdsc = emit_sdsc("MatMul_0", &op, &folds, None).expect("emit_sdsc accepts it");
        let json = serde_json::to_string(&sdsc).expect("SdscOp serializes");
        (sdsc, json)
    }

    fn dsc_of(sdsc: &SdscOp) -> &Dsc {
        &sdsc.dscs_[0]["MatMul_0"]
    }

    /// `indirectAllocType_` is the ONE field of the set that is not optional: a plain
    /// `&'static str` with no `skip_serializing_if`, so it is present in EVERY alloc node of every
    /// bundle already shipped, and its value is the whole of today's declaration. Pin the value and
    /// pin the count, so a change that flips one node of three cannot pass.
    #[test]
    fn every_alloc_node_declares_no_indirection() {
        let (sdsc, json) = non_gather_sdsc();
        let nodes = &dsc_of(&sdsc).scheduleTree_;

        // ⛔ ANTI-VACUITY FIRST. A `for` over an empty vec asserts nothing, and every claim below
        // this line is quantified over these nodes. Three distinct operand names → three nodes.
        assert_eq!(
            nodes.len(),
            3,
            "one alloc node per distinct operand (Tensor0/1/2) — if this moved, every \
             per-node claim below changed meaning"
        );

        for n in nodes {
            assert_eq!(
                n.indirectAllocType_, "no_indirection",
                "alloc node {} declares no indirection",
                n.name_
            );
        }
        // And in the bytes, once per node — no more, no fewer.
        assert_eq!(
            count(&json, "\"indirectAllocType_\":\"no_indirection\""),
            nodes.len(),
            "every alloc node carries the string, and nothing else in the SDSC does"
        );

        // ⛔ THE TWO INDIRECT VALUES ARE ABSENT FROM THE WHOLE DESCRIPTOR. `no_indirection` is one
        // of three values this field can take; the other two are what a gather sets it to. Neither
        // occurs anywhere in this crate's output today, and this is the line that says so.
        assert_eq!(count(&json, "index_tensor"), 0);
        assert_eq!(count(&json, "value_tensor"), 0);
    }

    /// The three OPTIONAL alloc-node fields, and the compute op's index list. All four are
    /// `skip_serializing_if`, so the typed `None`/empty and the omitted key are separate facts.
    #[test]
    fn the_optional_indirect_fields_are_none_and_omitted() {
        let (sdsc, json) = non_gather_sdsc();
        let dsc = dsc_of(&sdsc);

        for n in &dsc.scheduleTree_ {
            assert_eq!(
                n.relatedIndirectAccessAlloc_, None,
                "{}: no cross-link partner",
                n.name_
            );
            assert_eq!(n.indexTensorType_, None, "{}: not an index tensor", n.name_);
            // A time=1 concrete-address op: the symbolic-start path a gathered value tensor takes
            // is not entered here either.
            assert_eq!(
                n.isStartAddrSymbolic_, None,
                "{}: start address is concrete",
                n.name_
            );
        }

        assert!(!dsc.computeOp_.is_empty(), "there is a compute op to check");
        for c in &dsc.computeOp_ {
            assert!(
                c.indirectAccessIndexLabeledDs.is_empty(),
                "compute op '{}' names no index dataspace",
                c.opFuncName
            );
        }

        // The bytes: the KEYS are absent, not present-and-null. `null` here would be a new key in
        // every alloc node of every shipped bundle.
        for key in [
            "relatedIndirectAccessAlloc_",
            "indexTensorType_",
            "isStartAddrSymbolic_",
            "indirectAccessIndexLabeledDs",
        ] {
            assert_eq!(
                count(&json, key),
                0,
                "{key} is omitted from the JSON, not serialized as null"
            );
        }
        assert_eq!(count(&json, "null"), 0, "nothing in this SDSC is null");
    }

    /// ⛔ THE NEGATIVE CONTROL for the assertions above. Four `!contains` checks on hand-typed field
    /// names pass just as happily when a name is misspelled or the field has been renamed away — the
    /// test would then be asserting the absence of a string nothing was ever going to emit. So set
    /// the same three fields on a REAL node taken from the emission above and show the keys appear:
    /// the omission is `skip_serializing_if` doing its job, not the spelling failing to match.
    #[test]
    fn omission_is_the_serializer_not_the_spelling() {
        let (sdsc, _) = non_gather_sdsc();
        let mut node = dsc_of(&sdsc).scheduleTree_[0].clone();

        node.relatedIndirectAccessAlloc_ = Some("allocate-Tensor1_hbm".to_string());
        node.indexTensorType_ = Some("index");
        node.isStartAddrSymbolic_ = Some(1);
        let set = serde_json::to_string(&node).expect("AllocNode serializes");

        for key in [
            "relatedIndirectAccessAlloc_",
            "indexTensorType_",
            "isStartAddrSymbolic_",
        ] {
            assert_eq!(
                count(&set, key),
                1,
                "{key} DOES serialize when set — so its absence above is the Option, \
                 not a name this serializer never writes"
            );
        }

        // Same node, fields back to None: the keys go away again. Present-vs-absent on ONE node,
        // which is the property `skip_serializing_if` actually provides.
        node.relatedIndirectAccessAlloc_ = None;
        node.indexTensorType_ = None;
        node.isStartAddrSymbolic_ = None;
        let cleared = serde_json::to_string(&node).expect("AllocNode serializes");
        for key in [
            "relatedIndirectAccessAlloc_",
            "indexTensorType_",
            "isStartAddrSymbolic_",
        ] {
            assert_eq!(count(&cleared, key), 0, "{key} omitted once cleared");
        }
        // The other keys are untouched by the round trip, so the two strings above differ in
        // exactly the three fields under test.
        assert!(cleared.contains("\"indirectAllocType_\":\"no_indirection\""));
    }

    /// [`MemOrg::hbm_only`] is the residency an index tensor needs (HBM-only — there is no LX
    /// indirect addressing) and it has ZERO call sites: `emit_sdsc` reaches only
    /// [`MemOrg::hbm_lx`] and [`MemOrg::lx_only`]. Pin BOTH halves — the exact bytes it would
    /// produce, so the first caller inherits a recorded shape rather than deriving one; and that
    /// nothing on today's path produces those bytes, which is what "unused" means in the output.
    ///
    /// ⚠️ It is `pub` on a `pub` type in a `pub` module, so rustc's `dead_code` lint cannot see that
    /// it is unreachable and does not warn. Its being uncalled is therefore recorded here or nowhere.
    #[test]
    fn mem_org_hbm_only_is_unreached_by_todays_emission() {
        assert_eq!(
            serde_json::to_string(&MemOrg::hbm_only()).unwrap(),
            "{\"hbm\":{\"isPresent\":1}}",
            "hbm-only omits the lx component entirely (never `\"lx\":null`)"
        );

        let (sdsc, json) = non_gather_sdsc();
        let n_ds = dsc_of(&sdsc).labeledDs_.len();
        assert_eq!(n_ds, 3, "one labeledDs per operand");
        // Every dataspace this op emits is hbm+lx, so the hbm-only form appears nowhere. Counted
        // against the dataspace total: a single one flipping would drop the count.
        assert_eq!(
            count(
                &json,
                "\"memOrg_\":{\"hbm\":{\"isPresent\":1},\"lx\":{\"isPresent\":1}}"
            ),
            n_ds,
            "every operand is hbm+lx resident"
        );
        assert_eq!(
            count(&json, "\"memOrg_\":{\"hbm\":{\"isPresent\":1}}"),
            0,
            "MemOrg::hbm_only is not on this path"
        );
    }
}
