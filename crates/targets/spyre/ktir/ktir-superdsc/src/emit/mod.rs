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
use crate::sdsc_abstract::{
    BlockCols, FlatTag, KindTag, Lanes, RowBlockedTag, RowCount, StickKind, StickLayout, Stk,
};
use crate::superdsc_error::SuperDscError;
use crate::superdsc_opspec::{
    Allocation, AnyTensorArg, ArgView, DataFormat, DeviceTileLayout, Df, Fp8, Fp16, Fp32,
    IndirectRole, ItDim, MAX_CORES, MaxCores, OpFunc, OpInfo, OpSpec, Role, Scale, SdscFoldSet,
    StickExtent, TensorArg, USABLE_LX_BYTES, WorkPlan, assert_df_stick_multiple,
};
use crate::wire::{
    AddrFold, AllocNode, ComputeAttrs, ComputeOp, Dsc, FoldFunc, FoldProp, IterSpace, LabeledDs,
    LayoutInfo, MemOrg, SdscFolds, SdscOp, StageParam, build_coordinates, core_dsc_schedule,
    gen_coord_info_value, segment_base,
};
use crate::work::{FP16_ELEMS_PER_STICK, core_to_wk_slice, distribute_cores};
use std::collections::BTreeMap;

/// THE WHOLE-FUNCTION DOOR — walk a multi-op KTIR program in program order and dispatch each
/// compute op to the per-kind entry point that already lowers it. A producer that hands over a
/// whole kernel (a Triton-derived `IRFunction` holding an MLP's two projections and its silu, or a
/// decoder layer's twelve `linalg.matmul`s) has no door otherwise: stating one `Program` for such a
/// function is refused by arity, correctly, and there is nothing else to state.
/// A batched gathered contraction — the gather's entry axis as the dot's batch axis. See the
/// module doc for the frame law and the honesty row.
pub mod batched_gathered;
/// main's `lower_matmul_node` arity-3 branch — the W8A8 fp8 matmul descriptors, reached from the
/// facts the KTIR states. See its own header.
pub mod ktir_matmul_fp8;
/// KTIR → SuperDSC: `ibm/main`'s eight `lower_*_node` bodies with their input door changed. The two
/// that cross a model-geometry const door ([`lower_ktir_to_superdsc::attn_at`],
/// [`lower_ktir_to_superdsc::rope_at`]) are public generic entry points the CALLER instantiates.
pub mod lower_ktir_to_superdsc;
pub mod whole_function;

/// A per-`layout`-dim field in the order dxp reads it: as stated, or REVERSED for a view that
/// states [`crate::superdsc_opspec::TensorArg::with_dxp_innermost_first`]. Every serialization of a
/// view's dim order — `layoutDimOrder_` on the alloc node and on `primaryDsInfo_`, `scale_` — goes
/// through here, so the three cannot disagree about which end of `layout` dxp reads first.
fn wire_order<T>(v: &ArgView<'_>, mut xs: Vec<T>) -> Vec<T> {
    if v.dxp_innermost_first {
        xs.reverse();
    }
    xs
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
    // ⭐ A CORE-LOCAL LX OPERAND STARTS AT ITS PIN ON EVERY CORE — no corner. Each core's scratchpad holds
    // only that core's own slice, and the op's coordinates (the same ones an HBM operand carries) say
    // which slice it is. The reference's softmax intermediates are exactly this: one LX start address
    // repeated on all 32 cores.
    if view.allocation.is_core_local() {
        return Ok(vec![intra_base; cores as usize]);
    }
    // LX residency is core-local: there is no HBM segment base to add, but the per-core slice
    // CORNER fold below still applies. A replicated producer ([`WorkPlan::replicate`]) walks the
    // whole tensor from the pin in EVERY core's private copy; a consumer's own slice corner then
    // selects its rows inside its core's copy — the SAME corner arithmetic as HBM, minus the
    // segment. (The old early return here handed every core the bare base: only the corner-0
    // case, correct at decode m=1 but silently row-0-pinning for a split consumer at m>1.)
    let base = if view.allocation.is_lx() {
        intra_base // LX uses a scratchpad-local pinned offset, no segment.
    } else {
        // `seg_base` is the RESOLVED HBM segment base for this tensor — the GLOBAL
        // role-segment from the BundleLayout (task #55) when the walk supplies one,
        // else the per-op `segment_base(arg_index)` fallback (single-op fixtures). The
        // per-core work-slice offset is added on top.
        seg_base + intra_base
    };

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
    // Stick guard at the OPERAND's device format: fp8/int8 = 128-elem stick (StickExtent<Fp8>
    // rejects a non-128-multiple at cargo build — the compile-time fp8 width guard), fp16/bf16 =
    // 64, fp32 = 32. The ONE exception — the packed fp8 kernel — is the `(Df::Fp8, true)` arm
    // below; see its own comment for why its law is 64-granular, not 128.
    let packed_fp8_kernel = matches!(view.role, Role::Kernel) && matches!(view.df, Df::Fp8);
    match (view.df, packed_fp8_kernel) {
        // A NON-KERNEL fp8/int8 operand (an activation, a dequant scale) keeps the FULL 128-law:
        // it is not retiled into the packed kernel form, so a sub-128 stick extent is a real
        // sub-stick tile dxp rejects (L3DlOpsScheduler:1040).
        (Df::Fp8 | Df::SenInt8, false) => {
            DeviceTileLayout::<Fp8>::new(layout, view.stick, &host_size)?;
        }
        // ⛔ THE PACKED fp8 KERNEL — the W8A8 matmul WEIGHT — stages as `[N/64, K/2, 64, 2]` (its
        // RetileDescriptor), so its STICK axis (`out`, N) is legal at any 64-multiple and its `in`
        // (K) at any even count, NOT the flat 128-law. This is not a weakening: every ADDRESSING
        // consumer already states the 64-law (`fp8_kernel_stage_off`, `gen_fp8_kernel_out_fold`'s
        // 8×8 N-subtile, the retile's own `N%64==0` guard — see
        // `work::FP8_PACKED_KERNEL_OUT_GRAN`), and the flat model here was the ONLY dissenting
        // reader. Inert on every fp8 model whose n was 128-aligned by luck (granite, g8b, g12b);
        // gemma-4-26b's GATE/UP projection (n=2112 = 33 packed out-sticks) is the first
        // 64-but-not-128-aligned N, and the flat model refused it.
        (Df::Fp8, true) => {
            // The packed kernel's own per-axis law: stick (out/N) at 64, in (K) at 2. Both axes
            // are named — the granularities cannot be attached to the wrong one.
            let stick_out =
                host_size[layout
                    .iter()
                    .position(|&d| d == view.stick)
                    .ok_or_else(|| {
                        SuperDscError(format!(
                            "packed fp8 kernel '{}': stick dim '{}' absent from layout {:?}",
                            view.name, view.stick, layout
                        ))
                    })?];
            if !stick_out.is_multiple_of(crate::work::FP8_PACKED_KERNEL_OUT_GRAN as u64) {
                return Err(SuperDscError(format!(
                    "packed fp8 kernel '{}': OUT extent {stick_out} is not a whole multiple of \
                     the {}-col packed-kernel stick (the RetileDescriptor's [N/64, K/2, 64, 2] \
                     tile — see work::FP8_PACKED_KERNEL_OUT_GRAN)",
                    view.name,
                    crate::work::FP8_PACKED_KERNEL_OUT_GRAN
                )));
            }
            if let Some(in_extent) = layout
                .iter()
                .zip(host_size.iter())
                .find(|&(&d, _)| d != view.stick)
                .map(|(_, &e)| e)
                && !in_extent.is_multiple_of(crate::work::FP8_PACKED_KERNEL_IN_GRAN as u64)
            {
                return Err(SuperDscError(format!(
                    "packed fp8 kernel '{}': IN (K) extent {in_extent} is not a whole \
                     multiple of the 2-packed fp8 K pair (the RetileDescriptor's \
                     [N/64, K/2, 64, 2] tile)",
                    view.name
                )));
            }
        }
        // Unreachable in practice (`packed_fp8_kernel` requires `Df::Fp8`), but the tuple match
        // cannot know that — an int8 KERNEL gets the same flat 128-law as every non-packed operand.
        (Df::SenInt8, true) => {
            DeviceTileLayout::<Fp8>::new(layout, view.stick, &host_size)?;
        }
        (Df::Fp32, _) => {
            DeviceTileLayout::<Fp32>::new(layout, view.stick, &host_size)?;
        }
        (Df::Fp16 | Df::Bf16, _) => {
            DeviceTileLayout::<Fp16>::new(layout, view.stick, &host_size)?;
        }
        // ⛔⛔⛔ A GATHER'S INDEX TABLE IS **EXEMPT FROM THE STICK-MULTIPLE GUARD**, and both vendor
        // fixtures settle that outright:
        // ```text
        // test_gather_1core   index ["mb"]     N_ mb_ = 3   KERNEL_IDX stickDimOrder_ ["mb"] stickSize_ [32]
        // sdsc_add_paged_l3lu index ["x","y"]  N_ y_  = 2   KERNEL_IDX stickDimOrder_ ["y"]  stickSize_ [32]
        // ```
        // Extents of THREE and TWO against a 32-entry stick. An index is not a tiled compute operand —
        // it is read into the IBR one 128-byte stick at a time (`senulator/memoryElement.cpp:810`) and
        // never staged to LX, which is the same property that forces its HBM-only `memOrg_` and that
        // makes `allocAllMem` skip it. `L3DlOpsScheduler:1040`, which the guard's own message cites,
        // rejects sub-stick TILES; there is no tile here.
        //
        // ⛔ NOT A WEAKENING TO GET PAST A REFUSAL. The guard was already being SATISFIED by picking a
        // different stick dim than the vendor picks (`out` rather than the innermost) — a different HBM
        // layout for the table the host stages, which is an address rather than a build error. Removing
        // it is what lets the stick dim be the vendor's own. `Df::SenUint32` occurs on index operands
        // and nowhere else (`attach_gather_index` is its only producer), so this arm cannot exempt
        // anything else.
        (Df::SenUint32, _) => {}
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
/// goes through, whether their `TileOp` came from here or from a caller that built its own.
fn pointwise_opspec(
    op: OpFunc,
    rows: u32,
    cols: u32,
    in_names: &[&str],
    o_name: &str,
    head_major: bool,
    layout: Option<&BundleLayout>,
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
    pointwise_opspec_from_tile(&tile_op, op, in_names, o_name, head_major, layout)
}

/// How one elementwise operand maps onto the `[mb, out, y]` iteration space — its
/// per-dim [`Scale`]. The OUTPUT is always all-`Active`; an INPUT may BROADCAST a
/// dim: `mb` broadcast (a `[1, cols]` row vector like RmsNorm's gamma) → `mb` =
/// `RedNonStick` (-1, a non-stick size-1 dim); `out` broadcast (a `[m, 1]` column
/// vector like RmsNorm's inv_rms, or a `[1,1]` scalar const) → `out` = `RedStick`
/// (-2, the one-stick `alpha_=0` broadcast READ proven on-card 2026-06-25). Both
/// are MaterializedStick-safe. `y` is `Active` for every operand but the ROW-SPLIT form
/// ([`In::y_broadcast`]): a `[mb, out]` source sprayed down the op's leading `y` dim carries
/// `y = RedNonStick` there — the vendor's own reduced-dim law (`superdsc.py`'s
/// `reduced_dims → scale = -1`), the same scale the on-card-proven pmask fusion uses.
#[derive(Clone, Copy)]
pub struct EwOperand<'a> {
    // Fields are PRIVATE and there are NO raw ctors: the ONLY way to build an EwOperand is `In::ew()`,
    // which requires a typed `&Stk<K>` handle. So a raw `EwOperand { name: &str, .. }` or the old
    // `EwOperand::full(&str)` is UNCONSTRUCTABLE outside this module — the weakly-typed input path is
    // eliminated. Every pointwise/reduce input carries its layout kind in its type via `In<K>`.
    name: &'a str,
    /// The handle's declared [`StickLayout`], captured by the `In` door that built this operand —
    /// the shape authority the family mint (`mint_ew_operands`) checks every swept window against.
    /// Carried IN the operand (not re-read from a handle at the funnel) so the checked shape is the
    /// one the operand's own construction saw, and a broadcast form can never outlive the handle it
    /// swept past.
    layout: StickLayout,
    mb_broadcast: bool,
    out_broadcast: bool,
    /// ⭐ THE ROW-SPLIT (Y) BROADCAST COUNT — 0 = not a y-broadcast; n ≥ 2 = the source is a
    /// `[mb, out]` tile sprayed down the op's leading `y` dim, n sprays deep. Set only by
    /// [`In::y_broadcast`], whose callers are the whole-function walk's Y admission (a
    /// leading-axis `linalg.broadcast` over a multi-row source whose result is `[y, mb, ..]`),
    /// so the count is the broadcast's OWN stated leading extent — not a caller's pick.
    y_broadcast: u32,
    col_offset: u32,
    /// The [`crate::addr::ColWindow`] `col_offset` came from — set only by [`In::col_window`], so a
    /// stick-major presentation of a column window is admitted by type, never by testing the integer.
    col_window: Option<crate::addr::ColWindow>,
    /// ⭐ THE POOLED MARK — the reach the gather door admitted this operand under, set only by
    /// [`mark_pooled`] on the gathered operand of a table whose layout declares a
    /// `BundleLayout::gather_reach` for it. The mark CARRIES the reach so the window the mint
    /// selects and the bound the door admitted under are one value, not two facts that could
    /// disagree; the mint then reads the FAMILY from the placement — a slice of a pool its
    /// index re-bases takes ONE ENTRY's direct coverage, a placement holding the WHOLE reach
    /// (a source extent) takes `min(swept, the reach's ceiling)`. Nothing else about the
    /// operand — bytes, scales, offsets — changes; the mark selects a bound, never a descriptor.
    pooled: Option<crate::sdsc_abstract::GatherReach>,
    /// ⭐ THE GATHERED MARK — set only by [`mark_gathered`] on the operand an index reads through
    /// ([`assemble_pointwise_broadcast_gather`]'s `gathered_input`) when the table declares NO
    /// reach. It selects the SOURCE-WINDOW LAW at the family mint ([`mint_ew_operands`]): a
    /// gathered operand's addressable set is the index's image over the source's rows, never
    /// the swept rectangle, so a sweep that fits the placement keeps its window and an
    /// over-sweep is REFUSED, fail-closed, until the table declares its reach (the
    /// `GatherReach` the pooled mark above then carries) — see that branch's doc for the law.
    /// Like [`pooled`], the mark selects a bound and never a descriptor.
    gathered: bool,
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
            if self.y_broadcast > 0 {
                Scale::RedNonStick
            } else {
                Scale::Active
            },
        ]
    }

    /// The SAME scales in Y-MAJOR order `[y, mb, out]` — the layout order the row-split mode's
    /// op presents its tensors in (the `TensorArg` layout of the y-major frame, the same
    /// `["y", "mb", "out"]` head-outermost spelling the batched matmul's walks use). A rotation
    /// of [`scale`](Self::scale)'s `[mb, out, y]`, so the y-major builder reads one ordered
    /// triple per operand instead of re-deriving it at its own call site.
    pub(crate) fn scale_ymajor(&self) -> [Scale; 3] {
        let s = self.scale();
        [s[2], s[0], s[1]]
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
    pub(crate) fn mb_broadcast(&self) -> bool {
        self.mb_broadcast
    }
    /// The row-split spray count — 0 when this operand is not a y-broadcast.
    pub(crate) fn y_broadcast(&self) -> u32 {
        self.y_broadcast
    }
    pub(crate) fn col_offset(&self) -> u32 {
        self.col_offset
    }
    /// Whether this operand is a [`crate::addr::ColWindow`] whose `[rows, ·]` view is stick-major.
    pub(crate) fn is_stick_major_window_for(&self, rows: u32) -> bool {
        self.col_window.is_some_and(|w| w.is_stick_major_for(rows))
    }
    /// ⭐ THE POOLED MARK — the same operand under the gather door's pooled-table law, carrying
    /// the reach it was admitted under. Set only by [`assemble_pointwise_broadcast_gather`] for
    /// the gathered operand of a table whose layout declares a `BundleLayout::gather_reach`; see
    /// the field's own doc for why the mark selects a bound and never a descriptor.
    pub(crate) fn mark_pooled(&self, reach: crate::sdsc_abstract::GatherReach) -> Self {
        Self {
            pooled: Some(reach),
            ..*self
        }
    }
    /// The reach this operand was admitted under, if the gather door pooled it (see
    /// [`mark_pooled`]).
    pub(crate) fn pooled(&self) -> Option<crate::sdsc_abstract::GatherReach> {
        self.pooled
    }
    /// ⭐ THE GATHERED MARK — the same operand under the gather door's source-window law. Set
    /// only by [`assemble_pointwise_broadcast_gather`] for the operand its index reads through;
    /// see the field's own doc for why the mark selects a bound and never a descriptor.
    pub(crate) fn mark_gathered(&self) -> Self {
        Self {
            gathered: true,
            ..*self
        }
    }
    /// Whether an index reads this operand through (see [`mark_gathered`]).
    pub(crate) fn gathered(&self) -> bool {
        self.gathered
    }
    /// The handle's declared [`StickLayout`], captured by the `In` door that built this operand.
    pub(crate) fn declared(&self) -> StickLayout {
        self.layout
    }
    /// The operand's SWEPT device-access window for a `[rows, cols]` op — the per-form law the
    /// family mint (`mint_ew_operands`) checks. `stick_elems` is the op's stick-padded swept
    /// width; `stick_lanes` one stick at the op's df:
    /// - full / sliced / [`In::col_window`]: `rows × stick_elems` — the op's whole extent, padded
    ///   lanes included (an `Active`-over-`out` operand reads the pad lanes of its last stick).
    /// - out-broadcast (a per-row scalar: `col`/`col_at`): `rows × stick_lanes` — ONE STICK PER
    ///   ROW, the RedStick `alpha_=0` broadcast READ.
    /// - mb-broadcast (a `[1, cols]` row vector: `mb`/`mb_at`): `1 × stick_elems`.
    /// - scalar (mb ∧ out, a `[1,1]` const): `1 × stick_lanes` — one stick.
    /// - y-broadcast (the row-split form: `y_broadcast`): `(rows/y) × stick_elems` — the
    ///   `[mb, out]` source at its own extent, one spray's worth of rows, at the SAME
    ///   stick-padded width the dense operands sweep (the pad lanes are the mint's own
    ///   footprint — the source region is single-stick by the frame law).
    pub(crate) fn swept_extents(
        &self,
        rows: u32,
        stick_elems: u32,
        stick_lanes: u32,
    ) -> (u32, u32) {
        if self.y_broadcast > 1 {
            return (rows / self.y_broadcast, stick_elems);
        }
        match (self.mb_broadcast, self.out_broadcast) {
            (true, true) => (1, stick_lanes),
            (true, false) => (1, stick_elems),
            (false, true) => (rows, stick_lanes),
            (false, false) => (rows, stick_elems),
        }
    }
}

/// ⭐⭐⭐ THE POINTWISE FAMILY'S OPERAND MINT — the #731 door for every elementwise input.
///
/// Each operand's SWEPT window ([`EwOperand::swept_extents`] — the per-form broadcast law) is
/// checked against BOTH authorities, the same doors [`BundleLayout::win`] runs for a
/// handle-minted [`Win`]: the handle's declared shape (always — the `In` door captured it when
/// the operand was built), and the placement footprint by name (when a `BundleLayout` is at
/// hand). An operand that reads past its tensor is an `Err` naming the tensor, its window and
/// the bound it exceeded — the class #717's card failure belongs to (`add_o1137` reading
/// `t1131` past its footprint at the width-512 prefix-capable continuation), refused at build
/// instead of detected on card. A gathered operand's window is its table's own rows (the index
/// selects within the table by contract), so a table smaller than the op's row count is caught
/// here too.
pub(crate) fn mint_ew_operands(
    op_name: &str,
    rows: u32,
    cols: u32,
    df: Df,
    inputs: &[EwOperand<'_>],
    layout: Option<&BundleLayout>,
) -> Result<(), String> {
    // The op's stick basis — the same two forms `pointwise_broadcast_opspec_df` tiles at: fp16's
    // 64-stick, or fp32's 32 for the torch-spyre fp32 rmsnorm island.
    let stick_elems = if matches!(df, Df::Fp32) {
        StickExtent::<Fp32>::new(cols)?.elems()
    } else {
        StickExtent::<Fp16>::new(cols)?.elems()
    };
    let stick_lanes = df.elems_per_stick();
    for i in inputs {
        let (sw_rows, sw_cols) = i.swept_extents(rows, stick_elems, stick_lanes);
        let what = format!("{op_name}: operand '{}'", i.name());
        // ⭐ THE OUT-BROADCAST (RedStick) FORMS' DEVICE ACCESS IS THE STICK **RUN** CONTAINING THE
        // OFFSET — the L3 EBR is stick-granular and rounds the reader's allocate down to the stick
        // base (rmsnorm's `rinv` half-stick law: the sub-stick remainder is a lane select inside
        // the stick, byte-granular on LX), and each op row steps one stick. So the window the two
        // doors bound starts at the stick BASE, while the VALUE each row reads sits at the offset
        // itself — stated as its own bound, so a sub-stick offset can neither overstate the access
        // (refusing a legal lane select) nor hide a scalar past the tensor's end. Every other form
        // reads a contiguous range from its offset and keeps the exact bound.
        let off = if i.out_broadcast() {
            let shape_elems = i.declared().rows as u64 * i.declared().cols as u64;
            // The per-row forms (col/col_at) step one op row per stick — the operand's own row
            // pitch. The scalar form (mb ∧ out) reads ONE value for the whole op, so no pitch.
            let stride = if i.mb_broadcast() {
                0
            } else {
                let pitch = i.declared().cols as u32;
                if pitch != stick_lanes {
                    return Err(format!(
                        "{what}: the broadcast operand is {pitch} columns wide — a per-row \
                         scalar broadcast reads ONE STICK PER ROW, and no bound is stated for a \
                         wider one"
                    ));
                }
                stick_lanes as u64
            };
            let last_value = i.col_offset() as u64 + (sw_rows.saturating_sub(1) as u64) * stride;
            if last_value >= shape_elems {
                return Err(format!(
                    "{what}: the broadcast read's last row scalar (element {last_value}) runs \
                     past the end its handle declares — {shape_elems} elements. An op that reads \
                     past its own tensor is a defect in the computation that produced this \
                     offset; it cannot compile."
                ));
            }
            i.col_offset() - i.col_offset() % stick_lanes
        } else {
            i.col_offset()
        };
        match layout {
            Some(l) => {
                // ⭐⭐⭐ A POOLED GATHERED TABLE'S WINDOW IS ONE ENTRY'S COVERAGE — the mark the
                // gather door sets when the layout declares a `gather_reach` for this table. Its
                // placement is ONE slice of a pool the index re-bases, so the swept rectangle
                // `sw_rows × sw_cols` is not this operand's access model (its rows are
                // index-named, never contiguous) and the slice placement is not its bound. What
                // this mint bounds instead is the DIRECT window: the bytes at the base before
                // the index moves it, which is ONE entry — and an entry is the reach's own
                // `entry_elems` (the mark CARRIES the value, so the window selected here and the
                // bound the door admitted under are one number, never two facts that could
                // disagree). The VALUES' bound is not here at all: it is the ceiling the
                // host-side `GatherEntry` composition mints through, unconstructible past it.
                if let Some(reach) = i.pooled() {
                    let d = i.declared();
                    let entry = reach.entry_elems() as usize;
                    // ⭐ THE REACH-DECLARED WINDOW, BY FAMILY (law (b)'s one law,
                    // [`BundleLayout::gathered_source_rows`]): the mark CARRIES the reach, so
                    // the window selected here and the bound the door admitted under are one
                    // number, never two facts that could disagree. A placement holding the
                    // WHOLE reach is a SOURCE EXTENT — the window is `min(swept, the reach's
                    // ceiling)`; a placement holding a SLICE is a POOL — the swept rectangle
                    // is not this operand's access model (its rows are index-named, never
                    // contiguous) and the runtime re-bases, so the direct window is ONE
                    // entry's coverage. The VALUES' bound is not here at all: it is the
                    // ceiling the host-side composition mints through, unconstructible past it.
                    let rows =
                        l.gathered_source_rows(i.name(), sw_rows, entry as u32, d.df, Some(reach))?;
                    l.check_window(
                        i.name(),
                        StickLayout::row_blocked_df(rows as usize, entry, d.df),
                        off as u64,
                        rows,
                        entry as u32,
                        &what,
                    )?
                } else if i.gathered() {
                    // ⭐⭐⭐ LAW (b) — A GATHERED SOURCE'S WINDOW IS ITS SWEPT EXTENT WHEREVER
                    // THE SWEEP FITS THE PLACEMENT, AND AN OVER-SWEEP IS REFUSED, FAIL-CLOSED,
                    // WITHOUT A DECLARED REACH. For a gathered source the addressable set is
                    // the index's IMAGE over the source's rows — `addr = idx·skip_addr + base`,
                    // one source row per entry — so the bytes this op can touch are the rows
                    // the INDEX names, never the contiguous `sw_rows × sw_cols` rectangle the
                    // plain window model assumes. Where the image's worst case (the whole
                    // source) fits the placement, the swept window is honest and kept byte for
                    // byte; where the sweep EXCEEDS the placement, the placement alone cannot
                    // bound the values — a repeat gather over a source placed whole may exceed
                    // it (each row read many entries' worth of times) while a pool slice hides
                    // a source the layout cannot see — so the over-sweep must carry the TYPED
                    // statement of its values' bound (a `GatherReach`; this branch is the
                    // NO-reach mark, so it refuses here, naming the sweep, the placement and
                    // the missing declaration). The reach-declared form takes the pooled mark
                    // above, one law, family-forked.
                    //
                    // ⛔ TABLE-VALUE CORRECTNESS IS NOT THIS GUARD'S: whether the staged values
                    // really name in-range rows is the staging family's own law at birth, not a
                    // fact a window over the placement could check — the same division the
                    // reach-declared branch's doc states for its ceiling.
                    let d = i.declared();
                    let rows = l.gathered_source_rows(i.name(), sw_rows, sw_cols, d.df, None)?;
                    l.check_window(
                        i.name(),
                        StickLayout::row_blocked_df(rows as usize, sw_cols as usize, d.df),
                        off as u64,
                        rows,
                        sw_cols,
                        &what,
                    )?
                } else {
                    l.check_window(i.name(), i.declared(), off as u64, sw_rows, sw_cols, &what)?
                }
            }
            None => crate::placement::window_fits_declared(
                i.name(),
                i.declared(),
                off as u64,
                sw_rows,
                sw_cols,
                &what,
            )?,
        }
    }
    Ok(())
}

/// ⭐ The pointwise family's OUTPUT mint — the output window `rows × stick_elems` at the corner
/// the assembler was given, checked against the output handle's declared shape and (when at
/// hand) its placement. The `(Stk, OutCorner)` pair stays typed at the assembler — the corner's
/// [`crate::addr::ColWindow`] provenance is what the stick-major predicate reads, so folding it
/// into a `Win` would lose it — and THIS is the door that refuses a corner running past the
/// output's own end.
pub(crate) fn mint_ew_output<O: KindTag>(
    op_name: &str,
    o: &Stk<O>,
    off_elems: u32,
    rows: u32,
    cols: u32,
    df: Df,
    layout: Option<&BundleLayout>,
) -> Result<(), String> {
    let stick_elems = if matches!(df, Df::Fp32) {
        StickExtent::<Fp32>::new(cols)?.elems()
    } else {
        StickExtent::<Fp16>::new(cols)?.elems()
    };
    let what = format!("{op_name}: output '{}'", o.name());
    mint_full_window(&what, o, off_elems as u64, rows, stick_elems, layout)
}

/// ⭐ THE FULL-WINDOW MINT — one whole `[rows, cols]` window at an element corner, checked against
/// BOTH authorities, the same doors [`mint_ew_operands`] runs for a pointwise operand: the handle's
/// declared shape (always), and the placement footprint by name (when a `BundleLayout` is at hand).
///
/// This is the convert and restickify families' door. Their ops read and write whole tiles at
/// nest-derived corners — no broadcast forms, no per-row value bounds — so the one question a mint
/// can ask of them is whether the tile fits the tensor it names, and this asks exactly that. The
/// corner arrives in ELEMENTS of the operand's own df (a `DevOff`'s raw elements, or 0), and the
/// handle's own df must be the operand's real one: the placement door prices bytes at the handle's
/// word length, so an fp16-annotated fp8 buffer overstates every window 2× and would refuse legal
/// slots (see [`rb_df`]).
pub(crate) fn mint_full_window<O: KindTag>(
    what: &str,
    h: &Stk<O>,
    off_elems: u64,
    rows: u32,
    cols: u32,
    layout: Option<&BundleLayout>,
) -> Result<(), String> {
    match layout {
        Some(l) => l.check_window(h.name(), h.layout(), off_elems, rows, cols, what),
        None => crate::placement::window_fits_declared(
            h.name(),
            h.layout(),
            off_elems,
            rows,
            cols,
            what,
        ),
    }
}

/// Whether a `[rows, cols]` fp16 pointwise op over `n_inputs` operands runs WITHOUT time-tiling — the
/// very tile op and tiler [`pointwise_broadcast_opspec_df`] builds, asked before the op is. A caller
/// cuts an op this answers `false` for, because a time-tiled op with a broadcast operand is refused
/// (`concrete_trips` would advance the broadcast operand per trip).
pub fn pointwise_fits_untiled(rows: u32, cols: u32, n_inputs: u32) -> bool {
    pointwise_untiled_slice_bytes(rows, cols, n_inputs).is_some()
}

/// Whether a `[rows, cols]` fp16 ROW reduction (`max`/`sum` over `cols`) runs WITHOUT time-tiling — the
/// very tile op the reduce builder (`reduce_opspec_off`) tiles, asked before the op is. A caller cuts a
/// reduction this answers `false` for: a chunk reduction's time trips do not combine on card (see
/// `attn`'s `no_chunk_op_is_time_tiled`).
pub fn reduce_fits_untiled(rows: u32, cols: u32) -> bool {
    let Ok(stick) = StickExtent::<Fp16>::new(cols) else {
        return false;
    };
    let dim = |name: &'static str, size: u32, is_reduction: bool, is_stick: bool| ItDim {
        name,
        size,
        is_reduction,
        is_stick,
        df: Df::Fp16,
    };
    crate::ir::island::tile_op::TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands: 2 },
        dims: vec![
            dim("mb", rows, false, false),
            dim("out", stick.elems(), true, true),
            dim("y", 1, false, false),
        ],
        df: Df::Fp16,
    }
    .tile(
        crate::superdsc_opspec::MaxCores::<{ crate::superdsc_opspec::MAX_CORES }>,
        crate::work::distribute_cores,
        "out",
    )
    .is_ok_and(|t| t.time_tile.is_none())
}

/// The bytes of ONE core's slice of that op's fp16 `[rows, cols]` output under the tiler's own work
/// division — what a core-local LX pin for it must hold ([`BundleLayout::synth_lx_core_local`]). `None`
/// when the op would be time-tiled, which a core-local tensor cannot be.
pub fn pointwise_untiled_slice_bytes(rows: u32, cols: u32, n_inputs: u32) -> Option<u64> {
    pointwise_untiled_slice_bytes_divided(rows, cols, n_inputs, crate::work::EwDivision::Law)
}

/// [`pointwise_untiled_slice_bytes`] for ops assembled under a STATED `division`
/// ([`assemble_pointwise_broadcast_divided`]), so a core-local pin is sized under the division its ops
/// state. The pinned tensor is core-local by definition, so the division resolves as it does for every op
/// touching one ([`crate::work::EwDivision::resolve`]): the law gives rows first, and a stated division
/// gives itself.
pub fn pointwise_untiled_slice_bytes_divided(
    rows: u32,
    cols: u32,
    n_inputs: u32,
    division: crate::work::EwDivision,
) -> Option<u64> {
    let stick = StickExtent::<Fp16>::new(cols).ok()?;
    let dim = |name: &'static str, size: u32, is_stick: bool| ItDim {
        name,
        size,
        is_reduction: false,
        is_stick,
        df: Df::Fp16,
    };
    let tile_op = crate::ir::island::tile_op::TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce {
            n_operands: n_inputs + 1,
        },
        dims: vec![
            dim("mb", rows, false),
            dim("out", stick.elems(), true),
            dim("y", 1, false),
        ],
        df: Df::Fp16,
    };
    tile_op
        .tile(
            crate::superdsc_opspec::MaxCores::<{ crate::superdsc_opspec::MAX_CORES }>,
            // `false` for the spray fact: this sizing helper's tile states `y=1`, so the row-split
            // (y-major) mode — the only presentation whose division that fact selects — cannot
            // reach it.
            division.resolve(true, true, false).splitter(),
            "out",
        )
        .ok()
        .filter(|t| t.time_tile.is_none())
        .map(|t| {
            t.plan
                .dims()
                .iter()
                .map(|d| u64::from(t.plan.per_core_extent(d.name)))
                .product::<u64>()
                * u64::from(Df::Fp16.word_length())
        })
}

/// Build the typed [`OpSpec`] for ONE elementwise op over `[rows, cols]` with
/// PER-OPERAND broadcast (the generalization of [`pointwise_opspec`], which is the
/// all-`full` case). Inputs may broadcast over `mb` or `out` ([`EwOperand`]); the
/// output spans full `[mb,out,y]`. `out` is the 64-stick axis. Broadcast over the
/// stick (`out`) is proven on-card (RedStick `alpha_=0` fold). A TIME-TILED op MAY
/// carry a broadcast operand: [`rewrite_op_for_time_tile`] exempts any operand whose
/// scale along the tiled dim is not `Active`, so its address never advances per trip
/// (pinned by `time_tiled_broadcast_operand_is_address_stable`).
/// `df` is `Df::Fp16` (the 64-stick default) for every caller but the torch-spyre fp32 RMSNorm.
/// `layout` is the bundle's, so the division can see which operands are core-local LX pins, and
/// `division` is [`crate::work::EwDivision::Law`] for every shipped op (see there).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pointwise_broadcast_opspec_df(
    op: OpFunc,
    rows: u32,
    cols: u32,
    inputs: &[EwOperand<'_>],
    o_name: &str,
    out_offset: crate::addr::OutCorner,
    head_major: bool,
    df: Df,
    layout: Option<&BundleLayout>,
    division: crate::work::EwDivision,
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
        &tile_op,
        rows,
        cols,
        op,
        inputs,
        o_name,
        out_offset,
        head_major,
        layout,
        // This door is the ungathered family's: the elementwise law, wrapped for the shared
        // builder's `CoreDivision` (a gathered op's own split is a `GatherSplit`, minted by its
        // leg — see `one_gathered_leg`).
        crate::work::CoreDivision::Ew(division),
    )
}

/// [`pointwise_broadcast_opspec_df`] → [`EmittedOp`].
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
        indirect: None,
        coordinate_mask: None,
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
        PointwiseFacts {
            op_name,
            op_func,
            rows,
            cols,
            inputs,
            o,
            out_offset: crate::addr::DevOff::ZERO.into(),
        },
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
        PointwiseFacts {
            op_name,
            op_func,
            rows,
            cols,
            inputs,
            o,
            out_offset: crate::addr::DevOff::ZERO.into(),
        },
        sym_id_base,
        layout,
    )
}

/// The operands and geometry of a GATHERED pointwise op — the parameter list of
/// [`assemble_pointwise_broadcast_gather`], carried as a struct.
///
/// A struct rather than nine positional parameters because the sibling assemblers below predate that
/// convention and each carries an `#[allow(clippy::too_many_arguments)]`; this follows
/// [`lower_ktir_to_superdsc::rope_at`]'s `RopeAt` instead, which is the newer shape in this crate and
/// adds no sixteenth `allow`.
pub struct PointwiseGather<'a, O: KindTag> {
    /// The descriptor's name.
    pub op_name: &'a str,
    /// The tile the KTIR program states. Its `dims` are the axis names the gather's pin is resolved
    /// against, which is why the tile is passed rather than rebuilt from `rows`/`cols`.
    pub tile_op: &'a crate::ir::island::tile_op::TileOp,
    /// The `opFuncName` — a gather does NOT change it. `dxp/test/test_gather_1core/sdsc_1.json` is an
    /// ordinary `identity` carrying one extra `labeledDs_` and one extra `computeOp_` field, so the
    /// arithmetic this op names is whatever the program said.
    pub op_func: &'static str,
    pub rows: u32,
    pub cols: u32,
    /// The op's COMPUTE inputs, with the index buffer ALREADY EXCLUDED — that exclusion is
    /// [`lower_ktir_to_superdsc::split_out_excluding`]'s, driven by the program.
    pub inputs: &'a [EwOperand<'a>],
    /// WHICH of `inputs` is read through the index. Must be the LAST input — see the refusal.
    pub gathered_input: usize,
    /// The index buffer's operand spelling — `crate::place::act_name(index_tid)`, the same form every
    /// data operand here uses, so it resolves through the SAME [`BundleLayout`].
    pub index_name: &'a str,
    /// ⭐ THIS GATHER'S INDEX RUN — the stick-aligned start (the program's own anchor, in ENTRIES:
    /// zero for every gather whose index is read from its start; an unrolled sweep's trip `b`
    /// states `b · BLOCK_N`) plus the WHOLE STICKS the node's entries span. A [`StickRun`], so a
    /// mid-stick run cannot be stated: the per-J merge form's run at entry 8 has no spelling, and
    /// a kernel asking for one is refused at the walk that builds this value. Each leg below is
    /// the run's own [`StickRun::leg`] — one stick, `k` sticks in — so the leg stride and the
    /// run's span are the same unit by construction (issue 201 item 7's run half).
    pub run: crate::superdsc_opspec::StickRun,
    pub o: &'a Stk<O>,
}

/// ⭐ THE POOL ADMISSION — ONE LAW, TWO DOORS. A table whose layout declares a
/// `BundleLayout::gather_reach` is a POOL SLICE, not a table: its placement covers one slice (the
/// paged KV cache's per-layer plane) while its index names stick-blocks RELATIVE TO THE PLANE'S
/// OWN BASE across the whole pool the runtime re-bases per launch, so the swept-rectangle law is
/// the wrong pair of facts for it. What replaces it is ONE check, on the one fact the admitting
/// door uniquely sees:
///
/// THE ENTRY UNIT. dxp turns an entry value into an address as `idx * skip_addr + base`, and a
/// gathered read of this pool strides one entry at `entry_elems` elements — while the reach's
/// entries are `entry_elems`-element stick blocks, the unit the host's `GatherEntry::of_page_block`
/// composes its runtime values in. Two units for one address is two authorities for the same
/// bytes: a descriptor striding at any other width reads every entry at the wrong address, from a
/// clean bake — refused here, by name, with both numbers.
///
/// ⛔ THE VALUES' BOUND IS NOT A CHECK HERE, because it cannot be one: the values are runtime data
/// and any run-shaped inequality against the reach compares two different tensors' coordinates.
/// The honest bound is the ceiling the reach carries, enforced where a value is BORN (the
/// host-side `gather_entry_ceiling` → `GatherEntry::of_page_block` composition); the extent law
/// holds by TYPE (`skip_addr` equal to `entry_elems` satisfies the reach's derived element extent
/// exactly). A table with NO declared reach returns `None` and takes none of this — its gather
/// keeps the swept-window law byte for byte.
///
/// Both doors that admit a gathered read of a pool run THIS function — the pointwise gather door
/// (which then marks its operand pooled, so the operand mint takes the one-entry window) and the
/// batched gathered contraction's copy — so the two cannot drift about what a pool's entry unit
/// is. Returns the reach when the table is pooled (after the entry-unit check), `None` when no
/// reach is declared for it.
pub(crate) fn pooled_reach_of(
    what: &str,
    layout: Option<&BundleLayout>,
    table: &str,
    entry_elems: u32,
) -> Result<Option<crate::sdsc_abstract::GatherReach>, SuperDscError> {
    let Some((tid, reach)) = layout.and_then(|l| match l.id_of(table) {
        Some(crate::place::PlaceId::Act(tid)) => l.gather_reach.get(&tid).map(|&r| (tid, r)),
        _ => None,
    }) else {
        return Ok(None);
    };
    if entry_elems as u64 != reach.entry_elems() {
        return Err(SuperDscError(format!(
            "{what}: this gathered read of '{table}' (t{tid}) strides one index entry at \
             {entry_elems} elements, but the bundle declares that pool's entries as {}-element \
             stick blocks (a ceiling of {} entry values, {} elements of reach). dxp turns a value \
             into an address as `idx * skip_addr + base`, and the host composes those values in \
             the pool's own unit (`GatherEntry::of_page_block`), so a descriptor striding at any \
             other width reads every entry at the wrong address — from a clean bake. State the \
             read at the pool's entry width, or declare the reach for the pool this read really \
             strides (`BundleLayout::gather_reach`).",
            reach.entry_elems(),
            reach.max_entry().get(),
            reach.reach_elems(),
        )));
    }
    Ok(Some(reach))
}

/// ⭐⭐⭐ A POINTWISE OP ONE OF WHOSE INPUTS IS **GATHERED** — a row lookup `table[ids[row], :]`.
///
/// ## It is a CALLER of the gather, not a second gather
/// Every field of the declaration comes from [`OpSpec::attach_gather_index`], which is already what
/// [`crate::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec`] and
/// [`crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec`] call. This is the third caller of that one
/// door, so `Role::Index`, `Df::SenUint32`, the cross-linked alloc pair, `indirectAccessIndexLabeledDs`,
/// the HBM-only `memOrg_` and the `maxDimSizes_` pin are all the emitter's existing behaviour reached
/// from a new op shape — nothing about the marking is restated here.
///
/// ## ⭐ WHY A TABLE LOOKUP IS ONE PIN ON `mb` AND NOT THE PAGED-KV TWO
/// [`GatherIndex::of_pool_blocks_per_row`] pins the slot axis for the page AND `mb` for the row, because
/// a KV read has both a page coordinate and a request coordinate. A table lookup has only the row: one
/// entry IS one row, so `entry_dim` is the op's own `mb` and `page` is
/// [`PageExtent::single_position`]. dxp then derives `skip_addr = 1 x the per-core extents of the
/// unpinned dims` = `cols`, i.e. one whole table row per index step, and the entries the host stages
/// are plain row numbers. That pin is the vendor fixture's own shape —
/// `zz_the_index_tensors_shape_is_the_paged_dims.rs` records `test_gather_1core` as
/// `value maxDimSizes_ [1,-1,-1]` against `index layoutDimOrder_ ["mb"]`, and its op is an `identity`,
/// not an attention.
///
/// ## ⛔ `cols` IS NOT LIMITED TO ONE STICK HERE, AND THAT IS NOT A WEAKENING OF `gather_copy_opspec`
/// That builder refuses `cols != 64` because ITS destination must be CONTIGUOUS: a matmul kernel then
/// reads the scratch per-batch at a stride dxp DERIVES, so a stick-major destination hands the score
/// kernel another request's slots (its own "WHY THE DESTINATION BEING CONTIGUOUS IS THE POINT"). This
/// op has no such consumer — its destination is an ordinary activation, written stick-major and read
/// stick-major by the rest of the model, which is what every other pointwise here emits. An embedding
/// row is `d_model` wide (4096 = 64 sticks) and must stay one row.
///
/// The half of that file's reasoning which DOES transfer is `skip_addr`, and it is enforced below.
///
/// ## ⛔ FALLIBLE, unlike its non-gathering siblings
/// They panic because they run inside scratchy's `#[forward]` proc-macro, where a panic IS the build
/// error. This door is reached from a third-party KTIR producer's driver, and every refusal below is a
/// statement about the PROGRAM rather than about this crate — so it returns them.
///
/// ## ⭐⭐⭐ AND IT RETURNS **MANY** OPS, BECAUSE ONE GATHER OP'S INDEX IS ONE STICK
/// The index reaches the L3LU IBR as a single `SenUint32` stick transfer
/// ([`crate::sdsc_abstract::CopyDims::ENTRIES_PER_OP`] entries), so a node with more entries than that
/// is emitted as one op per stick, each writing its own ROW WINDOW of the output in place — no scratch
/// and no relayout. The body carries the three card measurements that fixed that shape: what the
/// un-cut op did (`ids[r mod 32]`), what a row-major destination does (a PCIe bus fence), and which
/// wire field the window's stick-plane stride actually comes from (`N_`, not `maxDimSizes_`).
pub fn assemble_pointwise_broadcast_gather<O: KindTag>(
    g: PointwiseGather<'_, O>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let PointwiseGather {
        op_name,
        tile_op,
        op_func,
        rows,
        cols,
        inputs,
        gathered_input,
        index_name,
        run,
        o,
    } = g;
    // ⛔ THE GATHERED INPUT MUST BE THE **LAST** ONE, because that is the only position at which the
    // two placement laws coincide. `attach_gather_index` inserts the index at `out_idx` — immediately
    // before the output, which is where the vendor puts it (`[input, index, output]`) and which keeps
    // the output last however many operands are added. dbo separately needs the index to DOMINATE its
    // use (`DSC2ToDataflowIR.cpp:51` reports `operand #1 does not dominate this use` otherwise), i.e.
    // to sit next to the tensor it indexes. With the gathered input last, "before the output" and
    // "after the value" are the same slot; with it earlier they are not, and this function cannot
    // choose between them from what it is given. A program that gathers an earlier input is refused
    // rather than emitted at whichever position happens to be reachable.
    let last_input = inputs.len().checked_sub(1).ok_or_else(|| {
        SuperDscError(format!(
            "{op_name}: a gathered pointwise op has no inputs — there is no tensor for the index to \
             select rows of"
        ))
    })?;
    if gathered_input != last_input {
        return Err(SuperDscError(format!(
            "{op_name}: input {gathered_input} of {} is named as the gathered operand, but the index \
             is inserted immediately before the OUTPUT (`attach_gather_index`, the vendor's \
             `[input, index, output]` order) and dbo requires it to sit next to the tensor it indexes \
             (`operand #1 does not dominate this use`, DSC2ToDataflowIR.cpp:51). Those are the same \
             slot only for the LAST input. Reorder the op's inputs so the gathered one is last.",
            inputs.len()
        )));
    }
    // ⭐ THE OUTPUT MINT for the WHOLE NODE: the legs below each write a row window of the SAME
    // output in place (together `[0, rows)`), so the node's full window is the bound every leg's
    // write sits inside — `first_row·lanes + leg_rows·stick_elems ≤ rows·stick_elems` holds for
    // every leg because `lanes ≤ stick_elems` and `first_row + leg_rows ≤ rows`. Checked once
    // here, against the output handle's declared shape and placement. The legs' INPUTS are
    // minted per-leg in `pointwise_broadcast_opspec_from_tile` (the leg's own tile carries
    // `leg_rows`, so each broadcast form is checked at the rows that actually sweep it).
    mint_ew_output(op_name, o, 0, rows, cols, tile_op.df, layout).map_err(SuperDscError)?;
    // ⭐ THE ENTRY COUNT, DERIVED BEFORE ANY OP IS BUILT — because it decides how many ops there are.
    // The page is [`PageExtent::single_position`] for a table lookup (one entry IS one row, see the
    // header); it is named ONCE here and handed to every leg, so the number this function counts
    // entries with and the number `attach_gather_index` declares cannot differ.
    let page = crate::superdsc_opspec::PageExtent::single_position();
    let entries = page.entries_in(rows).ok_or_else(|| {
        SuperDscError(format!(
            "{op_name}: the gather pages `mb` by {} position(s) and this op's extent along it is \
             {rows}, which does not divide into whole entries — a partial entry is an address.",
            page.get(),
        ))
    })?;
    let cap = crate::sdsc_abstract::CopyDims::ENTRIES_PER_OP;
    // ⛔ THE RUN'S SPAN AND THE NODE'S ENTRIES ARE THE SAME FACT STATED TWICE, AND THE TYPE IS WHAT
    // LETS THEM BE CHECKED AGAINST EACH OTHER. The cut below emits one op per index stick, so the
    // stick count it cuts into IS the run's span; a caller that hand-built a shorter run (or
    // stated the node's rows independently of the run it handed) would emit legs reading past the
    // sticks the run declared — another run's words, converted as this one's page addresses.
    // Refused here, by name, where both numbers are in hand.
    let legs = entries.div_ceil(cap);
    if legs != run.sticks() {
        return Err(SuperDscError(format!(
            "{op_name}: this node consumes {entries} index entries, which spans {legs} stick(s), but \
             its gather run declares a span of {} stick(s) starting at stick {}. The cut emits one \
             op per stick of the RUN, so the two must agree — a run shorter than the node reads \
             index sticks it never declared, and a longer one leaves sticks no leg loads.",
            run.sticks(),
            run.start().sticks(),
        )));
    }
    // ⭐⭐⭐⭐⭐ THE POOLED TABLE — DESCRIBED AS IT REALLY LIVES. A gathered source whose layout
    // declares a `BundleLayout::gather_reach` is a POOL SLICE, not a table: its placement covers
    // one slice (the paged KV cache's per-layer plane) while its index names stick-blocks
    // RELATIVE TO THE PLANE'S OWN BASE across the whole pool the runtime re-bases per launch,
    // so the swept-rectangle-vs-placement law the legs' mint runs for every other operand is
    // the WRONG pair of facts for it — the rectangle `rows × cols` is not contiguous and the
    // placement is not the pool. What replaces it is ONE check, on the one fact this door
    // uniquely sees:
    //
    // THE ENTRY UNIT. dxp turns an entry value into an address as `idx * skip_addr + base`,
    // and this descriptor strides one entry at `cols` elements — while the reach's entries are
    // `entry_elems`-element stick blocks, the unit the host's `GatherEntry::of_page_block`
    // composes its runtime values in. Two units for one address is two authorities for the
    // same bytes: a descriptor striding at any other width reads every entry at the wrong
    // address, from a clean bake — refused here, by name, with both numbers.
    //
    // ⛔ AND THE VALUES' BOUND IS NOT A CHECK HERE, BECAUSE IT CANNOT BE ONE. The values are
    // runtime data (the host's index table); `run.start()` is a POSITION in the index buffer,
    // not a value, so any run-shaped inequality against the reach compares two different
    // tensors' coordinates and is vacuous on every real bundle. The honest bound on the
    // VALUES is the ceiling the reach carries — `(page_ceiling − 1) · per_page + plane_blocks`
    // — and it is enforced where a value is BORN: the host-side composition
    // (`gather_entry_ceiling` → `GatherEntry::of_page_block`) refuses every value at or above
    // it, and the same door minted the reach the producer declared, so the bake-time and
    // stage-time halves are one number by construction. The extent law — the descriptor's
    // `skip_addr` times the ceiling stays inside the reach — holds by TYPE: the reach's
    // element extent IS `ceiling · entry_elems` ([`GatherReach::reach_elems`], derived, never
    // stored), so a `skip_addr` equal to `entry_elems` satisfies it exactly and any other
    // `skip_addr` is the refusal below.
    //
    // A table with NO declared reach takes none of this: its gather keeps the swept-window law
    // exactly as the embedding's card-proven emission runs it, byte for byte.
    let mut pooled_inputs;
    let reach_of_table = pooled_reach_of(op_name, layout, inputs[gathered_input].name(), cols)?;
    let inputs = if let Some(reach) = reach_of_table {
        pooled_inputs = inputs.to_vec();
        pooled_inputs[gathered_input] = pooled_inputs[gathered_input].mark_pooled(reach);
        &pooled_inputs[..]
    } else {
        // ⭐ LAW (b)'s MARK — the operand the index reads through takes the source-window law
        // at the family mint even when the table is NOT a pool: a sweep that fits its
        // placement keeps its window byte for byte, and an over-sweep is REFUSED here,
        // fail-closed, until the table declares a reach (the typed bound on its index values —
        // a pool's `GatherReach::of_plane`, or a whole-placed source's
        // `GatherReach::of_source_extent`, which the mark above then carries). See the mark's
        // own doc and `mint_ew_operands`'s gathered branch for the law.
        pooled_inputs = inputs.to_vec();
        pooled_inputs[gathered_input] = pooled_inputs[gathered_input].mark_gathered();
        &pooled_inputs[..]
    };
    // 🅿️ PARKED — THE GENERAL-DOOR STAMP FOR POOL LEGS IN NON-CARRIER PROGRAMS. This door's
    // pool legs (a `GatherReach::of_plane` table reached here, in a program that is NOT a
    // spliced attention carrier) are UNSTAMPED by the unified gathered-fact slice: the
    // addressing fact is stated at the CARRIER's mint (`EmittedOp::paged_region`, the one
    // door) and stamped by the walk onto every op it lowers from that carrier — so a door
    // stamp here would be a SECOND writer for the same fact on the same ops, which the
    // one-stamping-site law forbids. No such program exists in the tree's models today
    // (every live pool gather rides an attention carrier); the door-law stamp for that
    // class is PARKED, not silently dropped — landing it means minting the carrier fact
    // for that program class first.
    // The OUTPUT kind drives the layout, exactly as in the sibling assemblers: a `Stk<FlatTag>` output
    // is a head-major rank-3 op, a `Stk<RowBlockedTag>` output is the rank-2 stick-major token stream.
    let head_major = O::kind() == StickKind::Flat;
    // ⭐ ONE INDEX STICK ⇒ ONE OP OVER THE WHOLE NODE. Byte-for-byte what this door emitted before the
    // cut existed, which is what keeps `embedding_granite_m32`'s card-proven descriptor unmoved.
    if entries <= cap {
        return Ok(vec![one_gathered_leg(
            GatherLeg {
                op_name,
                leg_name: op_name,
                tile_op,
                op_func,
                leg_rows: rows,
                node_rows: rows,
                first_row: 0,
                cols,
                inputs,
                gathered_input,
                index_name,
                page,
                // ⭐ THE NODE'S OWN RUN — the whole run is the program's, uncut (an uncut gather's
                // run and a one-stick gather's are both the program's own; only the CUT below
                // slices it per stick). The [`StickRun`] carries the stick-aligned start and the
                // one-stick span together, so the leg states exactly what the IBR loads.
                run,
                o_name: o.name(),
                head_major,
            },
            sym_id_base,
            layout,
        )?]);
    }
    // ⛔⛔⛔⛔⛔ ONE GATHER OP'S INDEX IS ONE STICK, AND THIS DOOR WAS THE ONE GATHER IN THE CRATE
    // WITHOUT THAT GUARD. `gather_copy_opspec` refuses a longer run BY NAME with the card measurement
    // behind it ("dxp loads the IBR in a single stick transfer and takes each core's read offset inside
    // it modulo that stick, so entries past the first 32 WRAP and those cores gather another core's
    // pages with a clean bake" — rung 8's corruption). Everything in that sentence is a property of the
    // INDEX OPERAND, not of that builder's destination, so it was never `gather_copy_opspec`'s alone to
    // enforce: this door declares the same `Role::Index` through the same `attach_gather_index` and
    // reaches the same IBR.
    //
    // ⛔⛔⛔ AND THE CARD MEASURED IT THROUGH **THIS** DOOR, WHICH IS WHERE THE CUT BELOW COMES FROM.
    // `bake_py embedding_granite` (`Program::ScalarMul` over a `[256, 4096]` gathered row tile,
    // `page = 1`, so 256 entries against a 32-entry stick) baked clean, `dxp_standalone --bundle`
    // exited 0, the launch returned rc=0 in 0.265 s — and every output row `r` held table row
    // `ids[r mod 32]`:
    //
    //   rows[0:32]    within_2pct_strict = 1.000000   (131072 elements, max|err| 0.0625 = f16 ulp)
    //   rows[32:64]   within_2pct_strict = 0.006226
    //   rows[0:256]   within_2pct_strict = 0.130597   ~= 32/256, and 256/256 rows fit `r mod 32`
    //
    // The whole tensor had the RIGHT rms (11.9603 against the reference's 11.9844) because a
    // vocabulary's rows are independent `randn` draws, so a bare "does it look like an embedding"
    // check passes it — which is why the ceiling is a build-time fact and not a review note.
    //
    // ⭐ AND IT PROVED THE REST OF THE EMISSION CORRECT, which is what made the remedy a CUT and not a
    // rewrite: rows 0..31 were EXACT, so the cross-linked alloc pair, `skip_addr = page x out` = one
    // table row, the `Role::Index`/`SenUint32` declaration, the row-major residency a `mb`-pinned value
    // tensor implies, and the stick-major output were all already right. Only the ENTRY COUNT was
    // wrong.
    //
    // ⭐⭐⭐⭐⭐ THE CUT: ONE OP PER INDEX STICK, EACH WRITING A **ROW WINDOW** OF THE OUTPUT IN PLACE.
    //
    // Each leg declares `cap` entries (one IBR stick), reads index words `[k·cap, (k+1)·cap)` and
    // writes rows `[k·cap, (k+1)·cap)` of the output. MEASURED on the card, `emb256_rowwin_D`, all
    // eight row-blocks of `embedding_granite` against the fixture's own `ref_out.bin`:
    //
    //   rows[0:32] … rows[224:256]   within_2pct_strict = 1.000000   (131072 elements each)
    //   rows[0:256]                  within_2pct_strict = 1.000000   max|err| 0.0625 = the f16 ulp
    //
    // ⛔⛔⛔ AND TWO OTHER WAYS OF WRITING THAT WINDOW ARE ON THE RECORD AS WRONG, because each of them
    // is the obvious one and neither faults:
    //
    // (a) A **ROW-MAJOR SCRATCH** plus a `restickify` — the shape this refusal used to name. Forcing
    //     the leg's destination to address `Flat` gives a per-core start of `c·cols·2` (8192 B) while
    //     `stickDimOrder_` still says `out`, so the two disagree about every byte above one stick and
    //     the card raises `RAS::PCI::BusFence` 0xa35e ("PCIe bus master fence") at prepare — measured,
    //     deterministically, at segment 0 AND at segment 4 (so it is the addressing, not the segment)
    //     and with the restickify REMOVED (so it is not the relayout). The eight-leg bundle is
    //     otherwise byte-identical and runs.
    //
    // (b) Pinning the row count in **`maxDimSizes_`**. `[256, -1]` and `[256, 4096]` both run and both
    //     give `within_2pct_strict = 0.017013` over the whole tensor. Read off the card's own bytes:
    //     the row-window BASE is honoured (device run `32k` holds `ids[32k]`'s plane 0 for every k) and
    //     only the stick-PLANE stride is wrong — 32 rows instead of 256 — so leg `k`'s plane `s` lands
    //     at run `32·(k+s)` and run 512 held leg 7's plane 9, exactly as that predicts.
    //
    // ⭐⭐⭐ SO THE RESIDENCY EXTENT dxp RECONSTRUCTS FROM IS **`N_`**, AND THAT IS THE ONE THING A LEG
    // HAS TO SAY DIFFERENTLY. `maxDimSizes_ = -1` means "reconstruct the stick-blocked `device_size`
    // from `N_`/`layoutDimOrder_`/`stickSize_`", and it is `N_`'s `mb_` that the reconstruction reads —
    // so a leg's `N_` carries the OUTPUT TENSOR's row count while its WORK division
    // (`numWkSlicesPerDim_`, `dataStageParam_.ss_`/`el_`, the per-core starts 128 B apart, the index's
    // own 32-entry `N_`/`stickSize_`) carries the leg's own 32 rows. Everything else — including this
    // crate's own `StickLayout` for the view — is left describing the leg, which is what makes the
    // change one number rather than a second layout.
    let rows_per_op = cap.checked_mul(page.get()).ok_or_else(|| {
        SuperDscError(format!(
            "{op_name}: a {cap}-entry index stick at {} position(s) per entry overflows a row count",
            page.get()
        ))
    })?;
    let legs = rows.div_ceil(rows_per_op);
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(legs as usize);
    let mut first_row = 0u32;
    for k in 0..legs {
        // The LAST leg may be SHORT, and a short index run is the vendor's own shape — both fixtures
        // declare a sub-stick index extent (`test_gather_1core` at 3 entries), and
        // `GatherScratch::copies` cuts its final run the same way
        // (`(rows - first).min(ENTRIES_PER_OP)`).
        let leg_rows = (rows - first_row).min(rows_per_op);
        let leg_name = format!("{op_name}_g{k}");
        ops.push(one_gathered_leg(
            GatherLeg {
                op_name,
                leg_name: &leg_name,
                tile_op,
                op_func,
                leg_rows,
                node_rows: rows,
                first_row,
                cols,
                inputs,
                gathered_input,
                index_name,
                page,
                // ⭐ THE RUN'S OWN LEG `k` — one stick, `k` sticks in, both facts from the run
                // itself ([`StickRun::leg`]). Leg `k` of a run starting at the program's anchor
                // reads index words `[start + k·32, start + k·32 + leg_rows/page)`, and dropping
                // either term is every leg reading the run's first stick (the `EntryBase` type
                // exists because the run-half of that happened). MEASURED: leg `k`'s index
                // `allocate` starts 128 B = one 32-entry `SenUint32` stick further on than leg
                // `k-1`'s, for all eight — and the leg stride being one stick BY CONSTRUCTION is
                // what `of_sticks` states, where `plus_entries` used to check it (a `cap` change
                // that broke the stride would now fail to compile against `leg`, not bake).
                run: run.leg(k).map_err(SuperDscError)?,
                o_name: o.name(),
                head_major,
            },
            sym_id_base,
            layout,
        )?);
        first_row += leg_rows;
    }
    Ok(ops)
}

/// ONE LEG of a gathered pointwise op — the whole node when the index fits one stick, and one index
/// stick's worth of rows when it does not. See [`assemble_pointwise_broadcast_gather`], whose body this
/// is; it takes a struct because the leg carries fourteen quantities and the sibling assemblers'
/// `#[allow(clippy::too_many_arguments)]` is not a convention worth extending.
struct GatherLeg<'a> {
    /// The NODE's descriptor name — used in refusals, so a leg's diagnostic names the op the program
    /// stated and not only the synthetic leg spelling.
    op_name: &'a str,
    /// THIS leg's descriptor name (`{op_name}_g{k}`, or `op_name` itself for an uncut op).
    leg_name: &'a str,
    tile_op: &'a crate::ir::island::tile_op::TileOp,
    op_func: &'static str,
    /// The leg's OWN row count — its work, and `<= CopyDims::ENTRIES_PER_OP * page`.
    leg_rows: u32,
    /// The OUTPUT TENSOR's row count — its residency, and what `N_` must carry. Equal to `leg_rows`
    /// for an uncut op, in which case nothing below is rewritten.
    node_rows: u32,
    /// Which row of the output this leg starts at (`k * leg_rows`).
    first_row: u32,
    cols: u32,
    inputs: &'a [EwOperand<'a>],
    gathered_input: usize,
    index_name: &'a str,
    page: crate::superdsc_opspec::PageExtent,
    /// THIS leg's index run — one whole stick at the run's own leg-`k` start ([`StickRun::leg`]).
    /// A [`StickRun`] and not a bare base, so a leg cannot state more than one stick of index
    /// (the wrap the one-stick guard below refuses) or start between sticks (the IBR law).
    run: crate::superdsc_opspec::StickRun,
    o_name: &'a str,
    head_major: bool,
}

fn one_gathered_leg(
    leg: GatherLeg<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    let GatherLeg {
        op_name,
        leg_name,
        tile_op,
        op_func,
        leg_rows,
        node_rows,
        first_row,
        cols,
        inputs,
        gathered_input,
        index_name,
        page,
        run,
        o_name,
        head_major,
    } = leg;
    // ⭐ THE LEG'S OWN TILE IS THE NODE'S TILE WITH `mb` AT THE LEG'S ROW COUNT, and nothing else
    // moved. The tile is the program's statement of the axis NAMES the gather's pin resolves against
    // (which is why the caller passes a tile rather than `rows`/`cols`), so rebuilding one here would
    // be a second, unwitnessed declaration; overriding the one extent the cut changes is not.
    let leg_tile = crate::ir::island::tile_op::TileOp {
        kind: tile_op.kind,
        dims: tile_op
            .dims
            .iter()
            .map(|d| {
                if d.name == "mb" {
                    crate::superdsc_opspec::ItDim {
                        size: leg_rows,
                        ..*d
                    }
                } else {
                    *d
                }
            })
            .collect(),
        df: tile_op.df,
    };
    // ⭐⭐⭐ THE LEG'S CORE SPLIT IS A `GatherSplit` — MINTED FROM THE LEG'S OWN FACTS, before any
    // operand is built. Only the entry axis (`mb`) may carry a split: dxp derives `skip_addr`
    // from the per-core extents of the unpinned dims, so a split elsewhere shrinks what one
    // index entry covers and every index step lands a fraction of a row short, from a clean
    // bake (the prefill attention kernel's leg t6 measured it: 8 entries on 32 cores left
    // `distribute_cores` to split `out` 4 ways). The mint refuses the shapes the bytes do not
    // support — no `mb` to enumerate, or a partial entry — and otherwise runs on
    // `min(entries, cores)` cores over `mb` alone (ruling 10-20: the KV copy and the gathered
    // matmul are card-proven at fewer cores than the machine; the idle cores are never filled
    // by splitting another axis), so this leg cannot be built with any other division: the
    // walk check this replaces IS the constructor now, and a wrong division is unrepresentable
    // rather than inspected.
    let split = crate::work::GatherSplit::over_entries(
        &leg_tile.dims,
        crate::superdsc_opspec::MAX_CORES,
        page.get(),
    )
    .map_err(|e| SuperDscError(format!("{op_name}: leg `{leg_name}`: {e}")))?;
    let mut op = pointwise_broadcast_opspec_from_tile(
        &leg_tile,
        leg_rows,
        cols,
        op_func_from_str(op_func),
        inputs,
        o_name,
        // ⛔ ZERO, ALWAYS — the row-window base is applied BELOW and not here. See
        // `AnyTensorArg::set_offset_elems`: this builder reads `out_offset` to decide the operand
        // RANK, and a row window is not the whole-block shift its stick-major (rank-2) form requires,
        // so passing it here would drop the leg to the rank-3 flat presentation.
        crate::addr::OutCorner::ZERO,
        head_major,
        layout,
        crate::work::CoreDivision::Gathered(split),
    )
    .map_err(SuperDscError)?;
    // ⭐⭐⭐ THE ROW WINDOW: THE ONLY BYTE OF THE OUTPUT OPERAND THE CUT MOVES.
    //
    // Row `r` of a stick-major `[R, C]` tensor begins at element `r · lanes` (its column-0 lane of
    // stick plane 0) — `C` does not enter, because the plane stride is what carries the columns. So
    // leg `k`'s base is `first_row · lanes`, NOT `first_row · cols`.
    //
    // ⛔ IT GOES THROUGH `AnyTensorArg::set_offset_elems` (which already existed, for the fused
    // epilogue's per-head slice) RATHER THAN THROUGH THE BUILDER'S `out_offset`, and that is not a
    // style choice. `pointwise_broadcast_opspec_from_tile` reads `out_offset` to decide the operand
    // RANK: its stick-major (rank-2) presentation requires the offset to be "an exact multiple of
    // rows·cols", the whole-BLOCK shift. `first_row · lanes` is not one, so passing it there would
    // silently drop the leg to the rank-3 flat form — a different descriptor from the one the card
    // proved. Setting it here leaves the rank, the two sibling operands and the work division exactly
    // as the uncut op declares them.
    if first_row > 0 {
        let lanes = tile_op.df.elems_per_stick();
        let off = first_row.checked_mul(lanes).ok_or_else(|| {
            SuperDscError(format!(
                "{op_name}: leg `{leg_name}`'s row base {first_row} x {lanes} lanes overflows an \
                 element offset"
            ))
        })?;
        op.args
            .last_mut()
            .ok_or_else(|| {
                SuperDscError(format!(
                    "{op_name}: leg `{leg_name}` has no operands, so it has no destination to move"
                ))
            })?
            .set_offset_elems(off);
    }
    // `inputs` are args `0..inputs.len()` and the output is pushed last, so an input's index IS its arg
    // position — the same identity `emit_sdsc` relies on for `Tensor{i}-idx{i}`.
    let ia = op
        .attach_gather_index(
            crate::superdsc_opspec::GatherIndex {
                name: index_name.to_string(),
                // ⭐ THE OP'S OWN ROW AXIS. One entry per row is what a table lookup IS.
                entry_dim: crate::superdsc_opspec::KernelAxis::Batch,
                // ⭐ ONE POSITION — an entry is one row, so `skip_addr` is one row's `cols` elements.
                page,
                // ⛔ NO SECOND PAGED AXIS. `out` is the factor `skip_addr` is made of; pinning it too
                // would collapse an entry to a single element.
                per_position: None,
                first_entry: run.start(),
            },
            gathered_input,
        )
        .ok_or_else(|| {
            SuperDscError(format!(
                "{op_name}: the gather declaration was refused for input {gathered_input}. Either the \
                 op's work plan has no `mb` dim for the entries to enumerate, or `mb` does not divide \
                 into whole entries. Both are properties of the tile the program states."
            ))
        })?;
    // ⛔⛔⛔⛔⛔ ONE LEG'S INDEX IS ONE STICK, AND THIS IS THE GUARD THE CARD WROTE — kept on the op that
    // DECLARES the index, not only at the caller that counts the entries. The measurement is in
    // [`assemble_pointwise_broadcast_gather`]'s own note; this is the same law asserted where the
    // declaration is made, so a future leg-builder cannot reintroduce the wrap by handing this function
    // a longer run. It is unreachable from the cut by construction
    // (`leg_rows <= ENTRIES_PER_OP * page`), which is what makes it a proof and not a path.
    let entries = ia.page.entries_in(leg_rows).ok_or_else(|| {
        SuperDscError(format!(
            "{op_name}: the gather pages `{}` by {} position(s) and leg `{leg_name}`'s extent along it \
             is {leg_rows}, which does not divide into whole entries — a partial entry is an address.",
            ia.entry_dim.dim(),
            ia.page.get(),
        ))
    })?;
    let cap = crate::sdsc_abstract::CopyDims::ENTRIES_PER_OP;
    if entries > cap || run.sticks() > 1 {
        return Err(SuperDscError(format!(
            "{op_name}: leg `{leg_name}` declares {entries} index entries ({leg_rows} `{}` \
             position(s) / a {}-position page) over a run spanning {} stick(s), and ONE gather op's \
             index is ONE {cap}-entry stick. dxp fills the L3LU IBR with a single stick transfer and \
             each core indexes it at its own work-slice word, so entries past the first {cap} WRAP — \
             MEASURED on the card through this door: every output row `r` held the row `ids[r mod \
             {cap}]` names, rows 0..{} exact and the whole tensor at 0.130597 with the right rms. Cut \
             the node into one leg per index stick.",
            ia.entry_dim.dim(),
            ia.page.get(),
            run.sticks(),
            cap - 1,
        )));
    }
    // (The work-division inspection that lived here — "only the entry axis may carry a split",
    // with the `skip_addr` derivation and the leg-t6 measurement — is `GatherSplit`'s own
    // constructor now, called above: the leg's split is minted `mb`-only before any operand is
    // built, so there is no division left to inspect and no way to state one that would shrink
    // an entry. The law and its card measurements live in `crate::work::GatherSplit`'s doc.)
    let entry_axis = ia.entry_dim.dim();
    let folds = SdscFoldSet::new(op.iter.cores_used());
    let mut emitted = emit_sdsc_tiled(leg_name, &op, &folds, sym_id_base, layout)?;
    // 🅿️ NO STAMP HERE — the door's pool legs take their addressing fact from the CARRIER's
    // mint (`EmittedOp::paged_region`), stamped by the walk onto the ops it lowers from that
    // carrier; a stamp here would be a second writer for the same fact on the same ops. The
    // parked general-door note at the reach binding above states the class this leaves
    // unstamped (non-carrier programs — none exist in the tree's models today).
    // ⭐⭐⭐⭐⭐ `N_` CARRIES THE **TENSOR'S** ROW COUNT, NOT THE LEG'S — THE ONE NUMBER THE CUT CHANGES
    // ABOUT THE WIRE, AND THE CARD DECIDED IT.
    //
    // `maxDimSizes_ = -1` on a stick-blocked operand means "reconstruct the `device_size` from
    // `N_`/`layoutDimOrder_`/`stickSize_`", and the extent that reconstruction reads for the stick-GROUP
    // stride is `N_`'s `mb_`. A leg iterates 32 rows of a 256-row tensor, so the two differ — and left
    // at the leg's own 32 the card strides each stick plane by 32 rows: leg `k`'s plane `s` lands at
    // device run `32·(k+s)`, the eight windows pile up, and `within_2pct_strict` is 0.017013 over the
    // whole tensor with the right rms and no fault. Read off the card's own bytes, run 512 held leg 7's
    // plane 9, exactly as that stride predicts. Pinning the row count in `maxDimSizes_` instead
    // (`[256, -1]` and `[256, 4096]`) moved NOTHING — both give the same 0.017013.
    //
    // At the tensor's 256 all eight blocks are EXACT: `within_2pct_strict = 1.000000` per 32-row block
    // and over all 1048576 elements, `max|err|` 0.0625 = the fp16 ulp, against the fixture's own
    // `ref_out.bin` — `emb256_rowwin_D` on pod `nickm3-5999dffbdf-t2wcl`.
    //
    // ⛔ ONLY `N_`, AND THAT IS NOT AN OVERSIGHT. `numWkSlicesPerDim_`, `coreIdToWkSlice_` and
    // `dataStageParam_.ss_`/`el_` keep the LEG's 32 rows — they are its WORK, and the measurement above
    // was taken with exactly that split — as do the per-core start addresses (128 B apart, the
    // stick-major column-0 corner, which is the same number under either row count) and the index's own
    // 32-entry `N_`/`stickSize_`. Rewriting the plan instead would divide 256 by 32 cores and give each
    // core 8 rows, which is a different op.
    if node_rows != leg_rows
        && let Some(dsc) = emitted.op.as_mut()
    {
        for dsc_map in dsc.dscs_.iter_mut() {
            for d in dsc_map.values_mut() {
                set_iter_dim(&mut d.N_, entry_axis, node_rows);
            }
        }
    }
    Ok(emitted)
}

/// ⭐⭐⭐ AN INDEXED WRITE — `table[ids[row], :] = src[row, :]`, the store twin of
/// [`assemble_pointwise_broadcast_gather`] and the vendor's `dxp/test/test_scatter_1core` shape: an
/// `identity` copy over `[rows, cols]` whose OUTPUT is the table, the index declared on it through
/// [`OpSpec::attach_scatter_index`]. One entry per row ([`PageExtent::single_position`]), so
/// `skip_addr` is one table row; the work split is [`crate::work::GatherSplit`]'s, `mb` only, for
/// the same `skip_addr` reason the gather leg states.
///
/// ⛔ ONE INDEX STICK: a scatter of more rows than one `SenUint32` stick of entries is refused by
/// name — the IBR loads a single stick, the gather leg's measured `ids[r mod 32]` wrap.
pub struct IndexedWrite<'a> {
    pub op_name: &'a str,
    pub rows: u32,
    pub cols: u32,
    /// The rows written, `[rows, cols]` stick-major.
    pub src: &'a str,
    /// The table written through the index.
    pub table: &'a str,
    /// The table's own row count — its residency, carried as the output's `N_` row extent the way
    /// the gather leg's cut carries its tensor's.
    pub table_rows: u32,
    pub index_name: &'a str,
    pub run: crate::superdsc_opspec::StickRun,
}

pub fn assemble_indexed_write(
    w: IndexedWrite<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    let IndexedWrite {
        op_name,
        rows,
        cols,
        src,
        table,
        table_rows,
        index_name,
        run,
    } = w;
    let page = crate::superdsc_opspec::PageExtent::single_position();
    let cap = crate::sdsc_abstract::CopyDims::ENTRIES_PER_OP;
    if rows > cap || run.sticks() != 1 {
        return Err(SuperDscError(format!(
            "{op_name}: an indexed write of {rows} rows over a run of {} stick(s) — ONE op's index is \
             ONE {cap}-entry stick (the IBR loads a single stick transfer and entries past it wrap). \
             Write at most {cap} rows per indexed store.",
            run.sticks()
        )));
    }
    let tile_op =
        crate::emit::lower_ktir_to_superdsc::pointwise_tile_op(rows, cols, 1, 2, Df::Fp16);
    let split = crate::work::GatherSplit::over_entries(
        &tile_op.dims,
        crate::superdsc_opspec::MAX_CORES,
        page.get(),
    )
    .map_err(|e| SuperDscError(format!("{op_name}: {e}")))?;
    // The write's rows are the scatter's entries and its cols the source row's width — the wide
    // doors, named once at the one `rb` spend.
    let src_h = rb(
        src,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let inputs = [In::full(&src_h).ew()];
    let mut op = pointwise_broadcast_opspec_from_tile(
        &tile_op,
        rows,
        cols,
        op_func_from_str("identity"),
        &inputs,
        table,
        crate::addr::OutCorner::ZERO,
        false,
        layout,
        crate::work::CoreDivision::Gathered(split),
    )
    .map_err(SuperDscError)?;
    op.attach_scatter_index(crate::superdsc_opspec::GatherIndex {
        name: index_name.to_string(),
        entry_dim: crate::superdsc_opspec::KernelAxis::Batch,
        page,
        per_position: None,
        first_entry: run.start(),
    })
    .ok_or_else(|| {
        SuperDscError(format!(
            "{op_name}: the scatter declaration was refused — the op's plan has no `mb` dim for the \
             entries to enumerate"
        ))
    })?;
    // ⛔ A TABLE WIDER THAN ONE STICK IS REFUSED: its stick-plane stride is reconstructed from
    // `N_`, which this op states at its own `rows` — the gather leg's measured plane-stride trap.
    // One stick wide, there is one plane and no stride to get wrong.
    if cols > Fp16::ELEMS_PER_STICK {
        return Err(SuperDscError(format!(
            "{op_name}: an indexed write into a {cols}-wide table ({table_rows} rows) — wider than \
             one {}-element stick, so the table's stick-plane stride would be reconstructed from \
             this op's {rows}-row `N_` instead of the table's {table_rows}",
            Fp16::ELEMS_PER_STICK
        )));
    }
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
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

/// ⭐ THE NAME-KEYED LX-RESIDENCY CONSULT — the pointwise door's residency rewrite, run on the
/// freshly-built [`OpSpec`] before emission. A tensor's residency is a property of the TENSOR
/// (decided where its name is minted, `BundleLayout::synth_lx_op_local`), not of any one op view, so the
/// consult is by NAME against the layout — the SAME authority that resolves its address — rather
/// than a builder parameter every pointwise caller would have to thread and agree on.
///
/// Two rewrites, one law each:
///   * every operand whose name is LX-pinned gets [`Allocation::Lx`] (its `memOrg_` becomes
///     lx-only, its allocate becomes an `lx` component at the pin — no L3 movement is generated
///     for it);
///   * an op whose OUTPUT is LX-pinned gets a REPLICATED plan ([`WorkPlan::replicate`]): LX is
///     per-core private, so the only copy a consumer on core c can read is the one core c's own
///     producer pass wrote. A split producer would partition the tensor across different cores'
///     scratchpads, unreadable where the consumers' slices point.
///
/// A CORE-LOCAL pin ([`BundleLayout::synth_lx_core_local`]) is the exception to the second law: its
/// producer keeps its own division, and `emit_sdsc_tiled` holds every op touching the tensor to it.
///
/// A no-op (and byte-identical emission) for every bundle that declares no LX intermediates.
pub(crate) fn apply_lx_residency(op: &mut OpSpec, layout: Option<&BundleLayout>) {
    let Some(l) = layout else { return };
    mark_lx_residency(op, l);
    if op
        .args
        .last()
        .is_some_and(|o| l.lx_pin(o.arg_name()).is_some() && !l.lx_core_local(o.arg_name()))
    {
        op.iter = WorkPlan::replicate(op.iter.dims());
    }
}

/// The allocation half of the residency consult: every operand whose name is LX-pinned gets its LX
/// [`Allocation`]. `emit_sdsc_tiled` runs it for EVERY op, so a matmul or a reduction reading a core-local
/// tensor (a prompt chunk's PV and row sum over the probabilities) is marked like a pointwise op is —
/// and then held to the read law. Allocations only: a replicated producer's PLAN is set by the pointwise
/// door ([`apply_lx_residency`]), before its folds are taken from it.
fn mark_lx_residency(op: &mut OpSpec, l: &BundleLayout) {
    for a in op.args.iter_mut() {
        if l.lx_pin(a.arg_name()).is_some() {
            a.set_allocation(if l.lx_core_local(a.arg_name()) {
                Allocation::LxCoreLocal
            } else {
                Allocation::Lx
            });
        }
    }
}

/// ⛔⛔⛔ THE CORE-LOCAL READ LAW, in the ONE emission path so no builder can route around it. Each core
/// holds only the slice its own producer pass wrote, so every op touching a core-local LX tensor must:
///   * have DECLARED it core-local (its view's allocation) — an op that reaches the pinned name without
///     the residency consult would address LX bytes as HBM;
///   * run UNTILED — a trip-sliced op walks a different slice per trip, and the pin holds one;
///   * fit one core's slice OF THAT TENSOR inside the pin;
///   * hand every core the SAME slice OF THAT TENSOR as the first op on it did
///     ([`BundleLayout::record_lx_division`]).
///
/// ⭐ THE SLICE IS THE TENSOR's BYTES, NOT THE OP's AXES. Two ops of different kinds name a tensor's
/// axes differently and in different orders:
///   * a gather copy writes its destination as `[mb, out]`;
///   * the PV that contracts it reads the same bytes as its KERNEL `[in, out]`;
///   * the score reads a Kᵗ copy's bytes as a stick-major `[in, out]` kernel, whose stick axis is the
///     copy's row axis.
///
/// Ops also divide dims the tensor does not have, such as a matmul's rows. So the signature is the
/// tensor's own DEVICE layout: where each core's slice starts, as an HBM operand would address it, and
/// how many bytes it holds. Two ops agree exactly when core `c` holds the same bytes of the tensor under
/// both. For two ops of one shape (`esub` → `exp`) that is the op-level comparison it always was.
///
/// Any other op is a build `Err` naming the tensor — never fluent garbage on card.
fn core_local_reads_agree(
    op_name: &str,
    op: &OpSpec,
    views: &[ArgView<'_>],
    layout: Option<&BundleLayout>,
) -> Result<(), SuperDscError> {
    let Some(l) = layout else { return Ok(()) };
    for v in views.iter().filter(|v| l.lx_core_local(v.name)) {
        let name = v.name;
        if !v.allocation.is_core_local() {
            return Err(SuperDscError(format!(
                "op '{op_name}' addresses core-local LX tensor '{name}' without declaring it core-local \
                 — it did not pass the residency consult (`apply_lx_residency`), so it would read LX \
                 bytes as an HBM address"
            )));
        }
        if op.time_tile.is_some() {
            return Err(SuperDscError(format!(
                "op '{op_name}' is time-tiled over core-local LX tensor '{name}' — a trip-sliced op \
                 walks a different slice per trip, and the pin holds one"
            )));
        }
        let plan = &op.iter;
        let slice_bytes = v
            .layout
            .iter()
            .map(|d| u64::from(plan.per_core_extent(d)))
            .product::<u64>()
            * u64::from(v.df.word_length());
        let (_, pin_bytes) = l.lx_pin(name).unwrap_or_default();
        if slice_bytes > pin_bytes {
            return Err(SuperDscError(format!(
                "op '{op_name}': one core's slice of core-local LX tensor '{name}' is {slice_bytes} B, \
                 past its {pin_bytes}-byte pin"
            )));
        }
        // Where each core's slice starts in the tensor's device layout: the very per-core START an HBM
        // operand of this view would carry (`per_core_addr` over `view_stick_layout`, at base 0).
        let as_hbm = ArgView {
            is_input: v.is_input,
            name: v.name,
            role: v.role,
            scale: v.scale,
            device_dims: v.device_dims,
            layout: v.layout,
            stick: v.stick,
            allocation: Allocation::Hbm,
            offset_elems: v.offset_elems,
            df: v.df,
            row_blocked: v.row_blocked,
            device_extent: v.device_extent,
            dxp_innermost_first: v.dxp_innermost_first,
        };
        let dims: Vec<&'static str> = plan.dims().iter().map(|d| d.name).collect();
        let corners = per_core_addr(
            0,
            &as_hbm,
            &view_stick_layout(&as_hbm, plan),
            plan,
            &crate::work::core_to_wk_slice(plan, &dims),
            0,
            plan.cores_used().get(),
        )?;
        let signature = format!("each core's {slice_bytes} B slice at {corners:?}");
        l.record_lx_division(name, op_name, signature)
            .map_err(SuperDscError)?;
    }
    Ok(())
}

/// [`assemble_pointwise_broadcast`] that writes the output to a COLUMN-BLOCK OFFSET
/// (`out_offset` elements into the output tensor) — see [`pointwise_chunk_out_offset`].
///
/// ⭐ THE OP'S WHOLE FACT SET, ON ONE CARRIER — [`PointwiseDivided`] minus the `division` field,
/// because this door's contract is that the LAW states the division; the body below fills it in
/// when it delegates to [`assemble_pointwise_broadcast_divided`]. This deletes the nine-parameter
/// positional run every caller spelled (the arrangement where a name, an extent or an offset
/// transposes silently). The sink pair (`sym_id_base`/`layout`) stays in the signature BESIDE the
/// carrier, as it does at every door of this family.
pub struct PointwiseFacts<'a, O: KindTag> {
    pub op_name: &'a str,
    pub op_func: &'static str,
    /// TYPED row/width slots. Every caller states WHICH row quantity (`RowCount`'s doors: mask rows,
    /// chunk rows, padded rows, pad remainder, token rows) and WHICH width (`BlockCols`'s doors: one
    /// stick, the new block's `mq_pad`, the head dim, feature cols) fills the slot — the quantities
    /// are numerically equal across whole rung families (64 == 64 at mq<=2, hd == stick at hd=64),
    /// so a bare `u32` could carry any of them without anything noticing.
    pub rows: crate::sdsc_abstract::RowCount,
    pub cols: crate::sdsc_abstract::BlockCols,
    pub inputs: &'a [EwOperand<'a>],
    pub o: &'a Stk<O>,
    /// TYPED, like the inputs' `In::sliced`. A write offset is a DEVICE address, so it must come from
    /// a nest that knows the tensor's extents; the only constructor is `View::dev`. When this was a
    /// bare `u32` a caller could pass a flat column index, and one did -- the split MLP intermediate
    /// wrote its second column block at element 8192 instead of `(8192/64)*(rows*64)`, correct at one
    /// row and wrong at every prefill. The reads were already typed and were already right.
    /// A [`crate::addr::ColWindow`] (a whole-stick column window, #622) converts too, and carries the fact
    /// that the window is a stick-major tensor of its own.
    pub out_offset: crate::addr::OutCorner,
}

pub fn assemble_pointwise_broadcast_off<O: KindTag>(
    p: PointwiseFacts<'_, O>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // ⭐ THE OP'S FACTS, DESTRUCTURED ONCE AT THE DOOR — the same names the parameter list
    // spelled, so the delegation below states the same op on the same values. The LAW division
    // this door has always implied is filled in here, at the one place that owns it.
    let PointwiseFacts {
        op_name,
        op_func,
        rows,
        cols,
        inputs,
        o,
        out_offset,
    } = p;
    assemble_pointwise_broadcast_divided(
        PointwiseDivided {
            op_name,
            op_func,
            rows,
            cols,
            inputs,
            o,
            out_offset,
            division: crate::work::EwDivision::Law,
        },
        sym_id_base,
        layout,
    )
}

/// The parameter list of [`assemble_pointwise_broadcast_divided`], carried as a struct — the
/// [`PointwiseGather`] / `RopeAt` shape, which adds no `#[allow(clippy::too_many_arguments)]`. Every field
/// but `division` is [`assemble_pointwise_broadcast_off`]'s, with the same contract.
pub struct PointwiseDivided<'a, O: KindTag> {
    pub op_name: &'a str,
    pub op_func: &'static str,
    pub rows: crate::sdsc_abstract::RowCount,
    pub cols: crate::sdsc_abstract::BlockCols,
    pub inputs: &'a [EwOperand<'a>],
    pub o: &'a Stk<O>,
    pub out_offset: crate::addr::OutCorner,
    /// The work division the op is assembled under. [`crate::work::EwDivision::Law`] is what the shipped
    /// door ([`assemble_pointwise_broadcast_off`]) states.
    pub division: crate::work::EwDivision,
}

/// [`assemble_pointwise_broadcast_off`] under a STATED work division instead of the law's. Two callers
/// take this door:
///   * a decode batch chunk's `esub` and `exp`, whose shared core-local `cesub` is laid out sticks first
///     (`attn`'s `ChunkBufs::sm_division_of` says why). The law alone would divide them rows first,
///     because every op touching a core-local pin keeps the division its pin was sized under;
///   * torch-spyre's descriptor parity (`element_torchspyre_parity.rs`). That claim is about the bytes
///     torch-spyre emits, and torch-spyre divides every elementwise op rows-first, so it is stated under
///     [`crate::work::EwDivision::RowsFirst`]. The law, measured faster on card, ships a different division
///     for the same shape.
pub fn assemble_pointwise_broadcast_divided<O: KindTag>(
    p: PointwiseDivided<'_, O>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let PointwiseDivided {
        op_name,
        op_func,
        rows,
        cols,
        inputs,
        o,
        out_offset,
        division,
    } = p;
    let rows = rows.get();
    let cols = cols.get();
    // ⭐ THE OUTPUT MINT — the output window `rows × stick-padded cols` at the corner, checked
    // against the output handle's declared shape and placement before any op is built (the
    // write-side twin of the input mint in `pointwise_broadcast_opspec_from_tile`, which runs
    // inside the funnel call below).
    mint_ew_output(op_name, o, out_offset.elems(), rows, cols, Df::Fp16, layout)
        .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast {op_name}: {e}"));
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
    let mut op = pointwise_broadcast_opspec_df(
        op_func_from_str(op_func),
        rows,
        cols,
        inputs,
        o_name,
        out_offset,
        head_major,
        Df::Fp16,
        layout,
        division,
    )
    .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast {op_name}: {e}"));
    apply_lx_residency(&mut op, layout);
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
        crate::addr::OutCorner::ZERO,
        true,
        df,
        layout,
        crate::work::EwDivision::Law,
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
    // ⭐ TRUE EXTENTS, not `rbo`'s `1×64` annotation: x and the output are the probe's own
    // `[rows, cols]` tensors, and v is the `[rows, one-stick]` column the out-broadcast read sweeps
    // (ONE STICK PER ROW — the RedStick law `EwOperand::swept_extents` states). The probe creates
    // its tensors only in the descriptor (`layout: None`), so the handle IS the statement; the mint
    // reads the dims as the bound — the typed pair crosses whole, no bare twin.
    // "multiply" is the INTERNAL op_func_from_str key (→ OpFunc::Multiply, whose
    // wire name() is "mul"); passing "mul" would miss the match and default to Add.
    assemble_pointwise_broadcast(
        op_name,
        "multiply",
        rows,
        cols,
        &[
            In::full(&rb(x_name, rows, cols)).ew(),
            EwOperand {
                mb_broadcast: false,
                out_broadcast: true,
                ..In::full(&rb(v_name, rows, BlockCols::of_one_stick(Lanes::FP16))).ew()
            },
        ],
        &rb(o_name, rows, cols),
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
        indirect: None,
        coordinate_mask: None,
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
    /// Where in its buffer this operand starts, in device elements (`ArgView::offset_elems`, 0 for a
    /// whole tensor). What [`crate::emit::lower_ktir_to_superdsc::column_windows_are_honoured`] holds
    /// a column-blocked piece's operands to.
    pub offset_elems: u32,
    /// ⭐⭐⭐ THE DOOR'S OWN ANSWER, CARRIED — `resolve_seg_base`'s resolution of THIS
    /// operand, stamped here by [`EmittedOp::new`] from the facts the door returned at
    /// emission time. There is ONE resolution in the codebase; a consumer that resolves
    /// again keeps its own copy of the door's address logic and the copy DRIFTS (the
    /// record-53 word-source drift, then the record-54 arm-order drift — both were the
    /// walk's private `resolve()` disagreeing with `resolve_seg_base` on the same
    /// operand). Not serialised into the op.
    ///
    /// The segment the door resolved this operand into (`segment_base`'s index), NOT
    /// derivable from `base_bytes` — the walk keys its Written table by (segment, bank)
    /// so two different banks of the same role never alias, and a consumer must not
    /// invert the base address to get it back.
    pub segment: usize,
    /// The bank WITHIN `segment` the door resolved — part of the Written-table key.
    pub bank: u32,
    /// The role the door resolved — the read-before-write walk's honest-set membership
    /// (`Weight | Activation | Kv`, nothing else).
    pub role: crate::placement::SegRole,
    /// The door's INTRA-segment base in bytes: the placement/synth/segment-fallback
    /// offset PLUS the `offset_elems · word` slice. The walk's byte interval base is
    /// EXACTLY this (plus per-trip affine shift), so the walk never re-derives the
    /// slice arithmetic either.
    pub base_bytes: u64,
    /// The door's LX verdict — true means core-local (a pin in core-local space, no
    /// HBM bytes), and the walk SKIPS the operand rather than judge core-local bytes.
    pub lx: bool,
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

/// ⭐ THE EMITTER CONTEXT — the op sink, the sym-id counter and the placement authority as ONE
/// value, because they are ONE fact: a single emission whose program order is the call order into
/// the ONE sink, whose sym ids are minted only through the ONE counter, and whose placement
/// questions have exactly one authority. Minted once at the door that owns the sink —
/// `assemble_attn` for the attention path, `lower_function` for the whole-function walk — and
/// threaded as ONE `&mut EmitCtx` through every emitter that pushes into it, never split into
/// per-field borrows at a call site, which would let two spellings of "the" emission disagree
/// about order, ids or placement. The one-op assemblers (`matmul`, `reduce`, `pointwise`) keep
/// their `(sym_id_base, layout)` pair WITHOUT the sink — they RETURN their op rather than push it,
/// the one-op-descriptor form — and a sink-threader hands them `ctx.sym_id_base` / `ctx.layout`
/// as plain call arguments.
pub(crate) struct EmitCtx<'a> {
    /// The one sink: program order is call order into this Vec.
    pub(crate) ops: &'a mut Vec<EmittedOp>,
    /// The one counter: every sym id is minted through this base.
    pub(crate) sym_id_base: &'a mut i64,
    /// The one placement authority — absent exactly where the pair's was.
    pub(crate) layout: Option<&'a BundleLayout>,
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
    /// ⭐⭐⭐ THIS OP'S BATCH AXIS WALKS A **GATHERED** SCRATCH, not the paged pool.
    ///
    /// Travels beside [`Self::kv_batched_requests`] because it answers the OTHER half of `collapsed`:
    /// with the axis over a contiguous destination the compiler placed, each row's PAGE is resolved by
    /// the index the host stages, so the pool needs no single stride and `LaunchPages::Affine` — which is
    /// false for any free-list allocation — stops being a precondition. Without it a request axis still
    /// needs that proof, so the runtime must be able to tell the two situations apart.
    pub kv_gathered: bool,
    /// ⭐ THE STAGING CHANNEL — this op's gathered table is the FILL-STAGED REGION FORM
    /// (`fill_attn_step_inputs`), not the composer's page-list table.
    ///
    /// `kv_gathered` alone carries TWO consequences that can diverge: the addressing half (the
    /// group's KV shift must be ZERO — absolute entries plus a page base compose two bases) and
    /// the staging half (the host must stage an index table). A region-form body answers YES to
    /// the first and NO to the second's page-list spelling: its table is already staged by the
    /// fill, and the composer must not drive it as a chunk body (a swept extent that is not a
    /// whole multiple of the page would refuse). BOTH channels' entries are ABSOLUTE, so the
    /// zero shift is correct for both — this bit only discriminates WHO stages the table.
    /// Written ONLY by the walk's chain-of-custody stamps (a fold carrier's or a region
    /// carrier's), never at a mint.
    pub kv_region_form: bool,
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
    /// ⭐⭐⭐⭐⭐ THE PAGED FOLD THIS OP'S PROGRAM WAS MINTED AS — the attribute the fold's chain
    /// of custody runs on. `None` on every op but a spliced fold family's BODY carrier: the
    /// splice STATES the family's laws here through [`Self::paged_fold`] (the ONE door), the
    /// walk STAMPS them onto every op it lowers from that carrier's program, and the bake's
    /// fallback proof CHECKS the stamped groups. The seed and finalize carriers of the same
    /// family stay `None` — they launch once per step, not once per page.
    pub kv_paged_fold: Option<crate::sdsc_abstract::PagedFold>,
    /// ⭐⭐⭐⭐⭐ THE PAGED REGION THIS OP'S PROGRAM GATHERS ITS KV THROUGH — the attribute the
    /// ONE-SHOT attention chain of custody runs on. `None` on every op but a spliced
    /// attention body: the splice STATES the region here through [`Self::paged_region`] (the
    /// ONE door, UNCONDITIONALLY — every one-shot attention program gathers the pool through
    /// the absolute-entry tables, whether or not its ladder stated a fold region), the walk
    /// STAMPS the addressing fact onto every op it lowers from that carrier's program
    /// (`kv_gathered`, plus the region-form channel bit the staging side keys on), and the
    /// entries the host stages are ABSOLUTE stick-block numbers — the op takes the ZERO KV
    /// shift, never the page base on top. The seed and finalize carriers of a fold family
    /// stay `None`; the fold body itself states its region through [`Self::kv_paged_fold`]
    /// instead, whose stamped ops carry the same region form. The pool tids ride with the
    /// region so the stamp site can check the layout's gather reach — see
    /// [`crate::sdsc_abstract::PagedRegion`].
    pub kv_paged_region: Option<crate::sdsc_abstract::PagedRegion>,
    /// ⭐ WHICH PAGE OF A MULTI-PAGE WINDOW THIS OP ADDRESSES — a
    /// [`PagePart`](crate::page_part::PagePart), never a number.
    ///
    /// One launch resolves ONE page per op from the request's block table, at the page holding the
    /// op's cursor. A window of `PREFILL_WINDOW_PAGES` pages therefore cannot be one write: the
    /// emitter cuts each `(kv head, slab)` copy into parts of `PAGE_SLOTS` rows, and the op carries
    /// THE CUT'S OWN part (stamped from the [`KvWritePart`](crate::sdsc_abstract::KvWritePart)
    /// `write_parts` returned — the only minter), so the runtime resolves ITS page at
    /// `cursor + part` (`fold_plan::op_slot_pos`). A part whose page the request does not hold (a
    /// short last chunk's padding) is SKIPPED, never aimed at part 0's page. Ops of different parts
    /// never share a group: one group is one page shift, and the fusion walk compares THIS TYPE —
    /// what the part carries is the argument's type, not this comment.
    pub kv_part: crate::page_part::PagePart,
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

    /// Expand this op into its CONCRETE per-trip [`SdscOp`]s — the ONE implementation, living with
    /// the `time`/`affine_strides` fields it reads. A time=1 op yields `[dsc().clone()]`. A time=N
    /// op yields N copies: in trip `t`, every tensor with an `affine_strides[ti]["out"]` entry has
    /// its AllocNode start addresses bumped by `t · stride_bytes`; an operand with an EMPTY stride
    /// map (an out-broadcast const, an LX tensor, a matmul's A) keeps its base — which is exactly
    /// the address-stability a broadcast operand in a time-tiled op requires (see
    /// [`rewrite_op_for_time_tile`]'s tiled predicate). The spyre crate's free `concrete_trips` is
    /// a thin delegate to this, so the bundle writer and this crate's tests cannot drift apart.
    pub fn concrete_trips(&self) -> Vec<SdscOp> {
        if self.time <= 1 {
            return vec![self.dsc().clone()];
        }
        (0..self.time)
            .map(|t| {
                let mut op = self.dsc().clone();
                for dsc_map in op.dscs_.iter_mut() {
                    for dsc in dsc_map.values_mut() {
                        for node in dsc.scheduleTree_.iter_mut() {
                            let ti = node.ldsIdx_ as usize;
                            let Some(stride) =
                                self.affine_strides.get(ti).and_then(|m| m.get("out"))
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
            kv_gathered: false,
            kv_region_form: false,
            kv_fold_rows: Default::default(),
            kv_page_slots: 0,
            kv_request: 0,
            kv_paged_fold: None,
            kv_paged_region: None,
            kv_part: crate::page_part::PagePart::at_cursor(),
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

    /// ⭐⭐⭐⭐⭐ THE ONE DOOR FOR THE PAGED-FOLD ATTRIBUTE — a fold family's BODY carrier, with
    /// the family's launch-ABI laws stated by the layer that minted it (the splice).
    ///
    /// ⛔ NOTHING ELSE MAY WRITE [`Self::kv_paged_fold`]. The attribute is the single carrier
    /// between the splice's statement and the walk's stamp (`ktir_groups_via_superdsc`, which
    /// copies these facts onto every op it lowers from this carrier's program) and the proof the
    /// bake runs over the stamped groups — so a second writer is a second source of the fold
    /// facts, which is the desync the field exists to prevent. The seed and finalize carriers
    /// of the same family are `bare` ops: they launch once per step, not once per page, and
    /// carrying the attribute on them would stamp a fold onto programs that must not re-launch.
    pub fn paged_fold(op_name: String, fold: crate::sdsc_abstract::PagedFold) -> EmittedOp {
        let mut e = EmittedOp::bare(op_name);
        e.kv_paged_fold = Some(fold);
        e
    }

    /// ⭐⭐⭐⭐⭐ THE ONE DOOR FOR THE PAGED-REGION ATTRIBUTE — a one-shot attention body, with
    /// the region stated by the layer that minted it (the splice, at the shared tail of both
    /// row arms) and the pool's own tids riding with it for the stamp site's premise check.
    ///
    /// ⛔ NOTHING ELSE MAY WRITE [`Self::kv_paged_region`]. The attribute is the single
    /// carrier between the splice's statement and the walk's stamp (which copies the
    /// addressing fact onto every op it lowers from this carrier's program), so a second
    /// writer is a second source of the region fact, which is the desync the field exists
    /// to prevent. `bare` stays the mint for every op that is not a one-shot attention
    /// carrier: the region is now stated UNCONDITIONALLY for the arm (a prefix-free rung
    /// gathers the pool through the same absolute-entry tables), so an attention body with
    /// NO attribute is a walk/splice desync, not an unpaged form.
    pub fn paged_region(op_name: String, region: crate::sdsc_abstract::PagedRegion) -> EmittedOp {
        let mut e = EmittedOp::bare(op_name);
        e.kv_paged_region = Some(region);
        e
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
            kv_gathered: self.kv_gathered,
            kv_region_form: self.kv_region_form,
            kv_fold_rows: self.kv_fold_rows,
            kv_page_slots: self.kv_page_slots,
            kv_request: self.kv_request,
            kv_paged_fold: self.kv_paged_fold,
            kv_paged_region: self.kv_paged_region,
            kv_part: self.kv_part,
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
    ///
    /// It also ASSERTS **FactsMatchBindings**: `facts.len() == spec_args.len()` —
    /// the door resolved every operand, so a parallel-array mismatch is a
    /// build-time panic (the same internal-consistency class as
    /// TiledHasStrideAndNode), never an unchecked parallel-index assumption that
    /// stamps one operand's resolution onto another's binding.
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
        // The door's per-operand resolutions (`resolve_seg_base`, collected by `emit_sdsc`) —
        // stamped onto binding[i] so there is ONE resolution in the codebase, never a second copy.
        facts: &[SegFacts],
    ) -> EmittedOp {
        assert!(
            facts.len() == spec_args.len(),
            "EmittedOp {op_name}: the door resolved {} operands but the op binds {} \
             (FactsMatchBindings witness)",
            facts.len(),
            spec_args.len()
        );
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
            .enumerate()
            .map(|(i, a)| {
                let v = a.view();
                let f = facts[i];
                ArgBinding {
                    buffer: v.name.to_string(),
                    is_input: v.is_input,
                    footprint_bytes: materialized_bytes(&v, iter),
                    touched: touched_blocks(&v, iter),
                    offset_elems: v.offset_elems,
                    segment: f.segment,
                    bank: f.bank,
                    role: f.role,
                    base_bytes: f.intra_base,
                    lx: f.lx,
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
            kv_gathered: false,
            kv_region_form: false,
            kv_fold_rows: crate::sdsc_abstract::FoldRowRegime::WholeBatch,
            kv_page_slots: 0,
            kv_request: 0,
            kv_paged_fold: None,
            kv_paged_region: None,
            kv_part: crate::page_part::PagePart::at_cursor(),
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
    // Every op kind passes the residency consult here, not only the pointwise door — a byte-identical
    // no-op for an op that names no LX-pinned tensor.
    let marked;
    let op = match layout {
        Some(l) if op.args.iter().any(|a| l.lx_pin(a.arg_name()).is_some()) => {
            let mut o = op.clone();
            mark_lx_residency(&mut o, l);
            marked = o;
            &marked
        }
        _ => op,
    };
    let time = op.time();
    let (inner, facts) = emit_sdsc(op_name, op, folds, layout)?;
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
            &facts,
        ));
    }
    rewrite_op_for_time_tile(op_name, op, folds, inner, time, sym_id_base, &facts)
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
    facts: &[SegFacts],
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
    // Which tensors are tiled: HBM tensors that RANGE over `tiled_dim` — their layout
    // contains it AND its scale is `Active` there. Two operands are thereby exempt:
    //   • an op whose layout lacks the dim (a matmul's INPUT/A does not carry `out`);
    //   • ⭐ an operand COLLAPSED ALONG THE TILED DIM — its scale there is RedStick or
    //     RedNonStick (`out_broadcast` → RedStick on `out`: the tanhsoftcap cap const,
    //     the rmsnorm eps; a reduce's RedStick accum on its reduced dim). Its coordInfo
    //     is ALREADY the collapsed one-stick/one-row fold `build_coordinates` wrote, and
    //     its address must NOT advance per trip: every trip's out-window reads the SAME
    //     broadcast value. The old predicate (`layout.contains(tiled_dim)` alone) handed
    //     such an operand a per-trip stride AND re-derived its collapsed coordinate as a
    //     FULL per-time fold (`divide_out_coordinate`) — silently-wrong on both counts,
    //     which the pointwise_broadcast builder refused outright. The exemption IS the
    //     fix: the operand is left byte-identical to what `emit_sdsc` wrote, and
    //     `concrete_trips` keeps its base (an empty stride map is its no-bump path).
    //     The collapse must be along the TILED dim: an operand collapsed along a
    //     DIFFERENT dim still RANGES this one and MUST advance per trip. `In::mb` (the
    //     rmsnorm gamma, a `[1,cols]` row vector) is RedNonStick on `mb` but ACTIVE on
    //     `out` — in an op tiled over `out` it takes the per-trip stride below and each
    //     trip reads its own out-window of that row (pinned: the mb operand's trip-start
    //     advances; a strideless ranging operand is the 2b guard's build Err).
    for (ti, v) in views.iter().enumerate() {
        // ── 2a. THE SEEDED REDUCE'S ACCUM — THE PARTIALS LAW (#603) ─────────────────
        // A seeded reduce that time-tiles writes each trip's partial into its OWN stick
        // of a multi-stick accum buffer, NEVER the same one-stick accum twice: the RedStick
        // scale exempts the accum from the tiled predicate below (its collapsed one-stick
        // coordinate is correct as written), so without a stride of its own both trips
        // would write the SAME HBM byte — the silently-wrong alias the bake's
        // `tiled_trips_alias` guard refuses, and the exact collision gemma-4's 262144-wide
        // max32 produced under the old over-charge. No generic builder emits that form
        // any more — the single-trip row is the only seeded form — so a time-tiled seeded
        // reduce whose accum is a plain single stick is a build Err naming the law: the
        // alternative is the alias, caught here at the emission rather than at the bake.
        if op.is_reduction
            && !v.allocation.is_lx()
            && v.layout
                .iter()
                .position(|&d| d == tiled_dim)
                .is_some_and(|si| matches!(v.scale.get(si), Some(Scale::RedStick)))
        {
            let si = v.layout.iter().position(|&d| d == tiled_dim).unwrap();
            let one_stick = v.df.elems_per_stick() as i64;
            let want_sticks = time as i64 * one_stick;
            let phys = v.device_extent.get(si).copied().flatten();
            if phys.map(|p| p as i64) != Some(want_sticks) {
                return Err(SuperDscError(format!(
                    "{op_name}: a SEEDED reduce time-tiled {time}× whose accum '{}' is not in \
                     the partials form — its device extent along '{tiled_dim}' is {phys:?}, and \
                     the law demands EXACTLY one stick per trip ({want_sticks} elems at {}). \
                     Without it, every trip writes the SAME one-stick accum (the #50 \
                     `tiled_trips_alias` alias). The single-trip row is the only seeded form \
                     this emitter has: let the row fit a single trip (the true accum charge \
                     bills the accum at one stick per row), or split the reduce yourself — \
                     per-slice single-shot reduces plus a combine, the fp8 qspt path's own \
                     pattern (emit/ktir_matmul_fp8.rs).",
                    v.name,
                    v.df.elems_per_stick(),
                )));
            }
            // ONE-STICK per-trip advance, and the accum's coordinate stays the collapsed
            // one-stick fold `build_coordinates` wrote — divide NOTHING (the code below is
            // the tiled-DATA path; the accum is not a tiled data operand).
            let mut sm = BTreeMap::new();
            sm.insert(tiled_dim, one_stick * v.df.word_length() as i64);
            affine_strides[ti] = sm;
            continue;
        }
        let tiled = !v.allocation.is_lx()
            && v.layout
                .iter()
                .position(|&d| d == tiled_dim)
                .is_some_and(|si| v.scale.get(si).is_none_or(|s| matches!(s, Scale::Active)));
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

    // ── 2b. INPUT-SIDE NO-ADVANCE GUARD ────────────────────────────────────────
    // The bake's `tiled_trips_alias` guard covers the OUTPUT side only (per-trip
    // OUTPUT allocations must be pairwise disjoint). The INPUT side had no guard: an
    // HBM operand that RANGES the tiled dim — Active scale there, more than one
    // trip's worth of physical extent — but carries NO per-trip advance would
    // silently re-read trip 0's window in every later trip: wrong output, no
    // refusal anywhere (the #53 class). Every such operand MUST carry a nonzero
    // stride along the tiled dim; the loop above gives each one exactly that, so
    // this guard is the belt-and-suspenders that makes a strideless ranging
    // operand a build Err instead of a silent re-read.
    for (ti, v) in views.iter().enumerate() {
        if v.allocation.is_lx() {
            continue; // on-chip: per-core private, outside the HBM stride path
        }
        let Some(si) = v.layout.iter().position(|&d| d == tiled_dim) else {
            continue; // does not range the tiled dim (a matmul's A): re-read is correct
        };
        if !v.scale.get(si).is_none_or(|s| matches!(s, Scale::Active)) {
            continue; // collapsed along the tiled dim: the broadcast exemption above
        }
        let phys = v
            .device_extent
            .get(si)
            .copied()
            .flatten()
            .map(|p| p as i64)
            .unwrap_or_else(|| op.iter.extent(tiled_dim) as i64);
        if phys > out_per_time && affine_strides[ti].get(tiled_dim).copied().unwrap_or(0) == 0 {
            return Err(SuperDscError(format!(
                "{op_name}: operand {ti} ('{}') RANGES the tiled dim '{tiled_dim}' ({phys} \
                 elems > the {out_per_time}-elem per-trip window) but carries NO per-trip \
                 advance — every trip after the first would re-read trip 0's window, \
                 silently wrong. An operand that must NOT advance is collapsed along the \
                 tiled dim (RedStick/RedNonStick there), which the tiled predicate exempts; \
                 this one is not.",
                v.name
            )));
        }
    }

    Ok(EmittedOp::new(
        op_name.to_string(),
        inner,
        &op.args,
        &op.iter,
        time,
        affine_strides,
        facts,
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

/// ⭐⭐⭐⭐⭐ THE RESOLVED PLACEMENT FACTS FOR ONE OPERAND — `resolve_seg_base`'s
/// answer, carried whole. Every field was already in hand per arm; the struct
/// exists so the answer can TRAVEL: `emit_sdsc` collects one per operand and
/// `EmittedOp::new` stamps it onto the matching [`ArgBinding`], which is how the
/// read-before-write walk reads placement facts OFF THE BINDING instead of
/// re-deriving them — records 53 and 54 were the same defect twice (the walk's
/// own copy of this resolution drifting from the door's at a different spot
/// each time), and the cure is ONE resolution, stated here, consumed
/// everywhere. `seg_base`/`intra_base` are the address pair the on-card walk
/// always consumed; `role`/`segment`/`bank` are the facts the coverage walk
/// keys on; `lx` marks the pinned-LX arm (per-core scratchpad, out of the HBM
/// walk's domain).
#[derive(Clone, Copy)]
pub struct SegFacts {
    /// The operand's HBM segment base (`SEGMENT_OFFSETS[segment]`); 0 for an
    /// LX-resident name, whose address is its pin.
    pub seg_base: u64,
    /// The intra-segment byte base — the placement/synth offset PLUS the
    /// `offset_elems · df.word_length()` slice, exactly the number the address
    /// math always consumed.
    pub intra_base: u64,
    /// The buffer's role — the honest-set key for the coverage walk.
    pub role: crate::placement::SegRole,
    /// The segment INDEX (never the base address — the walk keys its written
    /// table on the index, which cannot alias the way a number can).
    pub segment: usize,
    /// The weight bank, for the walk's `(segment, bank)` table key.
    pub bank: u32,
    /// TRUE when the name resolved through the pinned-LX arm: per-core
    /// scratchpad, invisible to (and skipped by) the HBM coverage walk.
    pub lx: bool,
}

/// The role a segment index belongs to — the inverse of `SegRole::segment()`,
/// stated for the NO-LAYOUT fallback only (a real bundle's facts come from the
/// placement row or the synth map, never from here). Segments 5 and 6 are
/// intermediate COLORS (`inter_segs`), so they answer Intermediate.
fn role_of_segment(seg: usize) -> crate::placement::SegRole {
    match seg {
        1 => crate::placement::SegRole::Weight,
        2 => crate::placement::SegRole::Kv,
        3 => crate::placement::SegRole::Activation,
        4 => crate::placement::SegRole::Logits,
        _ => crate::placement::SegRole::Intermediate,
    }
}

/// Resolve one arg's HBM segment base + intra-segment byte offset, and the
/// placement facts that travel with them. With a
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
    lane_read: bool,
) -> Result<SegFacts, SuperDscError> {
    let slice = offset_elems as u64 * df.word_length() as u64; // 4 B for fp32, 1 B fp8/int8, else 2 B fp16
    // The footprint guard's WINDOW END — the last byte this access can touch.
    //
    // The default models the access as the operand's whole materialized block at `+slice`: right
    // for every BLOCK-base offset (stick-multiples — RoPE's half-slices, the per-head attention
    // `col_at` bases), where the offset genuinely shifts the block.
    //
    // ⭐ A `lane_read` is the OTHER addressing shape: a SUB-STICK offset on a COLLAPSED stick dim
    // (the RedStick stride-0 out-broadcast fold). The fold makes every out-position of the
    // iteration read the SAME element of the operand's one materialized stick per row, so the
    // offset addresses a LANE WITHIN each row's stick (exx2's E[x²] pair slot at element 8, read
    // by rmsnorm's `rmxn`) — it does NOT shift the block. The touched set is one word per
    // materialized row at `row_base + slice`; its last byte is `arg_bytes − stick + slice + word`
    // (the block's last row's lane), NOT `slice + arg_bytes` (which shifts EVERY row and overshoots
    // the last row by the offset — a false positive against a footprint the access never leaves).
    // Block offsets keep the default model, so this branch is unreachable for every emission that
    // existed before the first lane read.
    let word = df.word_length() as u64;
    let stick = df.elems_per_stick() as u64 * word;
    let window_end = if lane_read {
        arg_bytes.max(1).saturating_sub(stick) + slice + word
    } else {
        slice + arg_bytes.max(1)
    };
    if let Some(l) = layout {
        // 0. PINNED-LX residency — the name-keyed consult, before any HBM placement. An
        //    LX-resident intermediate's address is its PIN (byte-granular, allocated by
        //    `BundleLayout::synth_lx_op_local`), so resolve to segment 0 + pin and let
        //    `per_core_addr` fold the per-core slice corner on top. This arm is the ONLY
        //    place a sub-stick offset on a collapsed stick dim (the `rinv` lane read at
        //    element 8) survives to the wire BYTE-exact: the HBM path's L3 staging is
        //    stick-granular (the EBR floors the +16 away and the reader's 2bsplat reads
        //    the landing stick base at imm 0 — element 0 — silently wrong), while an LX
        //    allocate is byte-granular end-to-end (bake-verified: the +16 lands in the
        //    compiled reader's lxlu register).
        if let Some((pin, fp)) = l.lx_pin(name) {
            // A CORE-LOCAL tensor's pin holds ONE core's slice, so the whole-operand window is not its
            // bound — `core_local_reads_agree` checks the per-core slice against it. And it has no
            // addressable offset: every core's slice starts AT the pin.
            if l.lx_core_local(name) {
                if slice != 0 {
                    return Err(SuperDscError(format!(
                        "core-local LX tensor '{name}' addressed at offset {slice}B — each core's slice \
                         sits at the pin, so a view into it has no offset to take"
                    )));
                }
                return Ok(SegFacts {
                    seg_base: 0,
                    intra_base: pin,
                    role: crate::placement::SegRole::Intermediate,
                    segment: crate::placement::SegRole::Intermediate.segment(),
                    bank: 0,
                    lx: true,
                });
            }
            if window_end > fp {
                return Err(SuperDscError(format!(
                    "lx-resident '{name}': access window end {window_end}B (offset {slice}B, \
                     operand {arg_bytes}B) exceeds its pinned-LX footprint {fp}B — shape \
                     mismatch (the access would run past the pin into the next pinned \
                     intermediate). Fix the declared shape (BundleLayout::synth_lx_op_local) or the \
                     op's view/offset."
                )));
            }
            return Ok(SegFacts {
                seg_base: 0,
                intra_base: pin + slice,
                role: crate::placement::SegRole::Intermediate,
                segment: crate::placement::SegRole::Intermediate.segment(),
                bank: 0,
                lx: true,
            });
        }
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
            if window_end > align128(p.size).max(128) {
                return Err(SuperDscError(format!(
                    "t{tid}: access window end {window_end}B (offset {slice}B, operand \
                         {arg_bytes}B) exceeds its placement footprint {}B (seg{}) — the op \
                         addresses PAST its own tensor and would alias whatever is placed next. \
                         The offset arithmetic and the placement must come from the SAME layout \
                         authority.",
                    p.size, p.segment
                )));
            }
            return Ok(SegFacts {
                seg_base: segment_base(p.segment)?,
                intra_base: p.offset + slice,
                role: p.role,
                segment: p.segment,
                bank: p.bank,
                lx: false,
            });
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
            if window_end > fp {
                return Err(SuperDscError(format!(
                    "synth '{name}': access window end {window_end}B (offset {slice}B, operand \
                     {arg_bytes}B) exceeds its declared footprint {fp}B — shape mismatch (would \
                     alias the next intermediate). Fix the declared shape (BundleLayout::synth) \
                     or the op's view/offset."
                )));
            }
        } else {
            // UNDECLARED synth (rmsnorm/silu decomposition, not yet shape-typed): grow-on-access
            // high-water so a piecewise multi-write still reserves its true footprint (correct,
            // but UNCHECKED — these should be migrated to declared shapes).
            let end = align128(off + window_end);
            if end > s.next {
                s.next = end;
            }
        }
        return Ok(SegFacts {
            seg_base: segment_base(seg)?,
            intra_base: off + slice,
            role: crate::placement::SegRole::Intermediate,
            segment: seg,
            bank: 0,
            lx: false,
        });
    }
    // No layout (single-op fixtures / unit tests): per-op segment by arg position.
    Ok(SegFacts {
        seg_base: segment_base(arg_index)?,
        intra_base: slice,
        role: role_of_segment(arg_index),
        segment: arg_index,
        bank: 0,
        lx: false,
    })
}

pub fn emit_sdsc(
    op_name: &str,
    op: &OpSpec,
    folds: &SdscFoldSet,
    layout: Option<&BundleLayout>,
) -> Result<(SdscOp, Vec<SegFacts>), SuperDscError> {
    let cores = folds.core_fold();
    let views: Vec<ArgView<'_>> = op.args.iter().map(|a| a.view()).collect();
    let out_idx = views.len().saturating_sub(1);
    // ⭐ AN LX-RESIDENT OUTPUT DEMANDS A REPLICATED PRODUCER — the BACKSTOP for
    // `apply_lx_residency` (the consult lives in the pointwise door; this guard lives in
    // the ONE emission path so no future builder can route around it). LX is per-core
    // private: a consumer on core c reads only the copy core c's own producer pass
    // wrote, so the producer must run the WHOLE tensor on EVERY core
    // (`WorkPlan::replicate`: all MAX_CORES, no splits) — and must not be time-tiled
    // (a trip-sliced producer leaves rows unwritten in most cores' copies).
    // Violations are a build `Err` naming the law, never silently-unreadable LX bytes.
    // A CORE-LOCAL output is the one exception: its producer keeps its own division, which
    // `apply_lx_residency` recorded and holds every reader to.
    if let Some(v) = views.get(out_idx)
        && v.allocation.is_lx()
        && !v.allocation.is_core_local()
    {
        let replicated =
            op.iter.cores_used().get() == MAX_CORES && op.iter.splits().values().all(|&s| s == 1);
        if !replicated || op.time_tile.is_some() {
            return Err(SuperDscError(format!(
                "op '{op_name}': its LX-resident output '{}' is produced by a non-replicated \
                 op — an LX-resident tensor must be produced by a REPLICATED op (all \
                 {MAX_CORES} cores, no work-division splits, no time tiling), because each \
                 core's LX is private and only holds what its own producer pass wrote. Declare \
                 the producer through the layout-consulted pointwise door \
                 (`apply_lx_residency`) or keep the tensor in HBM (`BundleLayout::synth`).",
                v.name
            )));
        }
    }
    core_local_reads_agree(op_name, op, &views, layout)?;
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
        // ⭐ THE TOP-K PAIR'S OUT DIM STREAMS ONE SLOT PER DATASTAGE, not the whole
        // per-core slice. Stating the slice (slots/ring — 8 at the 26b's slots=64,
        // ring=8) makes the vendor DDC's minimum-datastage check refuse the WHOLE
        // BUNDLE at bake: `allocAllMem(false)` → "Cannot allocate even the smallest
        // size" (ddcv1.cpp:1351) — the pair's opaque-op register charge scales with
        // the out unroll, and a slice-stated minimum prices the op past the
        // post-reserveFrontendLx pool ([`crate::superdsc_opspec::VENDOR_LX_SCHED_BYTES`],
        // 406,323 B). Card-mutation-proven at the 26b decode bundle (b6e29c0c9d27eadf,
        // group_7/sdsc_56 = rtkix): mutating ONLY `ss_`/`el_` `out_` 8→1 compiles the
        // same dsc clean; `N_`, `numWkSlicesPerDim`, `coreIdToWkSlice`, `coreIdToDsc`
        // untouched. ONE is also the vendor fixtures' own per-core out geometry
        // (slots=8, one slot per core — the only form the fixtures demonstrate), and
        // the DDC maximizes the datastage upward itself once the min-check passes,
        // so the work split, the landing bytes and every consumer are unchanged.
        let staged = if op.op.is_topk_pair() && d.name == "out" {
            1
        } else {
            per_core
        };
        set_iter_dim(&mut ss, d.name, staged);
    }
    let stage = StageParam {
        ss_: ss.clone(),
        el_: ss,
    };

    // The fused-epilogue's extra operand (if any): its Tensor{i} still gets a labeledDs_/primaryDsInfo_
    // entry below (it is a real tensor the on-card walk must address), but it is NOT one of THIS op's own
    // inputLabeledDs/outputLabeledDs — it belongs to the SECOND computeOp_ entry built after the loop.
    let epilogue_arg_idx: Option<usize> = match op.op_info {
        OpInfo::FusedEpilogue { arg_idx, .. } => Some(arg_idx),
        _ => None,
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
            scale_: wire_order(v, v.scale.iter().map(|s| s.to_i64()).collect()),
            wordLength: v.df.word_length(),
            // Per-tensor dtype from the arg's typed `Df` (`v.df`, the SINGLE source of truth) — NOT a name
            // suffix. `Df::Fp32` → IEEE_FP32 (4-byte, the KSPLIT fp32-SFP-merge partials); `Df::Fp8` →
            // SEN143_FP8 (1-byte, the packed fp8 W8A8 path); `Df::Bf16` → BF16E; else SEN169_FP16. INERT for
            // the working build (every fp16 arg keeps `Df::Fp16`, so the numbers are byte-identical).
            // `Df::Bf16` is DORMANT (no arg carries it; the score path stays SEN169_FP16 — bf16-output matmul
            // is dxp-rejected, see main's `lower_attn_node`). The matmul PSUM accumulates in SEN169_FP16 (DeepTools
            // `bmm.ddl`: `%ptsum_fp`/`%pesum` are both `%type_fp16 = SEN169_FP16` — RCUDD1A has NO fp32 psum;
            // only `bmm_sen1p5.ddl` binds IEEE_FP32). NOTE: SEN169_FP16 is 1-6-9 (6-bit exp, bias 31) ⇒ max
            // finite ≈ 4.3e9, NOT 65504 (that is IEEE-fp16's 5-bit-exp max). Kani-proven
            // (`sen169_holds_attention_score_no_overflow`): a ~1e5 score is a NORMAL SEN169 value.
            dataFormat_: v.df.dataformat(),
            // ⛔⛔⛔⛔⛔ A GATHER'S INDEX IS **HBM-ONLY** — no LX residency — and the scheduler refuses
            // the bake otherwise.
            //
            // MEASURED ON THE CARD POD with `hbm_lx()`:
            //   `sbf-ddc: DtException: Expect a valid allocate node.,
            //    L3DlOpsScheduler.cpp:2337`
            // That line is in `calculateFlopPerByte`, which for EVERY HBM-pinned labeledDs does
            // `lds.memOrg_.at(LX).allocateNode_` and `DT_CHECK_MSG(allocNode, ...)`. `allocAllMem` runs
            // first and gives an LX chunk to each staged operand — but not to the index, because an
            // index is not an arithmetic input (it is read into the IBR, never staged). So declaring an
            // LX residency the allocator will not fill leaves a null node the estimator then
            // dereferences.
            //
            // ⭐ AND THE VENDOR'S OWN INDEX IS EXACTLY THIS SHAPE:
            //   test_gather_1core/sdsc_1.json  index  memOrg_ { "hbm": { "isPresent": 1 } }
            //                                  value  memOrg_ { "hbm": ..., "lx": ... }
            // `MemOrg::hbm_only()` has existed in `wire.rs` since the gather's first port with ZERO
            // call sites and the doc comment "index tensors must reside in HBM — no LX indirect
            // addressing". This is that call site.
            memOrg_: match (v.allocation.is_lx(), &v.role) {
                (true, _) => MemOrg::lx_only(),
                (false, Role::Index) => MemOrg::hbm_only(),
                (false, _) => MemOrg::hbm_lx(),
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
                    layoutDimOrder_: wire_order(v, v.layout.to_vec()),
                    stickDimOrder_: vec!["out", "mb"],
                    stickSize_: vec![8, 8],
                }
            } else if is_fp8_kernel {
                LayoutInfo {
                    layoutDimOrder_: wire_order(v, v.layout.to_vec()),
                    stickDimOrder_: vec!["in", "out"],
                    stickSize_: vec![2, 64],
                }
            } else {
                LayoutInfo {
                    layoutDimOrder_: wire_order(v, v.layout.to_vec()),
                    stickDimOrder_: vec![v.stick],
                    stickSize_: vec![v.df.elems_per_stick()], // fp16/bf16: 64; fp32: 32; fp8/int8: 128
                }
            }
        });
        // computeOp_ wiring: `Tensor{i}-idx{i}`; the LAST arg is the output. The epilogue's own
        // operand is excluded here — the SECOND computeOp_ entry references it explicitly below.
        if epilogue_arg_idx == Some(i) {
            continue;
        }
        // ⛔⛔ THE INDEX OPERAND IS NOT AN ARITHMETIC INPUT, AND THE VENDOR FIXTURE IS EXPLICIT.
        // `dxp/test/test_gather_1core/sdsc_1.json` has three tensors and lists them as
        // `inputLabeledDs ["Tensor0-idx0"]`, `outputLabeledDs ["Tensor2-idx2"]`,
        // `indirectAccessIndexLabeledDs ["Tensor1-idx1"]` — the index appears in exactly ONE list,
        // and it is not the input list. It is read by the ADDRESSING, never by the arithmetic.
        //
        // Listing it as an input would hand dxp an op with one more operand than its DDL declares:
        // a 2-input identity, a 3-input matmul. That is an arity the op function has no register for,
        // and the failure is not a clean rejection — `attach_gather_index` shapes the index operand
        // FROM the value operand (so its rank and stick law are legal), which is exactly what makes a
        // spurious input plausible enough to lower.
        if matches!(
            op.indirect.and_then(|ia| ia.role_of(i)),
            Some(IndirectRole::Index { .. })
        ) {
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
    //
    // ⛔⛔⛔ EXCEPT WHEN THE OP FUNC'S OWN DDL BINDS A DIFFERENT INPUT ARITY, WHICH IS A HARD MATCH
    // CRITERION AND NOT A HINT. `ddl_conversion.cpp:2161-2163` (`DdlConversion::matchDdl2Dsc`) walks
    // every `ddl.operation_bind` in the selected template and skips any computeOp whose
    // `inputLabeledDs.size() != opBindInputs.size()` — an arity mismatch is indistinguishable from
    // "wrong op", so the template is silently declared unsuitable and the whole descriptor is refused
    // with a message that names only the op. A SEEDED reduce (`sum`/`max`/`mean` →
    // `summeanmaxexx2.ddl`) binds TWO inputs and is unchanged here; the machine's own FUSED reduces
    // (`quantscalepertokenfp8` → `quant_scale_per_token.ddl:41`, one `%input_tensor`, one
    // `%output_tensor`, its accumulator an on-chip `ddl.internal_tensor`) bind ONE.
    //
    // ⭐ DRIVEN BY [`OpFunc::ddl_input_arity`], a const table — a target-ABI fact this shared pass
    // takes as INPUT, never a match arm here.
    if op.is_reduction && views.len() == 2 && op.op.ddl_input_arity().is_none_or(|n| n == 2) {
        input_refs.push(format!("Tensor{out_idx}-idx{out_idx}"));
    }
    // ⭐⭐⭐ THE ARITY REFUSAL, MOVED FROM THE BAKE TO THE BUILD. `emit_sdsc` runs inside `#[forward]`,
    // so this `Err` is a `cargo build` error naming the contract — not a `DtException: Scheduler failed
    // to find a suitable op mapping` hours later on a card, which is how the mismatch above was found.
    if let Some(want) = op.op.ddl_input_arity()
        && input_refs.len() != want
    {
        return Err(SuperDscError(format!(
            "op '{op_name}' ({}): its DDL template binds exactly {want} input tensor(s), but this \
             descriptor lists {} ({input_refs:?}). `DdlConversion::matchDdl2Dsc` \
             (ddl_conversion.cpp:2161) SKIPS a computeOp whose input count differs from the \
             `ddl.operation_bind`'s, so the template is declared unsuitable and dxp refuses the \
             whole bundle with a message that names only the op. Fix the operand list, or the \
             `OpFunc::ddl_input_arity` row if the template really binds a different arity.",
            op.op.name(),
            input_refs.len()
        )));
    }

    // ⛔⛔⛔⛔⛔ A GATHER ON A `KERNEL`-BEARING OP IS REFUSED AT BUILD — EXCEPT THE ONE FORM THE CARD
    // PROBE PROVED SCHEDULABLE: the TOKEN-PLANE form (entries on `mb`, one position each, no second
    // pin — the prefill-m>1 per-expert batched matmul).
    //
    // ## The original refusal, and what superseded it
    // MEASURED ON THE CARD (pod nickm-7db9667cdd-z2jc6, granite-3.1-2b fp8), with the gather on the
    // prefix score leg's Kᵗ operand. Two bakes, two refusals, bracketing the cause:
    //   index memOrg_ = hbm + lx  ->  sbf-ddc: DtException: Expect a valid allocate node.
    //                                 L3DlOpsScheduler.cpp:2337
    //   index memOrg_ = hbm       ->  sbf-ddc: DtException: Expect LX in labeledDs memOrg_.
    //                                 L3DlOpsScheduler.cpp:2334
    // Both lines are in `calculateFlopPerByte`, which for every HBM-pinned labeledDs requires an LX
    // `memOrg_` entry (2334) AND a non-null LX allocate node (2337). `allocAllMem` gives an LX chunk
    // to each STAGED operand and never to an index — an index is read into the IBR, not staged — so
    // no `memOrg_` satisfies both. That measurement is real, and it is also NOT the whole story.
    //
    // ⭐ THE 2026-10-09 PROBE (`/tmp/prefill-m1-probe-report.md`) ROOT-CAUSED AND ESCAPED IT:
    // `calculateFlopPerByte` runs only inside CHUNK-PARAM EXPLORATION (`addChunkDataStage → … →
    // findBestParamsForArithmeticIntensity`, gated on `isReuse` at `:1550`), and demands — via
    // `getLdsL3TransferNodes` (`:1779`) — TRANSFER USERS on the index's HBM alloc. Those users are
    // inexpressible in the validated input schema (only `allocate` nodes with the curated field set)
    // and are created only by `createStoreIndexTensorToLx`, which runs in `processHbmPagedTensors`
    // AFTER `addChunkDataStage`: a VENDOR ORDERING GAP, not a gather prohibition. With
    // `DISABLE_ABOVE_LX_CHUNK_EXPLORE=1` at bake, the token-plane form schedules END TO END — `RC=0`
    // at both the small (m8, in128) and 26b-gate (m96, in2816) geometries, `init_binary.bin`
    // emitted, the gather ENGAGED (kernel transfer `srcIndirect=YES`, index staged to the L3LU IBR,
    // the vendor's own paged loop nest, 10 syncs). The index spelling is `"index"` on every path —
    // the wall-#8 law at the `indexTensorType_` emission site: `"address"` bypasses the vendor's
    // idx2addr conversion and puts the ABSOLUTE bank base into the EAR-domain mutable constant,
    // which refuses past the 2^28-element EAR range at 26b chunk addresses.
    //
    // ## What stays refused, and why here
    // Every OTHER gathered-KERNEL shape — the paged-KV forms (`of_pool_blocks`,
    // `of_pool_blocks_per_row`) and any token-plane variant with a pin ≠ 1 — was NEVER probed, and an
    // unprobed descriptor that reaches dxp re-runs the card loop this refusal exists to close.
    // The predicate is decidable from what this function has already built, so it is decided here,
    // once, with both the admitted shape and the KERNEL-LESS alternative named in the message.
    let token_plane = match op.indirect.as_ref() {
        Some(ia) => {
            ia.entry_dim == crate::superdsc_opspec::KernelAxis::Batch
                && ia.per_position.is_none()
                && ia.page.get() == 1
        }
        None => false,
    };
    let kernel_bearing = primary.len() > 1 && primary.contains_key("KERNEL");
    if op.indirect.is_some() && kernel_bearing && !token_plane {
        return Err(SuperDscError(format!(
            "op '{op_name}': a GATHER is declared on an op whose primaryDsInfo_ carries a KERNEL \
             ({:?}) in a shape that was never probe-validated. The card refused the paged-KV form \
             twice (hasDimensionReuse L3DlOpsScheduler.cpp:303 turns on the arithmetic-intensity \
             explorer at :1550, whose calculateFlopPerByte demands an LX allocate node for every \
             HBM-pinned labeledDs at :2334/:2337 — and allocAllMem never gives one to an index); the \
             ONE gathered-KERNEL shape that schedules is the TOKEN-PLANE form (entries on `mb`, one \
             position each, no second pin — GatherIndex::of_expert_planes), probe-proven RC=0 at \
             both probe geometries with DISABLE_ABOVE_LX_CHUNK_EXPLORE=1 at the bake. Otherwise \
             gather on a KERNEL-LESS op: both vendor fixtures use an elementwise copy (`identity` / \
             `AddZero`) whose operands are all typed OUTPUT, and feed its RESULT to the matmul.",
            primary.keys().collect::<Vec<_>>()
        )));
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
    // ⭐ THE PER-ARG RESOLVED FACTS, collected beside the addresses they came from —
    // one `SegFacts` per operand, threaded to `EmittedOp::new` so the binding
    // carries the door's OWN resolution and no consumer ever re-derives it.
    let mut arg_facts: Vec<SegFacts> = Vec::with_capacity(views.len());
    for (i, v) in views.iter().enumerate() {
        let comp = v.allocation.component();
        let name = format!("allocate-Tensor{i}_{comp}");
        if !seen.insert(name.clone()) {
            continue;
        }
        // This operand's side of a gathered access, if the op declares one. Derived from the op's
        // single `Option`, never from a per-operand flag, so the index and value sides cannot
        // disagree about which pair they belong to.
        let indirect_role = op.indirect.as_ref().and_then(|ia| ia.role_of(i));
        // ── THE PAGED DIMS ──
        // ⛔ PINNING A DIM IS WHAT MAKES IT PAGED: `getPageSize` (`dsc2.cpp:4493-4526`) erases every
        // negative entry of `maxDimSizes_`, so the pinned dims ARE the paged dims and each must
        // satisfy `ss_(dim) % pageSize(dim) == 0` (`L3DlOpsScheduler:6655`).
        //
        // ⭐ PLURAL. This pinned exactly one dim, which caps a gather at collapsing the PAGE axis: an
        // entry then names a page but not a batch ROW, so a launch still serves one row. IBM's paged
        // attention pins TWO (`[-1,-1,64,1]`: a page granularity of 64 and a second axis at 1, one
        // entry per position) and that is what puts both relaunch factors inside the index.
        //
        // ⭐ RESOLVED AGAINST THIS OPERAND'S OWN LAYOUT, never by a carried position. `KernelAxis` is
        // a closed set, which is what makes these lookups succeed on the shipped path — but it is NOT
        // a substitute for looking: an activation's layout is `["mb","in"]`, so a positional pin would
        // land on a different axis entirely (see the note on `KernelAxis` for the draft that did
        // exactly that).
        //
        // Resolved before the node is built because the failure is otherwise SILENT: a value tensor
        // whose axis names a dim it does not have would still emit a `value_tensor` node, still
        // cross-link, and still bake — gathering with no page declared, i.e. wrong addresses from a
        // clean build.
        let paged_dims: Vec<(usize, crate::superdsc_opspec::PageExtent)> = match &indirect_role {
            Some(IndirectRole::Value { pins, .. }) => pins
                .iter()
                .map(|&(d, page)| {
                    let at = v.layout.iter().position(|&x| x == d).ok_or_else(|| {
                        SuperDscError(format!(
                            "op {op_name}: operand {i} ('{}') is the gathered VALUE tensor, but the \
                             declared paged axis '{d}' is not in its layout {:?}. An unresolvable \
                             axis would emit a value_tensor with a MISSING page, which gathers wrong \
                             addresses and still bakes. A gathered kernel must be declared with every \
                             axis it pages — including `mb` when the batch is paged.",
                            v.name, v.layout,
                        ))
                    })?;
                    Ok((at, page))
                })
                .collect::<Result<Vec<_>, SuperDscError>>()?,
            None | Some(IndirectRole::Index { .. }) => Vec::new(),
        };
        // ⭐⭐⭐⭐⭐ THE ITERATION SPACE **THIS OPERAND** IS MEASURED IN, resolved ONCE for every reader
        // below. Every operand but one is measured in the op's own plan; the gather's INDEX is measured
        // in ENTRIES ([`WorkPlan::of_index_entries`]), because dxp derives the entry count as the op's
        // extent divided by the value tensor's page and converts exactly that many, contiguously.
        //
        // ⛔ ONE BINDING, NOT A DIVISION PER READER. The extents of a view are read FOUR times below —
        // the arrangement classifier, the coordinate folds, the on-card walk and the per-core START —
        // and only two of them go through `DeviceExtents::of_view`, so a declared extent reaches half of
        // them. That is how the index came to declare the op's 2048 `mb` positions and step each core's
        // start by 64 entries (256 B) into a 128-byte buffer: 31 of 32 cores read bytes nothing
        // converted, as ABSOLUTE stick addresses. Substituting the plan once cannot be half-applied.
        let iter = match &indirect_role {
            Some(IndirectRole::Index { pins, .. }) => {
                std::borrow::Cow::Owned(op.iter.of_index_entries(pins).ok_or_else(|| {
                    SuperDscError(format!(
                        "op {op_name}: operand {i} ('{}') is the gather's INDEX, but its paged dims \
                         {pins:?} do not divide this op's iteration extents into whole entries. dxp \
                         converts exactly `extent / page` entries and no declaration changes that, so \
                         a partial entry is an address rather than a shape.",
                        v.name,
                    ))
                })?)
            }
            None | Some(IndirectRole::Value { .. }) => std::borrow::Cow::Borrowed(&op.iter),
        };
        let iter: &WorkPlan = &iter;
        // ⛔⛔ skip_addr IS THE PIN × THE PER-CORE EXTENTS OF THE **UNPINNED** DIMS — so a gathered
        // value operand whose op SPLITS an unpinned dim shrinks every index step below the plane the
        // entries enumerate. That is a DEFECT only where the semantics demand the whole plane: the
        // KERNEL-bearing token-plane gather (the prefill-m>1 per-expert batch matmul), where an entry
        // names ONE EXPERT PLANE `in·out` and the probe's plan split `mb` only ({in:1, mb:32, out:1})
        // — a K-split under that gather is a build error, not a silently-short gather on the card.
        // On a KERNEL-LESS copy an unpinned dim is legal by design (IBM's own paged fixture
        // `sdsc_add_paged_l3lu` leaves `mb` unpinned, and the copy path's split law — whole entries,
        // `gather_copy_cores` — is enforced where the plan is made), so the guard is scoped to the
        // KERNEL predicate, exactly the `hasDimensionReuse` condition that gates dxp's own reuse-path
        // handling of the gather.
        if let Some(IndirectRole::Value { pins, .. }) = &indirect_role
            && primary.contains_key("KERNEL")
        {
            for &d in v.layout.iter() {
                if pins.iter().any(|&(pd, _)| pd == d) {
                    continue;
                }
                let split = iter.split_of(d);
                if split > 1 {
                    return Err(SuperDscError(format!(
                        "op {op_name}: operand {i} ('{}') is the gathered VALUE tensor, and the \
                         work division splits its unpinned axis '{d}' ({split} ways) — skip_addr is \
                         the pin × the per-core extents of the unpinned dims, so each index entry \
                         would step a FRACTION of the expert plane it names. Split the entry axis \
                         only.",
                        v.name,
                    )));
                }
            }
        }
        // THE ONE device layout for this tensor view, computed ONCE and handed to BOTH the on-card WALK
        // (build_coordinates) and the per-core START (per_core_addr). Neither derives its own, so the walk
        // and start cannot address the tensor differently — a divergence is not representable (task #11).
        let sl = view_stick_layout(v, iter);
        let coordinates = build_coordinates(v, &sl, iter, fp8_matmul, op.is_reduction);
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
        let arg_bytes = materialized_bytes(v, iter);
        // ⭐⭐⭐ A GATHERED VALUE OPERAND SPANS **ONE ENTRY**, NOT ITS DECLARED WALK — and the footprint
        // guard has to measure the same thing dxp does or it refuses the shipped shape.
        //
        // `addr = idx * skip_addr + base_addr`: the walk's paged axis is CLAMPED to its pin
        // (`getBufferCapacityForNodePerDim`), and the index supplies the rest of the address. So the copy
        // op's gathered source declares `[mb = nkvh*nb*mq, out = hd*64]` while it only ever reads ONE
        // `out`-row at a time out of the pool — 512 KB declared against a 256 KB resident Kᵗ plane at
        // bs=8, which the guard below would call an overrun of a tensor the op never overruns.
        //
        // ⛔ AND THIS IS NOT A WEAKENING: where the entries POINT is checked where they are BUILT
        // (`gather_index_table` refuses a page a row does not own, an entry that overflows int32, or more
        // rows than a pass block holds). What stays checked here is what a gather cannot be told from the
        // index — that ONE entry fits inside the tensor at all.
        let arg_bytes = if paged_dims.is_empty() {
            arg_bytes
        } else {
            let mut clamped = 1u64;
            for (li, &d) in v.layout.iter().enumerate() {
                let ext = match v.scale.get(li) {
                    Some(Scale::Active) => iter.extent(d) as u64,
                    _ if d == v.stick => v.df.elems_per_stick() as u64,
                    _ => 1,
                };
                let pin = paged_dims
                    .iter()
                    .find(|&&(at, _)| at == li)
                    .map(|&(_, p)| p.get() as u64);
                clamped *= match pin {
                    Some(p) => ext.min(p),
                    None => ext,
                };
            }
            clamped * v.df.word_length() as u64
        };
        let facts = resolve_seg_base(layout, v.name, i, v.offset_elems, arg_bytes, v.df, {
            // ⭐ A SUB-STICK OFFSET ON A COLLAPSED STICK DIM IS A LANE READ — the RedStick
            // stride-0 fold reads ONE element per materialized row at `row_base + offset`
            // (exx2's E[x²] pair slot at element 8, read by rmsnorm's `rmxn`), so the offset
            // addresses a lane within each row's stick rather than shifting the block. Every
            // prior offset on a collapsed-stick operand is a stick-MULTIPLE block base (the
            // per-head attention `col_at` bases), which keeps the default window model — this
            // fires only for the lane read. See `resolve_seg_base`'s own doc for the window
            // arithmetic the flag selects.
            let stick_collapsed = v
                .layout
                .iter()
                .position(|&d| d == v.stick)
                .is_some_and(|p| !matches!(v.scale.get(p), Some(Scale::Active) | None));
            stick_collapsed
                && v.offset_elems > 0
                && !(v.offset_elems as u64).is_multiple_of(v.df.elems_per_stick() as u64)
        })?;
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
                l.declare_arrangement(v.name, view_stick_layout(v, iter))?;
            }
        }
        arg_facts.push(facts);
        let per_core = per_core_addr(
            facts.seg_base,
            v,
            &sl,
            iter,
            &wk_slice,
            facts.intra_base,
            cores,
        )?;
        // No two cores of an OUTPUT tensor may share a byte address (aliased write) — EXCEPT a K-split
        // matmul, where the `k` cores each contract a K-slice into a PARTIAL product that dxp
        // PSUM-accumulates into the SHARED output tile (torch-spyre's reduction-dim split). That case is
        // legitimate ONLY when a dim the op REDUCES — split, and absent from THIS output's layout (`in`
        // for a matmul, whose output is `[mb,out]`) — is the one split. An `mb`/`out` shared output is
        // still a real collision, so the guard stays for every non-reduction split.
        //
        // ⭐ AND A SEEDED REDUCE SPLIT ALONG THE AXIS IT REDUCES — the same law for the reduce family's
        // own cross-core combine: `summeanmaxexx2.ddl` binds `genericpartialreduction` and rings each
        // core's partial over its `core_to_core_communication(%reduce_dim…)` (FMAX for max, FMA16 for
        // sum), the last core writing the row's result. The reduced dim is present in the accum's layout
        // but COLLAPSED there (`Scale::RedStick`), so every core addresses the same accum stick — the
        // shared output the ring writes. Admitted only for that exact shape AND only for an op whose
        // template provably carries the combine (`OpFunc::template_combines_a_core_split_reduction`, a
        // const table), never because "it is a reduce".
        let collapsed_reduction_split = op.is_reduction
            && op.op.template_combines_a_core_split_reduction()
            && op.iter.dims().iter().any(|d| {
                d.is_reduction
                    && op.iter.split_of(d.name) > 1
                    && v.layout
                        .iter()
                        .position(|&l| l == d.name)
                        .is_some_and(|li| v.scale.get(li).is_some_and(|s| s.is_stick_reduction()))
            });
        let reduction_split =
            (op.iter.split_of("in") > 1 && !v.layout.contains(&"in")) || collapsed_reduction_split;
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
            // ⛔⛔⛔⛔⛔ A GATHERED VALUE TENSOR IS **NOT** SYMBOLIC — `Some(0)`, IBM's paged fixture's
            // own value, and this was `Some(1)`.
            //
            // MEASURED ON THE CARD POD: with `Some(1)` and a concrete per-core address, dbo refuses the
            // bake —
            //   `dbo: DtException: Symbol operand does not exist: 34360262656,
            //    VariableDefinition.cpp:303`
            // — 34360262656 being the operand's real BYTE ADDRESS, read as a symbol id. `Some(1)` says
            // "the start address in `startAddressCoreCorelet_.data_` is a symbol id, resolve it", and
            // ours holds a concrete address (`AddrFold::new`, not `AddrFold::symbolic`). The two must
            // agree, and the flag was set without the address ever changing.
            //
            // ⛔ AND THE FIX IS TO DROP THE FLAG, NOT TO MINT A SYMBOL, because the two vendor fixtures
            // disagree and the PAGED one governs:
            //   dxp/test/test_gather_1core/sdsc_1.json  value  isStartAddrSymbolic_ 1, data_ "-1"
            //   dcg/.../test/sdsc_add_paged_l3lu.json   value  isStartAddrSymbolic_ 0, data_ "128000"
            // The dxp file is a unit test OF THE SYMBOLIC PATH — its own `bundle.mlir` says so, and
            // passes `%base_addr` with `symbol_ids=[-1]` to trigger it. IBM's paged attention, which is
            // the case this emitter is reproducing, uses a CONCRETE base.
            //
            // ⭐ AND THAT SETTLES WHAT `base_addr` IS, which nothing else could: dxp's
            // `addr = idx * skip_addr + base_addr` adds the index to the operand's OWN declared start,
            // so the plane / kv-head / slot / feature terms the score leg bakes into its offset SURVIVE
            // and the index supplies only the PAGE term. That is exactly the arithmetic
            // `gather_entries_per_page` assumes.
            isStartAddrSymbolic_: match &indirect_role {
                Some(IndirectRole::Value { .. }) => Some(0),
                None | Some(IndirectRole::Index { .. }) => None,
            },
            // torch-spyre emits the HOST dim order verbatim (`[mb, in]`, mb-outermost) even for a
            // stick-major tensor — dxp applies `get_generic_stick_layout` INTERNALLY when it reconstructs
            // the stick-blocked `device_size` from `-1`, so the mb-outermost label + `-1` already yields the
            // stick-major walk (`element_torchspyre_parity`). Reordering here DIVERGES from the golden.
            layoutDimOrder_: wire_order(v, v.layout.to_vec()),
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
            maxDimSizes_: {
                let walk = sl.device_walk(
                    crate::sdsc_abstract::DeviceExtents::of_view(
                        &v.layout
                            .iter()
                            .map(|&d| iter.extent(d) as usize)
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
                );
                // Paged only on the gathered value side. Every pin was resolved against this operand's
                // own layout above, so `paged_at` cannot be the thing that drops one.
                //
                // ⭐ A dxp-INNERMOST-FIRST VIEW IS RECONSTRUCTED, NOT PINNED: dxp builds its walk from
                // the REVERSED `layoutDimOrder_` plus the view's back gaps, and a pinned extents array
                // would be read at the opposite end first — the same strides it derives from the labels,
                // restated in an order the labels do not carry. `-1` per dim hands the reconstruction to
                // the labels (the same spelling every stick-blocked view above takes). Ungathered only:
                // a paged axis keeps its pin, which is what `paged_at` states.
                if v.dxp_innermost_first && paged_dims.is_empty() {
                    crate::sdsc_abstract::StickLayout::kernel(1, 1)
                        .device_walk(&vec![1i64; v.layout.len()])
                } else if paged_dims.is_empty() {
                    walk
                } else {
                    walk.clone().paged_at(&paged_dims).unwrap_or(walk)
                }
            },
            // ── INDIRECT (GATHERED) HBM ACCESS ──
            // Reproduces `dxp/test/test_gather_1core/sdsc_1.json`, the vendor fixture that BAKES:
            // the two nodes CROSS-LINK (each names the other) and the compute op names the index
            // side. Driven by ONE `Option` on the op, so a half-declared pair — which lowers to a
            // message-less `map::at` inside dxp — is unrepresentable rather than caught late.
            //
            // `None` (every op today) yields exactly the previous three values, so this is
            // byte-identical for every shipped bundle. A test asserts that rather than trusting it.
            indirectAllocType_: match &indirect_role {
                None => "no_indirection",
                Some(IndirectRole::Index { .. }) => "index_tensor",
                Some(IndirectRole::Value { .. }) => "value_tensor",
            },
            relatedIndirectAccessAlloc_: indirect_role.as_ref().map(|r| {
                // The partner's node name, built by the SAME `format!` as `name` above so the
                // cross-link cannot drift from the node it points at.
                let partner = match r {
                    IndirectRole::Index { value, .. } => value,
                    IndirectRole::Value { index, .. } => index,
                };
                format!(
                    "allocate-Tensor{partner}_{}",
                    views[*partner].allocation.component()
                )
            }),
            // Only the INDEX side carries a tensor type; the vendor's value node omits it.
            //
            // ⭐⭐⭐ AND THE SPELLING IS `"index"` ON EVERY PATH — the 2026-10-09 wall-#8 law. An
            // `"address"`-typed index means the buffer ALREADY HOLDS page addresses and the value
            // tensor's start is only a within-page offset (GatherBuffers.cpp:198-208 zeroes it after
            // the vendor's own conversion: "the value tensor start address is the offset within a
            // page, that we do not need"). Spelling `"address"` while the mint still writes RAW
            // EXPERT IDS and the value start still carries the ABSOLUTE bank base is doubly wrong:
            // dcc routes that start into the EAR-domain mutable constant modulo 2^31 sticks-domain
            // wrap, and the EAR range is 2^21·128 B = 2^28 fp8 elements — so at 26b, gate/up chunk
            // c10's base 17,460,359,424 (mod 2^31 = 280,490,240 > 268,435,456) refused with
            // `MutableAddrSplitting.cpp:814 EAR overflow` (probe-pinned: start 17,229,672,704 and
            // 4,400,000,000 bake; 17,460,359,424, 268,435,457 and 278,435,456 refuse; the KV gather
            // — `"index"`-typed — is clean at the SAME 17,460,359,424). `"index"` instead engages the
            // vendor's GatherIndexConversion: it builds an INT32IDXTOADDR sdsc per gather computing
            // `base + e·skip` in the INTEGER domain (base = value start/256, skip = the per-plane
            // stride — 704·256 B = exactly one expert's chunk plane for the chunk-major bank, so the
            // chunk-major law needs NO extra emission), zeroes the value start, and flips the index
            // type to ADDRESS itself (GatherBuffers.cpp:199) before the scheduler's reuse path reads
            // it — `isIndexLds` (L3DlOpsScheduler.cpp:7614) passes on the CONVERTED descriptor. The
            // earlier `"index"`-died-at-`:1517`/`:1535` finding was the pre-wall-2/3 mint form; at
            // the current form the whole m>1 group bakes `RC=0` with the conversion ENGAGED
            // (init_binary grows by 185,856 B over 128 gathered sdscs — the idx2addr programs). The
            // fp32 mint must NOT compute the addresses itself: the base needs 26+ bits and fp32
            // carries 24.
            indexTensorType_: match &indirect_role {
                Some(IndirectRole::Index { .. }) => Some("index"),
                None | Some(IndirectRole::Value { .. }) => None,
            },
            startAddressCoreCorelet_: addr,
            // HBM back gaps, `-1`-keyed for all cores — torch-spyre's own spelling (`compute_ops.py`'s
            // `backGapCore_`). Absent unless a dim states one, and the ONE thing that states one on
            // this crate is a dxp-innermost-first view's declared `device_extent` wider than its sweep
            // (`phys − swept`): the only spelling of a physical extent dxp honours as a stride
            // (`sizesWithGaps_`, dsc2.cpp:2899 — a pinned `maxDimSizes_` entry is not one). Every
            // existing view declares no extent or serializes as stated, so this stays `None` for every
            // bundle the crate emitted before the flag existed.
            backGapCore_: {
                let gaps: serde_json::Map<String, serde_json::Value> = v
                    .layout
                    .iter()
                    .enumerate()
                    .filter_map(|(li, d)| {
                        let gap = if v.dxp_innermost_first {
                            v.device_extent
                                .get(li)
                                .copied()
                                .flatten()
                                .map_or(0, |p| p.saturating_sub(iter.extent(d)))
                        } else {
                            0
                        };
                        (gap > 0)
                            .then(|| (d.to_string(), serde_json::json!({ "-1": gap.to_string() })))
                    })
                    .collect();
                (!gaps.is_empty()).then_some(serde_json::Value::Object(gaps))
            },
            coordinates_: coordinates,
        });
    }

    // ⭐⭐⭐ AN OP FUNC WHOSE DDL READS NAMED EXTERNAL CONSTANTS SUPPLIES THEM HERE — `constantInfo_`,
    // NOT `opConsts`, and it REPLACES whatever the generic path would have written.
    //
    // `ddc/ddl_templates/test/sdsc_layernormscale.json` is the frontend-stage witness: a fused
    // reduce-style op carrying its `eps` as `constantInfo_: {"0": {dataFormat_, name_: "eps", data_}}`,
    // with ONE input, ONE output, no interim operand and `exUnit: sfp`. `opConsts_` in
    // `sengraphToPerfDsc/sengraphFoldingHelperAuxMethods.cpp:1432` is perfdsc's INTERNAL structure on
    // the way here, not the descriptor field ddc matches against — emitting the constants there left
    // `ddc.run_v1` refusing exactly as before.
    //
    // ⛔ IT MUST TAKE PRECEDENCE OVER [`OpInfo::ReduceScaling`], WHICH IS WHY IT IS NOT A MATCH ARM.
    // A fused reduce still arrives through the reduce assembler, so its `op_info` is
    // `ReduceScaling(1.0)` — the `scaling_factor` that `summeanmaxexx2.ddl` reads and
    // `quant_scale_per_token.ddl` does not. Written as a guarded arm it lost to `ReduceScaling` and the
    // three constants never reached the descriptor at all: MEASURED with
    // `L3DlOpsScheduler_standalone` + `ddc_standalone` on the emitted descriptor, `scaling_factor`
    // alone is refused and the three named constants are accepted.
    //
    // ⛔ THE TABLE IS THE WHOLE CONDITION: `op_consts()` is empty for every other op func, so every
    // shipped descriptor is byte-identical.
    let constant_info = if !op.op.op_consts().is_empty() {
        // ⭐ THE SAME POST-2026-09 CONTRACT THE REDUCE ARMS BELOW WRITE: `data_` is a FoldManager
        // OBJECT (`fold_manager_const`), one entry per named constant, NO `allocations_` (the
        // validator's additionalProperties-style rejection killed that field). Each value is ONE
        // SEN169 word — the DDL reads it per-element out of `dataFormat_`.
        serde_json::Value::Object(
            op.op
                .op_consts()
                .iter()
                .map(|&(cname, v)| {
                    serde_json::json!({
                        "dataFormat_": "SEN169_FP16",
                        "name_": cname,
                        "data_": fold_manager_const(sen169_bits(v) as u32, folds)
                    })
                })
                .enumerate()
                .map(|(i, entry)| (i.to_string(), entry))
                .collect(),
        )
    } else {
        match op.op_info {
            OpInfo::SfpConstTable => sfp_constant_table(folds),
            OpInfo::ReduceScaling(packed) => scaling_factor_const(packed, folds),
            OpInfo::ReduceScalingFp32(bits) => scaling_factor_const_fp32(bits, folds),
            OpInfo::Exx2Scale(bits) => exx2_consts(bits, folds),
            OpInfo::None | OpInfo::FusedEpilogue { .. } => serde_json::json!({}),
        }
    };
    // ⭐ THE SAMV COORDINATE MASK (a padded reduce's identity-valued padding): when the op
    // states one, its `samv-maskvalue` constant is APPENDED here at the next insertion index —
    // the vendor's own rule (`constants["samv-maskvalue"] = _get_mask_value(op)` then
    // `masking_const_id = len(constants)`, codegen/superdsc.py:2357-2365) — so the id is the
    // index the appended entry occupies and can never collide with the scaling/exx2 entries
    // above. `mask_bits` is the RAW SEN169 word (`0xFFFE` = the DDL's own negInf pattern for
    // `max`, 0 otherwise), NOT a `sen169_bits` encoding: the backend's own multi-row max
    // fixture pins the wire value (`"65534"`, `ddc/ddl_templates/test/sdsc_max.json`). See
    // [`crate::superdsc_opspec::SamvCoordinateMask`] for the full law.
    //
    // The same mask feeds the two DSC-level fields: `coordinateMasking_` dims in ELEMENTS
    // ([[valid, pad]] — valid + pad = the stick-padded extent) and the appended constant's id.
    // Both stay empty/-1 for every stick-aligned reduce, byte-identical to before this field
    // existed. The fixture's own `maskingConstId_: 8` counts ITS eight preceding constants;
    // ours is 1 — `scaling_factor` occupies "0".
    let (constant_info, coordinate_masking, masking_const_id) = match op.coordinate_mask {
        None => (constant_info, BTreeMap::new(), -1),
        Some(m) => {
            let mut entries = constant_info.as_object().cloned().unwrap_or_default();
            let idx = entries.len() as i64;
            entries.insert(
                idx.to_string(),
                serde_json::json!({
                    "dataFormat_": "SEN169_FP16",
                    "name_": "samv-maskvalue",
                    "data_": fold_manager_const(m.mask_bits, folds)
                }),
            );
            let mut dims: BTreeMap<String, Vec<[i64; 2]>> = BTreeMap::new();
            dims.insert(m.dim.to_string(), vec![[m.valid as i64, m.pad as i64]]);
            (serde_json::Value::Object(entries), dims, idx)
        }
    };
    // torch-spyre's `generate_constant_info` returns the JSON *string* "{}" for the
    // EMPTY case and a DICT only when populated. VERIFIED on the REAL frontend (compiled
    // `torch.rsqrt` + `F.rms_norm`): rsqrt/add/mul → `"constantInfo_": "{}"` (a str), the
    // `mean` op's scaling_factor → a dict. The empty OBJECT `{}` is FALSY in dxp's Python
    // (`if constantInfo_:`), so the SFP constant-table / Newton-Raphson-refine setup is
    // SKIPPED — transcendentals collapse to their seed (rsqrt→1/x, sqrt→identity): the
    // 35×-magnitude rmsnorm bug, with the matmul (no SFP, no NR) unaffected. The truthy
    // string "{}" makes dxp run the setup so the NR refinement executes on-card.
    //
    // ⛔ THE VENDOR TOPK PAIR IS THE EXCEPTION — BOTH its fixtures ship the falsy empty OBJECT,
    // and falsy is CORRECT for it: `topk.ddl` reads no external constant (its negInf/negOne/one
    // ride its own `ddl.define_constant` LRFs) and has no NR refinement to run, so the truthy
    // string can only install a table the template never consumes. See
    // [`OpFunc::is_topk_pair`] for the fixture citations.
    let constant_info =
        if !op.op.is_topk_pair() && constant_info.as_object().is_some_and(|o| o.is_empty()) {
            serde_json::Value::String("{}".to_string())
        } else {
            constant_info
        };

    // The fused epilogue's SECOND computeOp_ entry (when present): mirrors `sdsc_bmm_lxopt.json`'s
    // `MatMul_122` golden — it reads THIS op's own output plus the extra operand, and writes back
    // to the SAME output in place (no new HBM tensor, no separate trip). Built before `dsc` so both
    // entries share one `computeOp_` vec.
    let epilogue_compute_ops: Vec<ComputeOp> = match op.op_info {
        OpInfo::FusedEpilogue { arg_idx, op_func } => {
            let out_ref = format!("Tensor{out_idx}-idx{out_idx}");
            let epi_ref = format!("Tensor{arg_idx}-idx{arg_idx}");
            vec![ComputeOp {
                exUnit: "sfp",
                opFuncName: op_func.name().to_string(),
                // Matches the golden's biasadd entry exactly: plain fp16, regular fidelity — the
                // epilogue's own operand is always fp16 (mask/bias/residual), independent of whatever
                // dtype the matmul itself ran in.
                attributes_: ComputeAttrs {
                    dataFormat_: Fp16::NAME,
                    fidelity_: "regular",
                },
                // The golden's biasadd entry historically carried `"location": "invalid"`, mirrored
                // verbatim — but `"invalid"` is not a member of deeptools' `LoopNames` enum
                // (dscdefn.h:48; its string table is Inner/dbin/dbout/...), and the dev stack's
                // new SDSC-JSON validator rejects it outright ("Must be 'Inner'"). The epilogue
                // runs in the same loop nest as its matmul, so "Inner" — the same location every
                // other computeOp in a bundle carries — is also the semantically right value.
                location: "Inner",
                inputLabeledDs: vec![out_ref.clone(), epi_ref],
                interimLabeledDs: vec![],
                outputLabeledDs: vec![out_ref],
                indirectAccessIndexLabeledDs: vec![],
            }]
        }
        _ => Vec::new(),
    };

    let dsc = Dsc {
        numCoresUsed_: cores,
        numCoreletsUsed_: folds.corelet_fold(),
        coreIdsUsed_: (0..cores).collect(),
        N_: n_space,
        coordinateMasking_: coordinate_masking,
        maskingConstId_: masking_const_id,
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
                    } else if matches!(op.op, OpFunc::Fp32ToInt32) {
                        // The value-cast converts are bound in fp32 MODE ONLY
                        // (unary_parallel.ddl:47-50 — int32tofp32/fp32toint32 are
                        // `operation_bind([%type_fp32], ...)`; the DDL comment:
                        // "the conversion is carried out in fp32 mode, ICVT
                        // imm=8"). The input TENSOR may be fp16 (`%inptensor`
                        // carries both types — the vendor fp32toint32 fixture's
                        // inputLabeledDs is IEEE_FP32 but the bind admits fp16),
                        // but the OP's mode marker is IEEE_FP32, not the input's
                        // format. An SEN169_FP16 marker here is dxp's
                        // "Scheduler failed to find a suitable op mapping"
                        // (SchedulerStages.cpp:40) — no fp16-mode bind exists.
                        Df::Fp32.dataformat()
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
                // The scheduling row is OMITTED for the pair too — the `ComputeOp` doc's
                // validator law. Both vendor topk fixtures carry the explicit row
                // (`auxLoopName: ""`, `isAtMainLoop: 1`, `isAtTop: 1`, `level: 0`,
                // `opConsts: {}`), but the dev stack's new SDSC-JSON validator rejects all
                // five fields outright ("Unrecognized field") — MEASURED: the tiny26 bake's
                // dxp refused the pair's group with exactly that error on both members,
                // reproduced via manual dxp_standalone on the persisted stage dir (the same
                // validator class as the `location: "invalid"` rejection above; the fixtures
                // predate the validator). The pair's ONE surviving divergence is
                // `constantInfo_` ([`OpFunc::is_topk_pair`]) — and it has no NR refine for
                // an explicit `auxLoopName` to suppress.
                inputLabeledDs: input_refs,
                interimLabeledDs: vec![],
                outputLabeledDs: output_refs,
                // ── THE THIRD FIELD OF THE GATHER: the op names the INDEX side ──
                // Same `Tensor{i}-idx{i}` spelling the labeled-DS refs above are built with, and the
                // same operand position the two allocate nodes cross-link through, so all three
                // references to the index operand come from one number. The vendor emits exactly
                // `["Tensor1-idx1"]` here.
                //
                // Empty for every op that declares no gather, which is all of them today — so this
                // is byte-identical for every shipped bundle.
                indirectAccessIndexLabeledDs: op.indirect.map_or_else(Vec::new, |ia| {
                    vec![format!("Tensor{}-idx{}", ia.index, ia.index)]
                }),
            }];
            ops.extend(epilogue_compute_ops);
            ops
        },
    };

    let sdsc = SdscOp {
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
    };
    Ok((sdsc, arg_facts))
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
///
/// ⭐⭐⭐⭐⭐ THE DIMS CROSS AS THE COUNTS THEY ARE — [`RowCount`] / [`BlockCols`], the same typed
/// pair the reduce/pointwise mints take, so a wrong-kind count (a slot extent in a row slot, a lane
/// count in a width slot, or the two positions swapped) is a `cargo build` type error at the call
/// site, not a silently mis-shaped handle. This mint was the LAST place an attention intermediate's
/// shape lost its name on its way into the opspec: the counts arrived typed at the assembler doors
/// and were destructured to bare `u32` "for the `rb` shape slots only" — that exception is closed;
/// a caller now spends the typed count it holds or crosses through a door that names the quantity.
pub fn rb(name: &str, rows: RowCount, cols: BlockCols) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(
        name,
        StickLayout::row_blocked(rows.get() as usize, cols.get() as usize),
    )
    .expect("token-stream matmul operand is RowBlocked by construction")
}

/// [`rb`] at the operand's OWN df — the fp8 activation's handle. The kind is annotation-only to the
/// emit (the OpSpec carries the real df), but the WINDOW MINT's placement door prices bytes at the
/// handle's word length, so a buffer whose elements are 1 byte must not be annotated at 2: every
/// window over it would be overstated 2× and the last slot's legal convert would refuse. Same
/// infallibility contract and same typed-dims law as [`rb`] — only the df reaches the layout.
pub fn rb_df(name: &str, rows: RowCount, cols: BlockCols, df: Df) -> Stk<RowBlockedTag> {
    Stk::<RowBlockedTag>::new(
        name,
        StickLayout::row_blocked_df(rows.get() as usize, cols.get() as usize, df),
    )
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
    rb(
        name,
        RowCount::of_broadcast_row(),
        BlockCols::of_one_stick(Lanes::FP16),
    )
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
    y_broadcast: u32,
    col_offset: u32,
    col_window: Option<crate::addr::ColWindow>,
}
#[allow(dead_code)] // ctors are used as the pointwise builders migrate to handle-flow (rmsnorm first)
impl<'a, K: KindTag> In<'a, K> {
    pub fn full(h: &'a Stk<K>) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: false,
            y_broadcast: 0,
            col_offset: 0,
            col_window: None,
        }
    }
    pub fn scalar(h: &'a Stk<K>) -> Self {
        In {
            h,
            mb_broadcast: true,
            out_broadcast: true,
            y_broadcast: 0,
            col_offset: 0,
            col_window: None,
        }
    }
    pub fn col(h: &'a Stk<K>) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: true,
            y_broadcast: 0,
            col_offset: 0,
            col_window: None,
        }
    }
    pub fn sliced(h: &'a Stk<K>, col_offset: crate::addr::DevOff) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: false,
            y_broadcast: 0,
            col_offset: col_offset.into_raw_elems(),
            col_window: None,
        }
    }
    /// [`sliced`](Self::sliced) at a [`crate::addr::ColWindow`] — a whole-stick column window of a
    /// stick-major tensor with the op's own row count. The same offset `sliced` would carry; what it adds is
    /// the FACT that the window is a stick-major tensor of its own, so the door may present the op rank-2
    /// stick-major and the work-division law reaches it (#622).
    pub fn col_window(h: &'a Stk<K>, w: crate::addr::ColWindow) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: false,
            y_broadcast: 0,
            col_offset: w.dev().into_raw_elems(),
            col_window: Some(w),
        }
    }
    /// [`col`] (per-row scalar, out-broadcast over the stick) PLUS a base `col_offset` — the per-head-block
    /// variant: a `[mqu,stk]` reduced scalar at `h·mqu·stk`, broadcast over the `[mqu,mqp]` score block.
    pub fn col_at(h: &'a Stk<K>, col_offset: crate::addr::DevOff) -> Self {
        In {
            h,
            mb_broadcast: false,
            out_broadcast: true,
            y_broadcast: 0,
            col_offset: col_offset.into_raw_elems(),
            col_window: None,
        }
    }
    /// Broadcast a `[1,cols]` const/gain over rows (the rmsnorm gamma / a mb-broadcast operand).
    pub fn mb(h: &'a Stk<K>) -> Self {
        In {
            h,
            mb_broadcast: true,
            out_broadcast: false,
            y_broadcast: 0,
            col_offset: 0,
            col_window: None,
        }
    }
    /// [`mb`] PLUS a base `col_offset` — the per-block prefix-mask slice (`pmask[.., b*stick..]`,
    /// broadcast over the mq new rows, one stick-wide block at a time).
    pub fn mb_at(h: &'a Stk<K>, col_offset: crate::addr::DevOff) -> Self {
        In {
            h,
            mb_broadcast: true,
            out_broadcast: false,
            y_broadcast: 0,
            col_offset: col_offset.into_raw_elems(),
            col_window: None,
        }
    }
    /// ⭐⭐⭐ THE ROW-SPLIT (Y) BROADCAST — a `[mb, out]` source sprayed down the op's LEADING
    /// `y` dim, `y` sprays deep: the batched paged-decode kernel's J-slot merge (`merged` at
    /// `[64, 4]`, read by its consumer as the op `[mb = 64, out = 64, y = 4]`). The source is
    /// addressed as the `[y, mb, out]` frame's `[mb, out]` slice at `y`'s stride-0 — scale
    /// `y = RedNonStick`, the vendor's reduced-dim law — so flat element `(j, b, c)` reads the
    /// source's `(b, c)` exactly. The frame law
    /// ([`crate::sdsc_abstract::RegionFrame::split_rows`]) is checked by the consumer door
    /// before this operand is built: `rows == y·mb`, the source single-stick. `y`
    /// must be ≥ 2 — a spray of one is a reshape, not a broadcast, and is refused by the
    /// admission upstream.
    pub fn y_broadcast(h: &'a Stk<K>, y: u32) -> Self {
        assert!(
            y > 1,
            "In::y_broadcast: a spray count of {y} is not a broadcast — the leading dim must \
             spray a real count (the walk's Y admission refuses y == 1 upstream)"
        );
        In {
            h,
            mb_broadcast: false,
            out_broadcast: false,
            y_broadcast: y,
            col_offset: 0,
            col_window: None,
        }
    }
    /// [`y_broadcast`](Self::y_broadcast) AT A ROW CORNER — the per-trip window form of the
    /// row-split source: a `[mb, out]` window at row `r_start` of a taller region-stated view
    /// (the prefill mask, page-major). The corner is the window's own device offset in the
    /// parent's stick-blocked storage (`rc_of` at `(r_start, 0)` — the same typed corner law
    /// [`In::mb_at`](Self::mb_at) states for the Mb source's column window); the source is
    /// single-stick by the frame law, so one stick per row means the corner composes
    /// additively with the mb-row addressing and the spray reads exactly the window's rows.
    pub fn y_at(h: &'a Stk<K>, y: u32, off: crate::addr::DevOff) -> Self {
        assert!(
            y > 1,
            "In::y_at: a spray count of {y} is not a broadcast — the leading dim must \
             spray a real count (the walk's Y admission refuses y == 1 upstream)"
        );
        In {
            h,
            mb_broadcast: false,
            out_broadcast: false,
            y_broadcast: y,
            col_offset: off.into_raw_elems(),
            col_window: None,
        }
    }
    pub fn ew(&self) -> EwOperand<'a> {
        EwOperand {
            name: self.h.name(),
            layout: self.h.layout(),
            mb_broadcast: self.mb_broadcast,
            out_broadcast: self.out_broadcast,
            y_broadcast: self.y_broadcast,
            col_offset: self.col_offset,
            col_window: self.col_window,
            pooled: None,
            gathered: false,
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
        PointwiseFacts {
            op_name,
            op_func,
            rows,
            cols,
            inputs: &[a.ew()],
            o,
            out_offset: crate::addr::DevOff::ZERO.into(),
        },
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
        PointwiseFacts {
            op_name,
            op_func,
            rows,
            cols,
            inputs: &[a.ew(), b.ew()],
            o,
            out_offset: crate::addr::DevOff::ZERO.into(),
        },
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
        // fp32toint32: IEEE_FP32 value → SENUINT32 index. ⛔ THE INPUT TENSOR
        // MUST BE FP32 (stick 32, wordLength 4), NOT fp16: the DDL's slice
        // constraint (unary_parallel.ddl:59) requires the input and output
        // SLICE sizes equal, and `getStickSizes` measures the slice in
        // ELEMENTS (elemInSlice = stickSize/8) — an fp16 input's 64-elem stick
        // yields slice 8 against the u32 output's 32-elem stick slice 4, NEVER
        // equal, so the direct fp16→u32 convert is dxp-refused at bake
        // ("slice size does not match", ddl_conversion.cpp:2728). The `%type_fp16`
        // entry on `%inptensor`'s type list (:20) serves the OTHER ops bound to
        // that tensor; the cast's own bind (:50) is fp32-mode and its vendor
        // fixture (`test/sdsc_fp32toint32.json`) states the legal form: an
        // IEEE_FP32 wordLength-4 input. A caller with an fp16 source stages it
        // through `dl16tofp32` FIRST — the widening convert carries no slice
        // constraint of its own (quantization_double_pad.ddl:58).
        OpFunc::Fp32ToInt32 => (Df::Fp32, Df::SenUint32),
        _ => (Df::Fp16, Df::Fp16),
    }
}

/// A dtype-convert's opspec: both operands keep their stick on `out`, addressed RowBlocked
/// at rows>1 — the rank-3 `[mb,out,1]` staging every residual-style consumer reads.
///
/// (The FLAT per-operand arrangement — BOTH operands stick `mb`, the DDL-equality form the
/// deleted per-expert index-table chain needed — is gone with that chain; the probe record
/// for its two DDL laws survives in this file's history.)
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
    // ⛔ fp32toint32 NARROWS TOO (output stick 32 < input 64) — the same rank-3 requirement: a
    // rank-2 collapse drops the `y` dim dxp's buffer walk indexes (the `getBufferCapacityForNode`
    // crash recorded above). Kept a SEPARATE flag from `is_fp32_convert` because that one also
    // drives the RowBlocked arrangement pair, and the u32 output is a GATHER INDEX whose rows are
    // consecutive u32 entries — the arrangement the index's own declaration states, not a
    // residual's.
    let is_narrowing_convert = is_fp32_convert || matches!(op, OpFunc::Fp32ToInt32);
    let stickmajor = rows > 1 && !is_narrowing_convert;
    let time_tile = plan
        .time_tile_for_lx(|p, opt| pointwise_lx_resident(p, 2, opt, Df::Fp16), "out")
        .map_err(|e| e.0)?;
    // The fp32 convert (rank-3) must ADDRESS RowBlocked so it reads/writes the RowBlocked residual byte-for-
    // byte (a rank-3 flat read row-mixes). Force the per-tensor ElementArrangement on BOTH operands (the fp16
    // side reads RowBlocked x; the fp32 side writes RowBlocked x32). Only for multi-row fp32 converts.
    let rb = is_fp32_convert && rows > 1;
    let mk = |is_input: bool, name: &str, role: Role, df: Df| -> Result<AnyTensorArg, String> {
        // Both operands keep their stick on `out`; the pair addresses RowBlocked at rows>1
        // for the fp32 residual seam (`rb` below).
        let stick = "out";
        let rb_arg = rb;
        if stickmajor {
            Ok(AnyTensorArg::R2(
                TensorArg::<2>::new(
                    is_input,
                    name.to_string(),
                    role,
                    [Scale::Active, Scale::Active],
                    device_dims_r2,
                    ["mb", "out"],
                    stick,
                    Allocation::Hbm,
                )
                .map_err(|e| e.0)?
                .with_df(df)
                .with_row_blocked(rb_arg),
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
                    stick,
                    Allocation::Hbm,
                )
                .map_err(|e| e.0)?
                .with_df(df)
                .with_row_blocked(rb_arg),
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
        indirect: None,
        coordinate_mask: None,
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
    //
    // ⭐ AND NOW MINTED: the whole-tensor window `[rows, cols]` at 0 against BOTH doors, on each side
    // at its own df — the #731 law for this family (an op sweeping past its tensor is a defect in the
    // computation that produced the offset; it cannot compile). The output handle's df is the OUTPUT's
    // own (`rb_df`), so the placement door prices its bytes at one byte per fp8 element.
    mint_full_window(
        &format!("{op_name}: operand '{}'", in_.name()),
        in_,
        0,
        rows,
        cols,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_convert {op_name}: {e}"));
    mint_full_window(
        &format!("{op_name}: output '{}'", o.name()),
        o,
        0,
        rows,
        cols,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_convert {op_name}: {e}"));
    let op = convert_opspec(op_func_from_str(op_func), rows, cols, in_.name(), o.name())
        .unwrap_or_else(|e| panic!("assemble_convert {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_convert {op_name}: {e}"))
}

/// Assemble a dtype-convert op over a COLUMN WINDOW of its operands (see [`convert_opspec`]):
/// `in_off`/`o_off` are [`crate::addr::DevOff`]s — nest-derived column offsets built by
/// [`crate::addr::col_of`], so the `(c/lanes)·(rows·lanes)` group-corner term of the stick-major
/// layout is present by construction. ⛔ THE ONE CONVERT DOOR FOR EVERY `m`: at `m == 1` a
/// `col_of` corner degenerates to the flat stick-axis slice exactly (one row ⇒ no group-corner
/// term), so the per-pair quantize tail's decode arm routes HERE too and the raw-`u32` twin that
/// used to serve it (`assemble_convert_off`, flat offsets with no nest behind them) is deleted —
/// there is no convert door left that takes an offset no nest minted. Emit-time panic on a
/// malformed spec = a `cargo build` error (guard-every-crash), same as the other `assemble_*`.
#[allow(clippy::too_many_arguments)]
pub fn assemble_convert_col(
    op_name: &str,
    op_func: &str,
    rows: u32,
    cols: u32,
    in_: &Stk<RowBlockedTag>,
    in_off: crate::addr::DevOff,
    o: &Stk<RowBlockedTag>,
    o_off: crate::addr::DevOff,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // TYPED, same as `assemble_convert`: both operands are the RowBlocked token stream and the
    // handles carry only `.name()` — the offsets ride the args, not the handles.
    //
    // ⭐ AND MINTED at the corner, both doors, each side at its own df — the #731 law for the
    // per-slot convert: slot `s`'s window `[rows, cols]` at `s·inn`/`s·inn_pad` must fit the
    // FULL-buffer handles the caller trued (`cl_reach`/`k_pad`), which is exactly the arithmetic
    // `cl_reach = (pairs−1)·inn + inn_pad` states — the last slot's window ends at the buffer's own
    // end, so a caller that shrinks either side without re-deriving the other is a build failure.
    mint_full_window(
        &format!("{op_name}: operand '{}'", in_.name()),
        in_,
        in_off.into_raw_elems() as u64,
        rows,
        cols,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_convert_col {op_name}: {e}"));
    mint_full_window(
        &format!("{op_name}: output '{}'", o.name()),
        o,
        o_off.into_raw_elems() as u64,
        rows,
        cols,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_convert_col {op_name}: {e}"));
    let mut op = convert_opspec(op_func_from_str(op_func), rows, cols, in_.name(), o.name())
        .unwrap_or_else(|e| panic!("assemble_convert_col {op_name}: {e}"));
    for (arg, off) in op.args.iter_mut().zip([in_off, o_off]) {
        arg.set_offset_elems(off.into_raw_elems());
    }
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_convert_col {op_name}: {e}"))
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
        layout,
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
/// The `[8, 8]` inter-slice block that IS the transpose output's stick — `emit_sdsc`'s
/// `is_transpose_out` arm, `stickSize_: [8, 8]` in the vendor golden. The atomic unit a core must
/// own a WHOLE number of, on both axes.
const TRANSPOSE_BLOCK: u32 = 8;

/// ⛔⛔⛔ THE TRANSPOSE'S OWN CORE DIVISION — BOTH AXES, AT THE BLOCK — MEASURED AGAINST THE VENDOR
/// GOLDEN `sdsc_interslicetranspose.json`.
///
/// `distribute_cores` splits `mb` alone to the hilt because it was written for FLAT row-major
/// activations (its own doc), and a transpose is not one: its OUTPUT is a grid of 8×8 inter-slice
/// blocks, so a core handed a fraction of an 8-row block shares an output stick with its neighbour
/// and each writes a partial one — the shared-stick failure class on this tree's record. The golden
/// — the vendor scheduler's own emission for this exact op — splits BOTH axes:
/// `numWkSlicesPerDim_ {"out": 8, "mb": 4}` at `mb 384, out 3072` on 32 cores, i.e. per-core
/// (mb 96, out 384), both whole blocks.
///
/// THE LAW (legality): the `out` split divides the INPUT stick count (`out`/64 — whole 64-element
/// sticks per core, which is also a whole number of 8-wide block columns a fortiori), the `mb`
/// split divides the 8-block count (`mb`/[`TRANSPOSE_BLOCK`]), and the product stays within the
/// budget. Every legal division gives each core whole sticks on the read side and whole 8×8 blocks
/// on the write side — the invariant [`transpose_opspec`]'s guard nets over the built plan.
///
/// THE CHOICE, from the one fixture point plus the structure it exposes:
///   1. the most cores (`s_out · s_mb`) — idle cores are the waste the golden does not make;
///   2. on a tie, the division that BALANCES the per-core atomic units — `|blocks/s_mb −
///      sticks/s_out|`, compared exactly — so no core waits on a disproportionately large slice;
///   3. on a tie, the larger `out` split: at the golden's own shape the balanced pairs are
///      `(4, 8)` and `(8, 4)` and the fixture picks `(8, 4)`.
///
/// At the golden's (mb 384, out 3072) this yields `{"out": 8, "mb": 4}` — the fixture's division
/// verbatim, pinned by `the_fixtures_own_shape_gets_the_fixtures_division` in
/// [`mod transpose_tests`]. At a square `[128, 128]` (the padded token count the prefill-m>1 index
/// mint transposes) it yields `{"out": 2, "mb": 16}`; at `[64, 64]`, `{"mb": 8}`.
fn transpose_core_split(dims: &[ItDim], max_cores: u32) -> BTreeMap<&'static str, u32> {
    let dim = |name: &str| {
        dims.iter()
            .find(|d| d.name == name)
            .unwrap_or_else(|| panic!("transpose_core_split: no '{name}' dim in {dims:?}"))
    };
    let sticks = dim("out").size / FP16_ELEMS_PER_STICK;
    let blocks = dim("mb").size / TRANSPOSE_BLOCK;
    // The divisors that respect the law, capped at the budget: ≤ 32 per axis, so the pair
    // enumeration is ≤ 1024 candidates at ANY extent — expansion-cheap.
    let out_divs: Vec<u32> = (1..=max_cores.min(sticks.max(1)))
        .filter(|d| sticks.is_multiple_of(*d))
        .collect();
    let mb_divs: Vec<u32> = (1..=max_cores.min(blocks.max(1)))
        .filter(|d| blocks.is_multiple_of(*d))
        .collect();
    // Seed with the always-legal trivial division; every candidate below beats it on key (1).
    let mut best: (u32, u32) = (1, 1);
    for &s_out in &out_divs {
        for &s_mb in &mb_divs {
            let product = s_out * s_mb;
            if product > max_cores {
                continue;
            }
            let (b_out, b_mb) = best;
            if product != b_out * b_mb {
                if product > b_out * b_mb {
                    best = (s_out, s_mb);
                }
                continue;
            }
            // Key (2): the per-core atomic-unit balance, `|blocks/s_mb − sticks/s_out|`, as an
            // exact rational (a/b vs c/d ⟺ a·d vs c·b — no floats at expansion time).
            let bal = |o: u32, m: u32| {
                (
                    (blocks as u64 * o as u64).abs_diff(sticks as u64 * m as u64),
                    o as u64 * m as u64,
                )
            };
            let (an, ad) = bal(s_out, s_mb);
            let (bn, bd) = bal(b_out, b_mb);
            match (an * bd).cmp(&(bn * ad)) {
                std::cmp::Ordering::Less => best = (s_out, s_mb),
                std::cmp::Ordering::Greater => {}
                // Key (3): the golden's `(8, 4)` over the mirrored `(4, 8)`.
                std::cmp::Ordering::Equal if s_out > b_out => best = (s_out, s_mb),
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    let mut splits = BTreeMap::new();
    if best.1 > 1 {
        splits.insert("mb", best.1);
    }
    if best.0 > 1 {
        splits.insert("out", best.0);
    }
    splits
}

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
    // ⛔ THE SPLITTER IS THIS OP'S OWN, NOT `distribute_cores`: the transpose's output is a grid of
    // 8×8 inter-slice blocks, so the division must cut BOTH axes at the block — the law measured
    // against the vendor golden (see `transpose_core_split`).
    let tile_op = crate::ir::island::tile_op::TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands: 2 },
        dims: dims.clone(),
        df: Df::Fp16,
    };
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, transpose_core_split, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    // ⛔⛔⛔ THE PER-CORE SLICE MUST BE WHOLE 8×8 OUTPUT BLOCKS. `transpose_core_split` GUARANTEES
    // THAT BY CONSTRUCTION — THIS GUARD IS THE NET THAT KEEPS IT TRUE, NOT THE REFUSING AUTHORITY.
    //
    // The OUTPUT's stick is the 8×8 inter-slice block over (`out`, `mb`) — `emit_sdsc`'s
    // `is_transpose_out` arm, `stickSize_: [8, 8]`. A stick is the atomic unit a core moves, so if a
    // core's work slice ends part-way through a block, TWO cores own bytes of one output stick and
    // each writes a partial one. That is the failure class already on this tree's record (the second Kᵀ
    // stick written wrong; the shared-stick per-core init).
    //
    // HISTORY: this guard used to REFUSE, because the builder then split `mb` alone via
    // `distribute_cores` — per-core (mb 12, out 3072) at the golden's own (mb 384, out 3072), 12
    // not a whole block — and the division was deliberately not guessed from one fixture point.
    // The division is now measured against that fixture (`transpose_core_split`: both axes at the
    // block, reproducing the golden's `{"out": 8, "mb": 4}` verbatim), so the shapes the old guard
    // refused — the golden's own, and every stick-multiple `mb` below 8·MAX_CORES — bake. The guard
    // STAYS, stated over the built plan: a future splitter regression is a `cargo` error here,
    // never a descriptor whose cores collide.
    let (per_mb, per_out) = (plan.per_core_extent("mb"), plan.per_core_extent("out"));
    if !per_mb.is_multiple_of(TRANSPOSE_BLOCK) || !per_out.is_multiple_of(TRANSPOSE_BLOCK) {
        return Err(format!(
            "transpose [{}, {}]: the core division gives each core (mb {per_mb}, out {per_out}) but \
             the output's stick is the {TRANSPOSE_BLOCK}x{TRANSPOSE_BLOCK} inter-slice block over \
             (out, mb), so a per-core extent that is not a whole multiple of {TRANSPOSE_BLOCK} puts \
             two cores inside one output stick and each writes a partial one. `transpose_core_split` \
             builds only whole-block divisions measured against the golden \
             `sdsc_interslicetranspose.json`, so this refusal names a splitter regression, not a \
             shape the builder rejects.",
            mb_ext.elems(),
            out_ext.elems(),
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
        indirect: None,
        coordinate_mask: None,
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
/// producer that has its own op to blame, and this builder refuses in ways a caller cannot pre-check
/// by inspecting extents — the stick law (which the lowering does re-state, so it can name the padding
/// alternative) and the LX-fit refusal, which depends on the built plan's per-core extents. (The
/// core-division law is no longer among them: `transpose_core_split` builds only whole-block
/// divisions, so its guard is a net and not a reachable refusal.) So the fallible form exists and the
/// door uses it; `assemble_transpose` keeps its signature and its panic for every existing caller.
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
/// wire name, the unit, and the per-core division — which [`transpose_core_split`] now reproduces
/// verbatim at the fixture's own shape. Everything else here is a REFUSAL with a control.
#[cfg(test)]
mod transpose_tests {
    use super::*;
    use crate::ktir_node::{BufferId, KtirNode, Program};
    use ktir_core::arena::Arena;
    use ktir_core::attrkey::AttrKey;
    use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
    use ktir_core::irtype::IrType;
    use ktir_core::opkind::OpKind;

    /// The module's representative row extent. ANY 64-multiple works since the division became the
    /// fixture-measured law ([`transpose_core_split`]) — 256 is kept for continuity with the tests
    /// that were written while `distribute_cores` split `mb` alone and only `mb ≥ 8·MAX_CORES`
    /// landed on whole blocks.
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

    /// The op's ITERATION SPACE is the fixture's, at the fixture's own extents: `out_ 3072`, `mb_ 384`,
    /// `y_ 1`. Both halves of the fixture comparison now agree — the shape reaches the divider intact
    /// (asserted here) and the division is the fixture's (asserted by
    /// `the_fixtures_own_shape_gets_the_fixtures_division` below).
    #[test]
    fn the_iteration_space_matches_the_fixture_at_the_fixture_extents() {
        let spec = transpose_opspec(384, 3072, "t1", "t2")
            .expect("the fixture's own shape bakes under the fixture-measured division");
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

    /// ⛔⛔⛔ THE DIVISION IS THE FIXTURE'S, AT THE FIXTURE'S OWN SHAPE. At (`mb_ 384`, `out_ 3072`,
    /// 32 cores) the fixture splits BOTH axes — `{"out": 8, "mb": 4}`, per-core (mb 96, out 384),
    /// both whole 8-blocks — and [`transpose_core_split`] now reproduces exactly that. This is the
    /// test that used to pin the divergence as a REFUSAL (`distribute_cores` split `mb` alone to 32,
    /// per-core mb 12, not a whole block); the fixture was the oracle named in the refusal, and this
    /// is the acceptance it demanded.
    #[test]
    fn the_fixtures_own_shape_gets_the_fixtures_division() {
        let spec = transpose_opspec(384, 3072, "t1", "t2")
            .expect("the fixture-measured division accepts the fixture's own shape");
        assert_eq!(
            spec.iter.splits(),
            &BTreeMap::from([("mb", 4), ("out", 8)]),
            "the fixture's `numWkSlicesPerDim_ {{\"out\": 8, \"mb\": 4}}`, verbatim"
        );
        assert_eq!(spec.iter.per_core_extent("mb"), 96, "384 / 4");
        assert_eq!(spec.iter.per_core_extent("out"), 384, "3072 / 8");
        assert_eq!(
            spec.iter.cores_used().get(),
            32,
            "the fixture's `numCoresUsed_ 32`"
        );
    }

    /// THE CHOICE RULE IS NOT NUMEROLOGY — it is exercised on its tie-break keys, not just its
    /// output at the golden's shape. At `[256, 128]`: 16 atomic 8-blocks, 2 input sticks; the
    /// max-product candidates are `(out 1, mb 32)` and `(out 2, mb 16)`, both 32 cores, both
    /// perfectly balanced in atomic units — key (3) (the larger out split, the golden's `(8, 4)`
    /// over the mirrored `(4, 8)`) is what picks `(2, 16)`. And the square shapes land on whole
    /// blocks with cores left over: `[128, 128]` takes `{"out": 2, "mb": 16}` (per-core mb 8 =
    /// one whole block), and `[64, 64]` takes `{"mb": 8}` (out is a single stick; per-core
    /// mb 8).
    #[test]
    fn the_choice_rule_breaks_its_ties_on_balance_then_the_stick_axis_and_covers_the_square_shapes()
    {
        let square_128 = transpose_opspec(128, 128, "t1", "t2")
            .expect("a [128,128] transpose is a whole-block division");
        assert_eq!(
            square_128.iter.splits(),
            &BTreeMap::from([("mb", 16), ("out", 2)])
        );
        assert_eq!(
            square_128.iter.per_core_extent("mb"),
            8,
            "one whole 8-block"
        );
        assert_eq!(
            square_128.iter.per_core_extent("out"),
            64,
            "one whole input stick"
        );

        let square_64 =
            transpose_opspec(64, 64, "t1", "t2").expect("one stick wide, one block tall");
        assert_eq!(square_64.iter.splits(), &BTreeMap::from([("mb", 8)]));
        assert_eq!(square_64.iter.per_core_extent("mb"), 8);
        assert_eq!(square_64.iter.per_core_extent("out"), 64);

        let tie = transpose_opspec(256, 128, "t1", "t2").expect("both candidates are legal");
        assert_eq!(
            tie.iter.splits(),
            &BTreeMap::from([("mb", 16), ("out", 2)]),
            "the tie between (1, 32) and (2, 16) breaks to the larger out split"
        );
        assert_eq!(tie.iter.per_core_extent("mb"), 16);
        assert_eq!(tie.iter.per_core_extent("out"), 64);
    }

    /// EVERY shape this builder ACCEPTS has whole-block per-core extents. This is the invariant, and it
    /// is stated over the builder's output rather than over `transpose_core_split` — so it holds however
    /// the division is later fixed, and it cannot be satisfied by a divider that got lucky on one shape.
    /// Since the division became the fixture-measured law every 64-multiple shape in the sweep is
    /// accepted, so the sweep is now exhaustive over the stick-multiple regime, not a sample.
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
            step_inputs: Vec::new(),
            node_out_tid: None,
            minted: Default::default(),
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
    /// ⛔⛔⛔ IT IS A `ReStickifyOpHBM`, NOT AN `interslicetranspose_fp16`, AND THAT IS THE WHOLE POINT
    /// OF THE REROUTE. `OpFunc::Transpose` sticks its output on the 8×8 inter-slice block over
    /// (`out`, `mb`) — TWO dims — and the V3 translator accepts a dynamic mask over exactly ONE loop
    /// carrying exactly ONE dim (`SNComputeLowering.cpp:68,74`), so it could not be lowered at the
    /// default RCUDD1A arch at ANY shape. `OpFunc::Restickify` does the same transposition with a
    /// one-dim output stick. See [`restickify_transpose_opspec_2d`], which states the addressing
    /// argument, and `tests/zz_the_transpose_door_carries_a_two_dim_output_stick.rs`, which pins both doors.
    ///
    /// The two roles' dim orders are asserted SWAPPED as well as the stick being one-dim: equal orders
    /// would be a re-stick with no transposition, which carries the same one-dim stick and would pass
    /// the stick assertion alone.
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
        assert_eq!(
            d.primaryDsInfo_["OUTPUT"].stickDimOrder_,
            vec!["y"],
            "the OUTPUT stick must be ONE dim — the only property the V3 translator's limit is a \
             function of",
        );
        assert_eq!(d.primaryDsInfo_["OUTPUT"].stickSize_, vec![64]);
        assert_eq!(
            (
                d.primaryDsInfo_["INPUT"].layoutDimOrder_.clone(),
                d.primaryDsInfo_["OUTPUT"].layoutDimOrder_.clone(),
            ),
            (vec!["y", "out"], vec!["out", "y"]),
            "the two roles' dim orders must be SWAPPED — that swap IS the transposition, and equal \
             orders would be a re-stick that copies",
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

    /// ⛔ AND THE BUILDER'S OWN GUARD REACHES THE ENTRY POINT AS AN `Error`, NOT A PANIC. Every
    /// `assemble_*` here `panic!`s on a refused opspec, so a shape that passes the stick law and then
    /// trips a builder law would abort the build with a bare panic instead of a producer-reportable
    /// error; the door calls the fallible form for exactly that reason.
    ///
    /// ⚖️ THE LAW BEING TRIPPED CHANGED WITH THE REROUTE, AND THAT IS WHY THIS SHAPE DID. It used to be
    /// `transpose_opspec`'s 8×8 core-division guard, reached at `mb = 64` (per-core mb 2). The door now
    /// emits a `ReStickifyOpHBM`, which has no block law — `[64, 128]` is one of the shapes the reroute
    /// exists to make lower — so the remaining builder law is the LX budget: a restickify carries
    /// `time_tile: None` (its output stick IS the whole `y` extent, so there is no stick dim left to
    /// divide in time) and therefore cannot tile its way out of an over-budget tile. `[4096, 4096]` is
    /// that shape: `distribute_cores` spends all 32 cores on `y`, leaving each one (y 128, out 4096) =
    /// 2_097_152 B against the 1_677_721-B scratchpad.
    #[test]
    fn a_stick_aligned_shape_whose_builder_law_fails_is_an_error_not_a_panic() {
        let e = refusal(&transpose_program((4096, 4096), (4096, 4096)));
        assert!(
            e.contains("(y 128, out 4096)"),
            "the builder's own per-core tile reaches the caller, got: {e}"
        );
        assert!(
            e.contains("1677721") && e.contains("time_tile"),
            "and names the budget AND why this door cannot time-tile instead, got: {e}"
        );
    }

    /// ⭐ THE CONTROL FOR THE SHAPE THE REROUTE EXISTS FOR. `[64, 128]` used to be refused by the 8×8
    /// core-division guard (per-core mb 2) and is exactly the class `decoder_layer_one_flat` needs; it
    /// now emits, with the one-dim stick. Without this the test above would read as "big shapes are
    /// refused" and would not record that the small one was UNBLOCKED.
    #[test]
    fn the_shape_the_block_division_used_to_refuse_now_lowers() {
        let ops = lower_transpose(&transpose_program((64, 128), (128, 64)))
            .expect("[64, 128] is the shape class the reroute unblocks");
        let [op] = &ops[..] else {
            panic!("a transpose is ONE descriptor, got {}", ops.len());
        };
        let d = dsc(op, "transpose_o102");
        assert_eq!(d.primaryDsInfo_["OUTPUT"].stickDimOrder_, vec!["y"]);
        assert_eq!(d.computeOp_[0].opFuncName, "ReStickifyOpHBM");
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

    /// NO `(mb, out)` in `0..=512²` that is not a pair of whole 64-element sticks is EVER accepted —
    /// and (since the division became the fixture-measured law) EVERY whole-stick pair IS.
    ///
    /// ⭐ BOTH DIRECTIONS, NOW. The converse used to be FALSE — `distribute_cores` split `mb` alone,
    /// so most whole-stick pairs were refused by the core-division guard (the history
    /// [`transpose_opspec`] records) — and this sweep deliberately skipped them rather than assert
    /// our own division as truth. [`transpose_core_split`] makes acceptance a property of the SHAPE
    /// (a whole-stick pair always has a whole-block division, `(1, 1)` at worst), so the sweep
    /// asserts it: acceptance is exactly the stick law, nothing narrower. The whole-block invariant
    /// on the accepted side is stated separately, over the builder's output, by
    /// `every_accepted_transpose_gives_each_core_whole_8x8_blocks`.
    ///
    /// 512 spans the extents a real transpose has (a chunk width against a head dim or feature width)
    /// and covers 8 stick multiples per axis, so the `% 64 != 0`, `% 8 == 0 but % 64 != 0` and zero
    /// cases are all inside it.
    #[test]
    fn no_unaligned_extent_can_produce_a_transpose_descriptor() {
        let stk = <Fp16 as DataFormat>::ELEMS_PER_STICK;
        let (mut checked, mut sub_stick_but_8_whole, mut whole_stick) = (0u32, 0u32, 0u32);
        for mb in 0..=512u32 {
            for out in 0..=512u32 {
                if mb != 0 && out != 0 && mb % stk == 0 && out % stk == 0 {
                    whole_stick += 1;
                    assert!(
                        transpose_opspec(mb, out, "t1", "t2").is_ok(),
                        "[{mb}, {out}] is a pair of whole {stk}-element sticks, so the \
                         fixture-measured division must accept it"
                    );
                    continue;
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
        // And the converse side really ran: 8 stick multiples per axis, squared.
        assert_eq!(
            whole_stick, 64,
            "the whole-stick pairs the fixture-measured division must accept"
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

// ⛔ THE 3-D RESTICKIFY FAMILY IS GONE — `restickify_opspec`/`assemble_restickify` (the batched
// mb=nqh form), `restickify_opspec_off`/`assemble_restickify_off` (its per-head raw-`u32` twin)
// and `restickify_v_opspec_2d`/`assemble_restickify_v_2d` (the V-side re-stick, further below) had
// ZERO callers: the Kᵀ re-stick runs through [`assemble_restickify_kt_2d`] and the value bmm reads
// the natural V cache directly. The raw-`u32` `in_off`/`out_off` pairs were the last restickify
// doors with no nest behind them — a caller reintroducing this op writes it against a minted
// handle at the kt_2d door, so an offset no view computed cannot be stated.

// ════════════════════════════════════════════════════════════════════════════════════════════
// THE VENDOR TOP-K PAIR — `topk.ddl`'s `topkindex`/`topkvalue`, the selection primitive that
// replaces the router's argsort chain (wall 11: the chain's dup ranks summed expert ids past the
// expert bank; the pair's counter-built indices are unique by construction).
// ════════════════════════════════════════════════════════════════════════════════════════════

/// The three extents a top-k is over, each named for its ROLE so the triple cannot be handed over
/// swapped (the [`assemble_restickify_kt_2d`] slots/feats law). See [`topk_opspec`] for what each
/// role means in the descriptor.
#[derive(Clone, Copy, Debug)]
pub struct TopKGeom {
    /// The CANDIDATE count — the reduction dim `in`: the router's expert width `W`, scanned whole
    /// by every core of the ring.
    pub candidates: u32,
    /// The SLOT count — the k dim `out`: how many (value, index) pairs the pair emits per token,
    /// DESCENDING (slot 0 = the max). The pair itself accepts any `slots ≥ 1` (the fixtures' own
    /// k is 8); the landing chain runs it at 64 so the rerouted-transpose door can re-orient the
    /// outputs token-major (its stick law needs a 64-multiple).
    pub slots: u32,
    /// The padded TOKEN count — the stick dim `y`, a whole 64-multiple: the one axis the pair's
    /// input and output share, so it is the only axis allowed on the stick.
    pub tokens: u32,
}

/// ⭐⭐⭐ THE PAIR'S OWN CORE DIVISION — `distribute_cores` cannot build it, because the top-k is
/// neither a pointwise nor a plain reduce: its k dim is a RING (`ddl.core_to_core_communication(%k)`,
/// `topk.ddl:72` — the displaced value ripples core-to-core through the insertion sort), and the
/// reduction dim must NEVER split (the fixture's own `numWkSlicesPerDim_` carries `in: 1`; splitting
/// the scan would need the partial-reduce collision machinery the pair's fixtures never exercise).
///
/// THE LAW, from the two fixtures plus the structure they expose:
///   1. the ring = the largest divisor of `slots` that is ≤ 8 — BOTH fixtures carry
///      `numWkSlicesPerDim_ {"out": 8}`, the only ring size any accepted top-k descriptor
///      demonstrates, and the size that keeps the per-core `k_loop` ripple (the insertion sort
///      over the core's k-vectorized internal tensors — `current_max`/`current_idx` are
///      `internal_tensor`s of the OUTPUT's layout, so N slots per core is the vectorized ripple)
///      at the depth the fixtures themselves run;
///   2. then fill with the token-stick instances (`y` split at whole sticks) up to the budget —
///      the fixtures' `mb: 2, i: 2` played exactly this role (instance parallelism outside the
///      ring: the `core_to_core_communication` is scoped to the k dim, so cores in different
///      `y` slices never communicate and each runs its own ring);
///   3. NEVER `in` — the scan axis.
///
/// At the fixtures' own geometry (candidates 64, slots 8, tokens 64 = one stick) this yields
/// `{"out": 8}` against their `{"in": 1, "out": 8, "mb": 2, "y": 1, "i": 2}`: the ring matches
/// verbatim, and the `mb`/`i` instance splits are the documented divergence — this builder has no
/// `mb`/`i` dims, because its `y` is already the multi-stick token axis those dims instanced.
/// Pinned by `the_split_is_the_ring_first_then_token_sticks` in
/// `tests/zz_the_topk_pair_matches_the_vendor_fixture.rs`.
fn topk_core_split(dims: &[ItDim], max_cores: u32) -> BTreeMap<&'static str, u32> {
    let dim = |name: &str| {
        dims.iter()
            .find(|d| d.name == name)
            .unwrap_or_else(|| panic!("topk_core_split: no '{name}' dim in {dims:?}"))
    };
    let slots = dim("out").size;
    let sticks = dim("y").size / FP16_ELEMS_PER_STICK;
    // (1) the ring: largest divisor of the slot count that is ≤ 8 (and inside the budget).
    let ring = (1..=max_cores.min(8))
        .filter(|d| slots.is_multiple_of(*d))
        .max()
        .unwrap_or(1);
    // (2) the token instances: the LARGEST stick-count divisor that fits beside the ring — not a
    // power-of-two climb, because stick counts are often odd (a 320-token prefill is 5 sticks, and
    // `y: 3` at 192 tokens is as legal as `y: 2` at 128).
    let y_split = (1..=sticks.min(max_cores))
        .filter(|d| sticks.is_multiple_of(*d) && ring * d <= max_cores)
        .max()
        .unwrap_or(1);
    let mut splits = BTreeMap::new();
    if y_split > 1 {
        splits.insert("y", y_split);
    }
    if ring > 1 {
        splits.insert("out", ring);
    }
    splits
}

/// ⭐⭐⭐ THE VENDOR TOP-K PAIR AS A DESCRIPTOR — ONE member (`OpFunc::TopKIndex` emits the
/// `[slots, tokens]` expert ids, `OpFunc::TopKValue` the values at those ids) over the TRANSPOSED
/// scores, replacing the argsort + rank-inversion chains the router door used to emit.
///
/// THE FORM, and why every part of it is forced:
///   • INPUT `[in, y]` = `[candidates, tokens]`, sticked on `y` (dim 1 = LAST, dxp-friendly) ⇒
///     device walk `[tokens/64, candidates, 64]` — EXACTLY the Kᵀ restickify door's output bytes
///     (`[cap/64, hd, 64]` at `candidates = hd`), so the chain scores → Kᵀ → pair is a
///     byte-identical seam at ANY token count.
///   • OUTPUT `[out, y]` = `[slots, tokens]`, sticked on `y` ⇒ walk `[tokens/64, slots, 64]` —
///     the FLAT restickify door's input contract and the rerouted-transpose door's input contract
///     are both EXACTLY this form, so both landings are one door each.
///   • THE STICK LAW (torch_spyre `_check_stick_topk`, `op_spec_validation.py:743`): neither the
///     reduction dim (`in`) nor the k dim (`out`) may ride the stick; any SURVIVING dim may. The
///     stick must therefore be the token axis — the one dim input and output share — and the dims
///     table below is that law made structural.
///
/// ⭐ WHY THE NAMES DON'T MATTER TO DDC — the dim↔DDL-symbol binding is a CONSTRAINT SOLVER, not a
/// naming convention (`DdlConversion::matchDdl2Dsc`, `ddl_conversion.cpp:2360-2460`): the
/// order-fixed slice/stick layouts pin `outer#0` to the SDSC's stick dim, the flexible global
/// layouts restrict the reduce symbols to the INPUT-only dims and `%k` to the OUTPUT-only dim, and
/// a backtracking search assigns the rest. With input `[in, y]`, output `[out, y]`, stick `y`, the
/// solver derives reduce=`in`, k=`out`, `outer#0`=`y` — the exact roles `topk.ddl`'s dataflow needs
/// (the element loop over the reduce dims, the ring over `%k`). The fixtures' own 4-dim
/// `["in","i","y","mb"]` form is the same solver's output at a geometry with two extra instance
/// dims; this 3-dim form is the minimal one.
///
/// The fixtures (`ddc/ddl_templates/test/sdsc_topKIndex.json`, `sdsc_topKValue.json`) are
/// STRUCTURAL oracles, not byte parity: they are hand-built (`hbmSize_ 256/32` against volumes of
/// 16384/2048 elements), carry `stickRepl_` and `pdsRelation_` this crate deliberately does not
/// emit, and instance their volume on `mb`/`i` dims this form does not have. What they pin — and
/// what `tests/zz_the_topk_pair_matches_the_vendor_fixture.rs` asserts — is the layout contract,
/// the wire names, the unit, the ring-first division, and the computeOp row
/// ([`OpFunc::is_topk_pair`]).
fn topk_opspec(
    member: OpFunc,
    geom: TopKGeom,
    in_name: &str,
    o_name: &str,
) -> Result<OpSpec, String> {
    if !member.is_topk_pair() {
        return Err(format!(
            "topk {:?}: {member:?} is not a member of the vendor top-k pair — this builder emits \
             the topk.ddl form (input [in, y] and output [out, y], both sticked on y, the SFP ring \
             over out), which only topkindex/topkvalue bind. Route plain reductions through the \
             reduce builder.",
            geom
        ));
    }
    let TopKGeom {
        candidates: in_elems,
        slots: out_elems,
        tokens: y_elems,
    } = geom;
    if in_elems == 0 || out_elems == 0 {
        return Err(format!(
            "topk [in {in_elems}, out {out_elems}, y {y_elems}]: a top-k over zero candidates or \
             into zero slots is empty — the vendor pair has no degenerate mode (its fixtures scan \
             in=64 into out=8), and an empty descriptor would be a silent no-op instead of a refusal."
        ));
    }
    // The whole-stick half of the stick law: the token axis must be a whole 64-multiple (the
    // fixtures' y=64; the pad-the-row-count law the landing chain already runs).
    let y_ext = StickExtent::<Fp16>::new(y_elems)?;
    // THE STICK LAW, MADE STRUCTURAL: `in` is the reduction dim, `out` is the k dim, NEITHER is
    // the stick, and `y` — the one dim both args share — is. `in` carries `is_reduction` so the
    // tiler's reduction ladder knows the scan axis (and never proposes splitting it).
    let dims = vec![
        ItDim {
            name: "in",
            size: in_elems,
            is_reduction: true,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "out",
            size: out_elems,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        ItDim {
            name: "y",
            size: y_ext.elems(),
            is_reduction: false,
            is_stick: true,
            df: Df::Fp16,
        },
    ];
    // LOWER to TileIR: the resident set is the 1 input + 1 output tile (n_operands = 2), tiled
    // over the `y` token dim if it overflows LX — every y tile is an independent top-k over its
    // own tokens, so time-tiling is semantically free. ⛔ THE SPLITTER IS THE PAIR'S OWN
    // (`topk_core_split`), NOT `distribute_cores`: the ring-over-k law and the never-split-the-
    // scan law are both outside `distribute_cores`' vocabulary.
    let tile_op = crate::ir::island::tile_op::TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands: 2 },
        dims: dims.clone(),
        df: Df::Fp16,
    };
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, topk_core_split, "y")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    // ⛔ THE MIN-DATASTAGE FOOTPRINT GUARD — a runtime allocation refusal is a
    // build error (the user's law). The emitter pins the pair's datastage
    // minimum on `out` at ONE slot (the vendor fixtures' own per-core geometry;
    // see the datastage law in `emit_sdsc_tiled` — a slice-stated minimum made
    // the DDC refuse the whole 26b decode bundle at bake, card-mutation-proven),
    // so the one term of the minimum LX charge this mint can still let grow is
    // the never-split INPUT scan: `in` is the reduction dim (`topk_core_split`
    // never proposes splitting it) and the y stick is the streaming unit,
    // staged double-buffered — `2 buffers · in · y_per_core · 2 B`. A scan that
    // alone overflows the vendor scheduler's post-reserveFrontendLx pool
    // ([`crate::superdsc_opspec::VENDOR_LX_SCHED_BYTES`]) would make the DDC
    // refuse the bundle at bake with nothing local to blame ("Cannot allocate
    // even the smallest size", ddcv1.cpp:1351) — refuse it HERE, where the
    // geometry names itself.
    let y_per_core = y_ext.elems() / plan.split_of("y").max(1);
    let scan_bytes = 2 * in_elems as u64 * y_per_core as u64 * 2;
    if scan_bytes > crate::superdsc_opspec::VENDOR_LX_SCHED_BYTES {
        return Err(format!(
            "topk [in {in_elems}, out {out_elems}, y {}]: the never-split input scan \
             {in_elems} × {y_per_core} tokens, double-buffered, is {scan_bytes} B — past the \
             vendor scheduler's post-reserveFrontendLx pool of {} B, so the DDC would refuse \
             this bundle at its minimum-datastage check (allocAllMem, ddcv1.cpp:1351). Split \
             the candidate axis at the producer or shrink the token stick",
            y_ext.elems(),
            crate::superdsc_opspec::VENDOR_LX_SCHED_BYTES
        ));
    }
    let input = TensorArg::<2>::new(
        true,
        in_name.to_string(),
        Role::Input,
        [Scale::Active, Scale::Active],
        plan.iter_syms(["in", "y"]),
        ["in", "y"],
        "y",
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?;
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
    .map_err(|e| e.0)?;
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    Ok(OpSpec {
        op: member,
        // A top-k IS a reduction (the `in` scan); `emit_sdsc_tiled`'s seeded-reduce arm cannot
        // fire on it because the pair's `ddl_input_arity` is `Some(1)` — the unseeded, internal-
        // state contract the registry row pins.
        is_reduction: true,
        iter: plan,
        args: vec![AnyTensorArg::R2(input), AnyTensorArg::R2(output)],
        op_info: OpInfo::None,
        tiled_symbols,
        time_tile,
        indirect: None,
        coordinate_mask: None,
    })
}

/// [`topk_opspec`] assembled — the `#[forward]` form, for callers inside an expansion where a
/// refusal IS the build error. Door arms that can name their own blame use [`try_assemble_topk`].
pub fn assemble_topk(
    op_name: &str,
    member: OpFunc,
    geom: TopKGeom,
    in_name: &str,
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    try_assemble_topk(op_name, member, geom, in_name, o_name, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_topk {op_name}: {e}"))
}

/// [`assemble_topk`] AS A `Result` — the form a door arm needs, so a refusal names the producer's
/// own op instead of panicking inside the expansion (the `try_assemble_transpose` law).
pub fn try_assemble_topk(
    op_name: &str,
    member: OpFunc,
    geom: TopKGeom,
    in_name: &str,
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, String> {
    let op = topk_opspec(member, geom, in_name, o_name)?;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout).map_err(|e| e.0)
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
        indirect: None,
        coordinate_mask: None,
    })
}

/// Assemble ONE per-head Kᵀ restickify (natural `[slots,feats]` → slot-sticked `[feats,slots]`).
/// See [`restickify_kt_opspec_2d`]. The tile's two extents arrive as
/// [`KtTileSlots`](crate::sdsc_abstract::KtTileSlots)/[`KtTileFeats`](crate::sdsc_abstract::KtTileFeats) —
/// each door names which quantity fills it, so the pair cannot be handed over swapped.
/// `in_off`/`out_off` are the per-head element offsets into the natural cache (`head·cap·hd`) and
/// the Kᵀ scratch (`head·hd·cap`).
///
/// ⭐ AND NOW MINTED, BOTH SIDES. The op re-lays one tile — `[slots, feats]` read slab-major,
/// `[feats, slots]` written slot-major — which is ONE CONTIGUOUS run of `slots·feats` elements at
/// each corner, so the window the mint bounds is that flat run against the handle's declared
/// buffer. The handles replace the names: the caller states what each tensor IS (`[mq_pad,
/// nkvh·hd]` for the roped new-K; the plane totals for the resident caches), and an offset whose
/// run would cross the tensor's end — the Kᵀ restickify writing past the `[nkvh, hd, mq_pad]`
/// scratch into the next intermediate's bytes, read back as keys with a clean bake — is a `cargo
/// build` failure at this door instead.
#[allow(clippy::too_many_arguments)]
pub fn assemble_restickify_kt_2d(
    op_name: &str,
    slots: crate::sdsc_abstract::KtTileSlots,
    feats: crate::sdsc_abstract::KtTileFeats,
    in_: &Stk<RowBlockedTag>,
    in_off: crate::addr::DevOff,
    o: &Stk<RowBlockedTag>,
    out_off: crate::addr::DevOff,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let in_off = in_off.into_raw_elems();
    let out_off = out_off.into_raw_elems();
    // THE TILE'S FLAT RUN — the element range each side touches. Not `[slots, feats]` as a strided
    // window: the restickify reads its source sticked on the FEATURE axis and writes its output
    // sticked on the SLOT axis, and both passes walk the SAME contiguous `slots·feats` elements
    // from their corner (the addressing doc in `restickify_kt_opspec_2d`). One flat window states
    // exactly the bytes either side can reach.
    let run = slots.extent().saturating_mul(feats.extent());
    mint_full_window(
        &format!("{op_name}: operand '{}'", in_.name()),
        in_,
        in_off as u64,
        1,
        run,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_restickify_kt_2d {op_name}: {e}"));
    mint_full_window(
        &format!("{op_name}: output '{}'", o.name()),
        o,
        out_off as u64,
        1,
        run,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_restickify_kt_2d {op_name}: {e}"));

    let op = restickify_kt_opspec_2d(
        slots.extent(),
        feats.extent(),
        in_.name(),
        o.name(),
        in_off,
        out_off,
    )
    .unwrap_or_else(|e| panic!("assemble_restickify_kt_2d {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_restickify_kt_2d {op_name}: {e}"))
}

/// A STANDALONE `[rows, cols] → [cols, rows]` fp16 TRANSPOSE ON THE RESTICKIFY DOOR — the form
/// [`lower_ktir_to_superdsc::transpose`] emits for a producer's `linalg.transpose`.
///
/// ⛔⛔⛔ WHY THIS DOOR AND NOT [`transpose_opspec`], WHICH IS NAMED FOR THE JOB. `OpFunc::Transpose`
/// (`interslicetranspose_fp16`) sticks its OUTPUT on the 8×8 inter-slice block over (`out`, `mb`) —
/// `emit_sdsc`'s `is_transpose_out` arm, `stickDimOrder_ ["out","mb"]`. A TWO-DIM output stick is what
/// the V3 translator cannot lower: `Ddc::transformForInterSliceRestickify`
/// (`ddc/ddc_transformation.cpp:2398`, unconditional from `Ddc::run_v1`) fires on any PT node whose
/// input and output stick orders differ and writes one mask offset per dim of the innermost loop
/// carrying `outputStickDimOrder[0]`, which for a 2-D (`mb`, `out`) tile is the FUSED
/// `loop_dsX_dsY_out_mb` — two dims — while `SNComputeLowering::constructDynamicMasking` accepts
/// exactly one loop carrying exactly one dim (`SNComputeLowering.cpp:68,74`). Measured through
/// `dxp_standalone`: `OpFunc::Transpose` refuses at EVERY shape, including shapes whose core division
/// splits both axes like the golden `sdsc_interslicetranspose.json`, which is what ruled the core
/// division out as the cause. `OpFunc::Restickify` sticks its output on the single `y` axis, DDC lands
/// on a one-dim loop, and the same masked-MACC path is accepted — exit 0.
///
/// ⭐ AND THE ADDRESSING IS THE LADDER'S OWN CONVENTION, which is why the substitution is sound and not
/// merely translatable. [`restickify_kt_opspec_2d`] reads its input as `[y=rows, out=cols]` sticked on
/// the LAST dim ⇒ `dev_off([rows,cols],1,·)` = `[cols/64, rows, 64]`, and writes `[out=cols, y=rows]`
/// sticked on ITS last dim ⇒ `[rows/64, cols, 64]`. That stick-major form IS the fp16 device layout
/// every operand on this path already carries, so both sides read and write bytes the rest of the
/// bundle addresses identically — with the axes swapped, which is the transposition.
///
/// The extents are plain `u32` here rather than the paged-KV
/// [`KtTileSlots`](crate::sdsc_abstract::KtTileSlots)/[`KtTileFeats`](crate::sdsc_abstract::KtTileFeats)
/// newtypes [`assemble_restickify_kt_2d`] takes: this door serves an arbitrary producer-stated shape,
/// which is exactly the thing those newtypes exist to forbid for the cache. Offsets are zero — a
/// `linalg.transpose` names whole buffers (its caller refuses a window).
fn restickify_transpose_opspec_2d(
    rows: u32,
    cols: u32,
    in_name: &str,
    o_name: &str,
) -> Result<OpSpec, String> {
    // The Kᵀ builder IS this descriptor: `cap`→`y` is the row axis, `hd`→`out` the column axis, and
    // both stick laws are checked there. Sharing it means the two doors cannot drift apart.
    let op = restickify_kt_opspec_2d(rows, cols, in_name, o_name, 0, 0)?;
    // ⛔⛔⛔ AND IT MUST FAIL CLOSED ON LX, BECAUSE THIS DOOR CANNOT TIME-TILE ITS WAY OUT.
    //
    // `transpose_opspec` reached `TileOp::tile`, which splits the `out` stick dim in TIME when the
    // per-core tile overflows the scratchpad. A restickify has no such axis to give: its output stick
    // IS the whole `y` extent, so dividing `y` in time would split an output stick, and
    // `restickify_kt_opspec_2d` therefore carries `time_tile: None`. So the reroute genuinely narrows
    // what this door accepts, and the narrowing has to be a NAMED REFUSAL — otherwise a big transpose
    // becomes a descriptor that addresses past LX and dies on-card as DtException 1535.
    //
    // The resident set is the 1 input + 1 output per-core tile, which for a relayout are the same
    // extents on both sides (the swap is in the addressing, not the size).
    let (per_y, per_out) = (op.iter.per_core_extent("y"), op.iter.per_core_extent("out"));
    let resident = 2 * u64::from(per_y) * u64::from(per_out) * u64::from(Fp16::WORD_LENGTH);
    if resident > USABLE_LX_BYTES {
        return Err(format!(
            "restickify transpose [{rows}, {cols}]: the per-core tile is (y {per_y}, out {per_out}) \
             on {} cores, so the resident input+output set is {resident} B against the \
             {USABLE_LX_BYTES}-B LX scratchpad. This door carries `time_tile: None` and cannot be \
             time-tiled instead — a restickify's output stick IS the whole `y` extent, so there is no \
             stick dim left to divide in time without splitting an output stick. Transpose a narrower \
             column block and loop, which keeps each descriptor's resident set inside LX.",
            op.iter.cores_used().get(),
        ));
    }
    Ok(op)
}

/// [`restickify_transpose_opspec_2d`] assembled — the fallible form, because this is reached by a
/// third-party KTIR producer that has its own op to blame (see [`try_assemble_transpose`]).
///
/// ⭐ MINTED, BOTH SIDES, at the whole-buffer window: the caller's own law refuses a windowed
/// transpose (whole views, both sides — see `lower_ktir_to_superdsc::transpose`), so the flat
/// `[1, rows·cols]` run at 0 IS the op's whole access, and the mint's question is whether each
/// handle's declared buffer — and, when a layout is at hand, its PLACEMENT — actually holds that
/// many elements. The placement half is the new fact: a view can be whole while the reservation
/// behind its name is smaller ("512 reserved against 2048 declared"), which is a clean-bake
/// overrun into the next tensor's bytes, and it refuses here instead.
pub fn try_assemble_restickify_transpose_2d(
    op_name: &str,
    rows: u32,
    cols: u32,
    in_: &Stk<RowBlockedTag>,
    o: &Stk<RowBlockedTag>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, String> {
    mint_full_window(
        &format!("{op_name}: operand '{}'", in_.name()),
        in_,
        0,
        rows,
        cols,
        layout,
    )?;
    mint_full_window(
        &format!("{op_name}: output '{}'", o.name()),
        o,
        0,
        cols,
        rows,
        layout,
    )?;
    let op = restickify_transpose_opspec_2d(rows, cols, in_.name(), o.name())?;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout).map_err(|e| e.0)
}

// ⛔ THE FLAT RESTICKIFY IS GONE — `restickify_flat_opspec_2d` /
// `try_assemble_restickify_flat_2d` (the second restickify of the per-expert index-table
// chain: the K-form's m_pad-stick re-laid flat W-stick for the `dl16tofp32` leg's
// shared-`%stick_dim` bind) existed only for the hand-written per-op MoE card lowering,
// which is deleted: the MoE lowers only from `kernels/moe.py` through KTIR. The probe
// record (the same-dim "silent" control refusing at `ddl_conversion.cpp:2539`, the
// two-distinct-stick-dims law the restackify DDL demands) survives in this file's
// history.
//
// The V-side re-stick (`restickify_v_opspec_2d`/`assemble_restickify_v_2d`, "PATH B") is deleted
// with them — see the tombstone at the 3-D family above.

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
        "fp32toint32" => OpFunc::Fp32ToInt32,
        "gelu" => OpFunc::Gelu,
        "mish" => OpFunc::Mish,
        "tanh" => OpFunc::Tanh,
        "sum" => OpFunc::Sum,
        "max" => OpFunc::Max,
        "mean" => OpFunc::Mean,
        // ⭐ WALL 12 — the nonstick twins of the seeded reduce (`summeanmaxexx2.ddl:39/:45`,
        // exact `opFuncName` matches; matchDdl2Dsc compares the string EXACTLY,
        // ddl_conversion.cpp:2117-2122, so these spellings are load-bearing wire names, never
        // renamed). These arms MUST land before any assembler touches the names — this function
        // panics on an unknown name rather than guessing, and the door derives the names from the
        // IR's own combiner (`ReduceKind::nonstick_op_func`).
        "sumnonstick" => OpFunc::SumNonStick,
        "maxnonstick" => OpFunc::MaxNonStick,
        // The ONLY runtime-supported spelling: plain `exx2` maps to `OpFuncs::EXX2` →
        // exx2_fwd.smc (updatelibrary.cpp). The `exx2_zeromean` spelling ddc can rewrite to
        // is a distinct enum with NO smc-table entry — see `exx2_consts`' doc.
        "exx2" => OpFunc::Exx2,
        "identity" => OpFunc::Identity,
        "maximum" => OpFunc::Maximum,
        "minimum" => OpFunc::Minimum,
        "clip" => OpFunc::Clip,
        "qfp8ch" => OpFunc::Qfp8ch,
        "quantscalepertokenfp8" => OpFunc::QuantScalePerTokenFp8,
        "lesserthan" => OpFunc::LesserThan,
        "equal" => OpFunc::Equal,
        "notequal" => OpFunc::NotEqual,
        "where3" => OpFunc::Where3,
        "topkindex" => OpFunc::TopKIndex,
        "topkvalue" => OpFunc::TopKValue,
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

/// `constantInfo_.data_` as the NEW (post-2026-09) dxp contract demands it: a FoldManager
/// OBJECT, not a bare array. `sdscJsonValidate.cpp::validate_constant_info` admits only
/// {name_, dataFormat_, data_} (additionalProperties-style rejection killed `allocations_`)
/// and `validate_fold_manager` requires `data_` to be an object with `dim_prop_func` /
/// `dim_prop_attr` arrays. `dsc2.cpp` imports it as `data_.importFromJson(data_, sdscFoldProps)`
/// where the deque is the FULL fold space — `[coreFoldProp_, coreletFoldProp_, *sdscFoldProps_]`
/// (dsc2.cpp:1012-1021) — so `dim_prop_attr` MUST have one entry per dim with factors matching
/// the same sdsc's own fold props (importFromJson checks each `props[i]->getSize() == factor_`).
/// The value lands at coordinate "[0, 0, 0]" and is a DECIMAL STRING inside a one-element
/// array: the dtype is `FoldManager<std::vector<int64_t>>`, so the array imports element-wise,
/// and json11's bare-number path truncates at int width (import_utils.h's own comment:
/// 0xFF7FFFFF came back as -2147483648) — which the SFP table's u32 entries and any negative
/// fp32 bit pattern would hit. Shape matches the golden `ddc/ddl_templates/test/sdsc_mean.json`
/// (`"data_": {"[0, 0, 0]": ["65534"]}`).
fn fold_manager_const(value: u32, folds: &SdscFoldSet) -> serde_json::Value {
    serde_json::json!({
        "dim_prop_func": [{"Const": {}}, {"Const": {}}, {"Const": {}}],
        "dim_prop_attr": [
            {"factor_": folds.core_fold() as i64, "label_": "core"},
            {"factor_": folds.corelet_fold() as i64, "label_": "corelet"},
            {"factor_": folds.time_fold() as i64, "label_": "time"}
        ],
        "data_": {"[0, 0, 0]": [value.to_string()]}
    })
}

fn scaling_factor_const(packed: u32, folds: &SdscFoldSet) -> serde_json::Value {
    serde_json::json!({
        "0": {
            "dataFormat_": "SEN169_FP16",
            "name_": "scaling_factor",
            "data_": fold_manager_const(packed, folds)
        }
    })
}

/// The `exx2` reduce's TWO named external constants (`exx2scale` + `useZeroMean`), same
/// `constantInfo_` shape as [`scaling_factor_const`]. `bits` is the single SEN169 word of 1/N
/// (the reduce's divisor); `useZeroMean` is the RAW integer 0 — the vendor's own production
/// spelling (their `lower_exx2` passes `False`, lowering.py:667) — NOT a sen169-encoded value.
/// ⛔ IT MUST STAY 0: ddcv1.cpp:2072 rewrites the opFuncName to `exx2_zeromean` when the const
/// compares `== 1`, and that REWRITTEN name is a distinct `OpFuncs` enum value
/// (arch_enums.cpp:395) with NO entry in the runtime's op→smc table (updatelibrary.cpp has
/// `EXX2` only, resolved via `libMapping.at()` — a throw on miss). The ZeroMean path is
/// compile-tested by the vendor's own fixtures but never run: their models all emit 0. The
/// first card run of this chain with 1 produced degenerate output with a green bake — the
/// exact silent-failure shape a table miss leaves behind. See
/// [`crate::superdsc_opspec::OpInfo::Exx2Scale`].
fn exx2_consts(bits: u32, folds: &SdscFoldSet) -> serde_json::Value {
    serde_json::json!({
        "0": {
            "dataFormat_": "SEN169_FP16",
            "name_": "exx2scale",
            "data_": fold_manager_const(bits, folds)
        },
        "1": {
            "dataFormat_": "SEN169_FP16",
            "name_": "useZeroMean",
            "data_": fold_manager_const(0, folds)
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
pub fn scaling_factor_const_fp32(bits: u32, folds: &SdscFoldSet) -> serde_json::Value {
    serde_json::json!({
        "0": {
            "dataFormat_": "IEEE_FP32",
            "name_": "scaling_factor",
            "data_": fold_manager_const(bits, folds)
        }
    })
}

fn sfp_constant_table(folds: &SdscFoldSet) -> serde_json::Value {
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
                "data_": fold_manager_const(*val, folds)
            }),
        );
    }
    serde_json::Value::Object(map)
}

/// Guard: a pointwise op's stick (cols) axis must be 64-fp16-aligned, since
/// `assemble_pointwise` emits no coordinateMasking_ (guard #3, pointwise twin).
pub(crate) fn check_pointwise_cols(
    cols: u32,
    df: Df,
    what: &str,
    out_id: u32,
) -> Result<(), SuperDscError> {
    // ⭐ WALL 13 — the stick law at the DATA'S OWN df: an f32 op's columns are multiple of the
    // 32-lane IEEE_FP32 stick, an f16 op's of the 64-lane SEN169 one (`Df::elems_per_stick`,
    // the same single source every door reads).
    let stick = df.elems_per_stick();
    if !cols.is_multiple_of(stick) {
        return Err(SuperDscError(format!(
            "{what} t{out_id}: cols={cols} not a multiple of the {stick}-lane {} stick \
             (assemble_pointwise emits no coordinateMasking_) — pad cols",
            df.name()
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
        let (sdsc, _) = emit_sdsc("MatMul_0", &op, &folds, None).expect("emit_sdsc accepts it");
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

/// ⭐⭐⭐⭐⭐ THE GATHER'S CUT — one op per index stick, each writing a ROW WINDOW of the output.
///
/// # WHAT THE CARD DECIDED, AND WHY THESE ARE THE PROPERTIES TO PIN
///
/// `embedding_granite` is a `[256, 4096]` gathered row tile against a 32-entry `SenUint32` index
/// stick, so it is EIGHT ops. Four shapes of it were run on pod `nickm3-5999dffbdf-t2wcl` against the
/// fixture's own `ref_out.bin`, and only the last is right:
///
/// ```text
///   one op, 256 entries          rows[0:32] 1.000000, rows[0:256] 0.130597   every row `ids[r mod 32]`
///   eight legs, ROW-MAJOR dest   RAS::PCI::BusFence 0xa35e at prepare        (at seg0 AND at seg4)
///   eight legs, N_.mb_ = 32      rows[0:256] 0.017013                        plane stride 32 rows
///   eight legs, N_.mb_ = 256     rows[0:256] 1.000000, max|err| 0.0625       8/8 blocks exact
/// ```
///
/// `cargo test` cannot run `dxp_standalone` or a launch. What it CAN pin is every descriptor property
/// the last line is a function of, so that a change to any of them is a test failure saying "re-run
/// the card leg" instead of a silent return to one of the first three.
#[cfg(test)]
mod gather_cut {
    use super::*;
    use crate::ir::island::tile_op::{TileOp, TileOpKind};

    /// `embedding_granite`'s own extents — its `N_TOK` and Granite's `d_model`.
    const ROWS: u32 = 256;
    const COLS: u32 = 4096;

    /// One 32-entry index stick.
    fn cap() -> u32 {
        crate::sdsc_abstract::CopyDims::ENTRIES_PER_OP
    }

    /// The `TileOp` `lower_ktir_to_superdsc::scalarmul_at` states for this node: `[mb, out, y]` with
    /// `out` the stick axis and two live operands (the `[1,1]` scale rides in the op).
    fn tile(rows: u32) -> TileOp {
        let d = |name: &'static str, size: u32, is_stick: bool| ItDim {
            name,
            size,
            is_reduction: false,
            is_stick,
            df: Df::Fp16,
        };
        TileOp {
            kind: TileOpKind::PointwiseOrReduce { n_operands: 2 },
            dims: vec![d("mb", rows, false), d("out", COLS, true), d("y", 1, false)],
            df: Df::Fp16,
        }
    }

    /// The door, driven exactly as `scalarmul_at` drives it: the `[1,1]` scale first and the gathered
    /// table LAST (the one position at which `attach_gather_index`'s insert and dbo's dominance
    /// requirement coincide). `layout: None` is this crate's unit-test arm, so addresses come from the
    /// per-arg segment fallback — inert for every property below, which are all DIFFERENCES.
    fn legs(rows: u32) -> Vec<EmittedOp> {
        // TRUE shapes, not `rbo` lies: the mint in `pointwise_broadcast_opspec_from_tile` (and the
        // gather door's output mint) checks every swept window against the handle's declared shape,
        // so the table and output are declared at the node's own `[rows, 4096]` — embedding_granite's
        // extents. The `[1,1]` scale keeps `rbo`: one RedStick IS its whole swept window.
        let scale = rbo("t4294967275");
        let table = rb(
            "t1",
            RowCount::of_token_rows(rows),
            BlockCols::of_feature_cols(COLS),
        );
        let out = rb(
            "t2",
            RowCount::of_token_rows(rows),
            BlockCols::of_feature_cols(COLS),
        );
        let t = tile(rows);
        let mut sid = 0i64;
        assemble_pointwise_broadcast_gather(
            PointwiseGather {
                op_name: "scalarmul_o2",
                tile_op: &t,
                op_func: "multiply",
                rows,
                cols: COLS,
                inputs: &[In::scalar(&scale).ew(), In::full(&table).ew()],
                gathered_input: 1,
                index_name: "t0",
                run: crate::superdsc_opspec::StickRun::covering(0, rows)
                    .expect("the whole-buffer run covers any node"),
                o: &out,
            },
            &mut sid,
            None,
        )
        .expect("the cut emits")
    }

    /// The `dscs_[0]` body of an emitted leg.
    fn body(op: &EmittedOp) -> &crate::wire::Dsc {
        op.op
            .as_ref()
            .expect("a gather leg carries a descriptor")
            .dscs_[0]
            .values()
            .next()
            .expect("one dsc per leg")
    }

    /// The per-core start addresses of operand `lds` of a leg, lowest first. Operand order is
    /// `[scale, table, index, output]` — the index is INSERTED before the output, so the output is 3.
    fn starts(op: &EmittedOp, lds: u32) -> Vec<u64> {
        let node = body(op)
            .scheduleTree_
            .iter()
            .find(|n| n.nodeType_ == "allocate" && n.ldsIdx_ == lds)
            .expect("an allocate node per operand");
        let mut v: Vec<u64> = node
            .startAddressCoreCorelet_
            .data_
            .values()
            .map(|s| s.parse::<u64>().expect("a decimal address"))
            .collect();
        v.sort_unstable();
        v
    }

    /// ⭐ ONE LEG PER INDEX STICK, AND THE ACCEPTING SIDE STILL EMITS ONE OP.
    ///
    /// `ENTRIES_PER_OP` is the whole reason the cut exists, so the leg count is asserted as a function
    /// of it rather than as the literal 8 — a change to that constant must move this test's
    /// expectation, not break its assertion.
    #[test]
    fn the_gather_is_cut_into_one_leg_per_index_stick() {
        assert_eq!(
            legs(ROWS).len() as u32,
            ROWS / cap(),
            "one leg per index stick"
        );
        // ⭐ THE CONTROL THAT MAKES IT A CUT AND NOT A REWRITE: at exactly one stick the door emits the
        // single whole-node op it always did — the descriptor `embedding_granite_m32` proved on card at
        // `within_2pct_strict = 1.000000`, and which still bakes BYTE-IDENTICALLY after this change.
        assert_eq!(legs(cap()).len(), 1, "one index stick is one op");
        // The leg names distinguish them, because they become the bake's per-file keys.
        let names: Vec<String> = legs(ROWS).iter().map(|o| o.op_name.clone()).collect();
        assert_eq!(names.first().map(String::as_str), Some("scalarmul_o2_g0"));
        assert_eq!(names.last().map(String::as_str), Some("scalarmul_o2_g7"));
        assert_eq!(
            legs(cap())[0].op_name,
            "scalarmul_o2",
            "an uncut op keeps the node's name"
        );
    }

    /// ⭐⭐⭐ `N_`'s `mb_` IS THE **TENSOR'S** ROW COUNT WHILE THE WORK DIVISION IS THE **LEG'S** — the
    /// one number that moved the card from 0.017013 to 1.000000.
    ///
    /// `maxDimSizes_ = -1` means "reconstruct the stick-blocked `device_size` from
    /// `N_`/`layoutDimOrder_`/`stickSize_`", and it is `N_`'s `mb_` that the reconstruction reads for
    /// the stick-PLANE stride. At the leg's own 32 the eight row windows overlap — measured off the
    /// card's own bytes, leg `k`'s plane `s` landing at device run `32·(k+s)`, with run 512 holding
    /// leg 7's plane 9; at the tensor's 256 all eight land exactly.
    #[test]
    fn a_legs_residency_is_the_whole_tensor_and_its_work_division_is_its_own() {
        for (k, op) in legs(ROWS).iter().enumerate() {
            let dsc = op.op.as_ref().expect("a descriptor");
            let d = body(op);
            assert_eq!(
                d.N_.mb_, ROWS as i64,
                "leg {k}: N_ carries the TENSOR's rows"
            );
            assert_eq!(d.N_.out_, COLS as i64, "leg {k}: N_ `out` is untouched");
            // The WORK is the leg's own: `cap` rows over `cap` cores, one row each.
            assert_eq!(
                dsc.numWkSlicesPerDim_["mb"],
                cap(),
                "leg {k}: the leg's own row split"
            );
            assert_eq!(
                d.dataStageParam_["0"].ss_.mb_, 1,
                "leg {k}: one row per core"
            );
            assert_eq!(
                d.dataStageParam_["0"].el_.mb_, 1,
                "leg {k}: one row per core"
            );
        }
        // ⛔ AND AN UNCUT OP IS UNTOUCHED, which is what keeps the m32 descriptor byte-identical: with
        // one stick there is no window, so `N_` is the plan's own extent exactly as before the cut.
        assert_eq!(
            body(&legs(cap())[0]).N_.mb_,
            cap() as i64,
            "an uncut op's N_ is its own extent"
        );
    }

    /// ⭐⭐⭐ EACH LEG READS ITS OWN INDEX STICK — `EntryBase::of_sticks(k)`. Without it every leg reads
    /// leg 0's entries, which is 32 right rows and 224 wrong ones: exactly the `ids[r mod 32]` the
    /// un-cut op measured at 0.130597.
    ///
    /// One 32-entry `SenUint32` stick is 128 B, so consecutive legs' index allocations are exactly that
    /// far apart. And the index is UNSPLIT — dxp gives `arrayB` no per-core HBM offset at all
    /// (`progCorrection.cpp:524`), so every core of a leg shares ONE address.
    #[test]
    fn each_leg_reads_the_next_index_stick_and_all_its_cores_share_it() {
        let stick_bytes = u64::from(cap())
            * u64::from(<crate::superdsc_opspec::SenUint32 as DataFormat>::WORD_LENGTH);
        assert_eq!(stick_bytes, 128, "a 32-entry int32 IBR stick is 128 B");
        let ops = legs(ROWS);
        let bases: Vec<u64> = ops
            .iter()
            .map(|o| {
                let s = starts(o, 2);
                assert!(
                    s.iter().all(|&a| a == s[0]),
                    "the index is UNSPLIT: every core reads one address"
                );
                s[0]
            })
            .collect();
        for k in 1..bases.len() {
            assert_eq!(
                bases[k] - bases[k - 1],
                stick_bytes,
                "leg {k} reads the NEXT index stick"
            );
        }
    }

    /// ⭐⭐⭐ EACH LEG WRITES THE NEXT ROW WINDOW — `first_row · lanes`, NOT `first_row · cols`.
    ///
    /// Row `r` of a stick-major `[R, C]` tensor begins at element `r · lanes` (its column-0 lane of
    /// plane 0); `C` does not enter, because the plane stride carries the columns. So consecutive legs'
    /// destinations are `ENTRIES_PER_OP · lanes · 2` bytes apart — 4096 for fp16 — and NOT the
    /// whole-block `ENTRIES_PER_OP · cols · 2` = 262144, which is what an eight-BLOCK output would be
    /// and which the host would then have to de-block. Both forms bake and neither faults; the card
    /// separates them (the block form scores 0.014110 read as a `[256, 4096]` tensor).
    #[test]
    fn each_leg_writes_the_next_row_window_of_the_output() {
        let lanes = u64::from(Fp16::ELEMS_PER_STICK);
        let w = u64::from(Fp16::WORD_LENGTH);
        let ops = legs(ROWS);
        let bases: Vec<u64> = ops.iter().map(|o| starts(o, 3)[0]).collect();
        let window = u64::from(cap()) * lanes * w;
        let block = u64::from(cap()) * u64::from(COLS) * w;
        assert_ne!(window, block, "the two candidate strides really differ");
        for k in 1..bases.len() {
            assert_eq!(
                bases[k] - bases[k - 1],
                window,
                "leg {k}'s destination is one {}-row WINDOW on, not one block",
                cap()
            );
        }
        // The per-core step is the stick-major column-0 corner, `lanes · 2` — the same number under
        // either row count, which is why only `N_` had to change.
        let s = starts(&ops[0], 3);
        assert_eq!(s[1] - s[0], lanes * w, "one core per row, stick-major");
        assert_eq!(s.len() as u32, cap(), "one core per row of the leg");
    }

    /// ⛔⛔⛔ THE 32-ENTRY CEILING IS STILL ENFORCED ON THE OP THAT DECLARES THE INDEX, and the cut did
    /// not weaken it: `one_gathered_leg` refuses a longer run BY NAME. It is unreachable from the cut
    /// by construction, so it is reached here with a hand-built over-long leg — which is what makes it
    /// a proof rather than dead code. The over-long leg names a TWO-STICK run, the spelling the
    /// [`crate::superdsc_opspec::StickRun`] type admits (a hand-built span) but the leg guard refuses.
    ///
    /// ⭐ AND THE DISCRIMINATING CONTROL IS ONE VARIABLE AWAY: the same leg at one stick emits.
    #[test]
    fn a_leg_longer_than_one_index_stick_is_refused_by_name() {
        let scale = rbo("t4294967275");
        // TRUE shape (`[cap()*2, 4096]`, covering both legs below): the input mint must not fire
        // before the one-stick ceiling this test exists to pin.
        let table = rb(
            "t1",
            RowCount::of_token_rows(cap() * 2),
            BlockCols::of_feature_cols(COLS),
        );
        let ins = [In::scalar(&scale).ew(), In::full(&table).ew()];
        let two = tile(cap() * 2);
        let mut sid = 0i64;
        let e = match one_gathered_leg(
            GatherLeg {
                op_name: "scalarmul_o2",
                leg_name: "scalarmul_o2_g0",
                tile_op: &two,
                op_func: "multiply",
                leg_rows: cap() * 2,
                node_rows: cap() * 2,
                first_row: 0,
                cols: COLS,
                inputs: &ins,
                gathered_input: 1,
                index_name: "t0",
                page: crate::superdsc_opspec::PageExtent::single_position(),
                run: crate::superdsc_opspec::StickRun::of_sticks(0, 2),
                o_name: "t2",
                head_major: false,
            },
            &mut sid,
            None,
        ) {
            Err(e) => e,
            Ok(_) => panic!("two index sticks in one leg must be refused"),
        };
        assert!(
            e.0.contains(&format!("{} index entries", cap() * 2)) && e.0.contains("ONE gather op"),
            "the refusal names the entry count and the one-stick law: {}",
            e.0
        );
        // THE DISCRIMINATING CONTROL, one variable away: the SAME leg at one stick emits.
        let one = tile(cap());
        let mut sid = 0i64;
        assert!(
            one_gathered_leg(
                GatherLeg {
                    op_name: "scalarmul_o2",
                    leg_name: "scalarmul_o2_g0",
                    tile_op: &one,
                    op_func: "multiply",
                    leg_rows: cap(),
                    node_rows: cap(),
                    first_row: 0,
                    cols: COLS,
                    inputs: &ins,
                    gathered_input: 1,
                    index_name: "t0",
                    page: crate::superdsc_opspec::PageExtent::single_position(),
                    run: crate::superdsc_opspec::StickRun::of_sticks(0, 1),
                    o_name: "t2",
                    head_major: false,
                },
                &mut sid,
                None,
            )
            .is_ok(),
            "one index stick in one leg emits"
        );
    }
}

/// ⭐ `fp32toint32` — the MoE expert gather's INDEX PRODUCER, and the descriptor properties the
/// vendor DDL states for it (unary_parallel.ddl:50, datastage lines 879-889).
///
/// `cargo test` cannot run a bake. What it CAN pin is every property the gather's DT_CHECKs are a
/// function of — so a change to any of them is a test failure saying "re-run the card leg" instead
/// of a silent bake refusal (`L3DlOpsScheduler.cpp:5928` wants index `wordLength == 4`;
/// `GatherIndexConversion.cpp:133` wants `dataFormat_ == SENUINT32`).
#[cfg(test)]
mod fp32toint32_convert_law {
    use super::*;

    /// The convert at the MoE router's own geometry: the top-k output's `[m, W]` rows over the
    /// router's padded expert width (tiny26: m=2, E=8, W=64).
    fn op(rows: u32) -> OpSpec {
        convert_opspec(OpFunc::Fp32ToInt32, rows, 64, "rt_topk", "moe_idx")
            .expect("the convert emits")
    }

    /// The output arg (arg 1, the non-input) — the INDEX dataspace-to-be.
    fn out_arg(op: &OpSpec) -> &AnyTensorArg {
        op.args
            .iter()
            .find(|a| !a.view().is_input)
            .expect("an output arg")
    }

    /// ⭐ THE OUTPUT IS SENUINT32 AT 4 BYTES and the INPUT IS FP32 AT 4 BYTES — the numbers the
    /// gather's DT_CHECKs read, and the slice-equality law (unary_parallel.ddl:59): both operands'
    /// sticks are 32-elem (slice 4 each, the match), and the pair (4, 4) is also the control that
    /// the two operands were not collapsed onto one df.
    #[test]
    fn the_output_is_senuint32_and_the_input_is_fp32_both_four_bytes() {
        let o = op(2);
        let out = out_arg(&o);
        assert_eq!(out.view().df, Df::SenUint32, "the gather's index format");
        assert_eq!(out.view().df.word_length(), 4, "DT_CHECK wordLength == 4");
        let inp = o
            .args
            .iter()
            .find(|a| a.view().is_input)
            .expect("an input arg");
        assert_eq!(inp.view().df, Df::Fp32, "the fp32 staging leg's format");
        assert_eq!(inp.view().df.word_length(), 4, "the input is 4-byte fp32");
    }

    /// ⭐ THE TWO STICKS MATCH AT 32 — the slice-equality constraint's arithmetic
    /// (unary_parallel.ddl:59: slice = stickSize/8 elements, must be EQUAL): both 32-elem
    /// sticks yield slice 4, the vendor `sdsc_fp32toint32.json` fixture's own geometry. An
    /// fp16 input here (stick 64, slice 8) is the dxp "slice size does not match" bake
    /// refusal (ddl_conversion.cpp:2728).
    #[test]
    fn both_operands_sticks_yield_the_same_slice() {
        let o = op(2);
        let inp = o
            .args
            .iter()
            .find(|a| a.view().is_input)
            .expect("an input arg");
        let out = out_arg(&o);
        assert_eq!(
            inp.view().df.elems_per_stick(),
            out.view().df.elems_per_stick(),
            "the slice-equality law: equal element counts per stick"
        );
        assert_eq!(
            out.view().df.elems_per_stick(),
            32,
            "the u32 output's stick"
        );
    }

    /// ⭐ THE NARROWING CONVERT STAYS RANK-3 — the `y` dim dxp's buffer walk indexes (the rank-2
    /// collapse of the fp16↔fp32 converts crashed `getBufferCapacityForNode` on a real pod). Pinned
    /// as the OUTPUT arg's rank, the side whose stick narrowed; the fp16↔fp32 pair drives the same
    /// law through `is_fp32_convert`, which is why the check here is the shared flag's effect
    /// (`is_narrowing_convert`) and not a copy of its body.
    #[test]
    fn the_narrowing_convert_stays_rank_3() {
        let o = op(2);
        let out = out_arg(&o);
        assert!(
            matches!(out, AnyTensorArg::R3(_)),
            "a narrowing convert's operands keep the `y` dim"
        );
    }

    /// ⭐ THE OPFUNC SPELLING — `fp32toint32`, the name `dscdefn.cpp` recognizes; a wrong name is
    /// an on-card `Unrecognized opFunc` refusal, and the string round-trip through
    /// `op_func_from_str` is what every assembler uses.
    #[test]
    fn the_name_round_trips() {
        assert_eq!(OpFunc::Fp32ToInt32.name(), "fp32toint32");
        assert_eq!(op_func_from_str("fp32toint32"), OpFunc::Fp32ToInt32);
    }

    /// ⭐ THE EX UNIT IS SFP — the cast runs on the SFP unit (the DDL datastage's `sfp`/`lxsu`
    /// units), which `OpFunc::ex_unit` derives structurally; pinned so a reclassification of the
    /// converts cannot silently move the op to `pt`.
    #[test]
    fn the_ex_unit_is_sfp() {
        assert_eq!(OpFunc::Fp32ToInt32.ex_unit().as_str(), "sfp");
    }

    /// The STAGED default: `convert_opspec` (every existing caller, the m=1 chain)
    /// still emits stick-on-`out` and classifies stick-major at rows>1.
    #[test]
    fn the_staged_default_is_unchanged() {
        let o = convert_opspec(OpFunc::Fp32ToInt32, 128, 128, "a", "b").expect("emits");
        let out = out_arg(&o);
        assert_eq!(out.view().stick, "out");
        assert!(
            matches!(
                view_stick_layout(&out.view(), &o.iter).addressing(),
                crate::sdsc_abstract::Addressing::StickBlocked
            ),
            "the staged output stays stick-major at rows>1"
        );
    }
}

/// ⛔⛔⛔ A BROADCAST OPERAND IN A TIME-TILED POINTWISE OP MUST BE **ADDRESS-STABLE** ACROSS TRIPS.
///
/// This is the class that fired on the gemma-4-12b fp8 bake (~80 min in): the prefill tanhsoftcap's
/// `tscdiv` — `realdiv` of the vocab-wide logits `[rows=35, cols=262144]` by the `[1,1]` cap const
/// (`In::scalar`, `out` scale `RedStick`) — overflows LX residency at full width, so the tiler picks
/// `time > 1`, and the old refusal (`"a broadcast-over-'out' operand in a TIME-TILED op is not yet
/// supported"`) turned the model's own softcap into a build failure.
///
/// The correct semantics is NOT "refuse" and NOT "advance": every trip's out-window must read the
/// SAME broadcast value at the SAME base address. That is what `rewrite_op_for_time_tile`'s tiled
/// predicate now guarantees — an operand is tiled only when it RANGES the tiled dim (`scale` ==
/// `Active` there); a RedStick/RedNonStick operand is exempt, keeps the collapsed fold
/// `build_coordinates` wrote, and gets an EMPTY affine-stride map, which is `concrete_trips`'s
/// no-bump path (an empty map keeps the base address).
///
/// The properties below pin the three observable halves of that contract on the real door
/// (`assemble_pointwise_broadcast_off`, the same one `assemble_tanhsoftcap` calls):
///   1. the op is genuinely time-tiled (`time > 1`) and the broadcast operand survives;
///   2. the broadcast operand has NO affine stride (the no-bump path) while some full-width operand
///      DOES (the control — an empty stride map everywhere would make (2) vacuous);
///   3. across the expanded concrete trips, the broadcast operand's start address is IDENTICAL in
///      every trip while a strided operand's moves — and it actually advances (a stride of 0 would
///      silently re-read trip 0, the #53-class bug).
#[cfg(test)]
mod time_tiled_broadcast_operand_is_address_stable {
    use super::*;
    use crate::sdsc_abstract::{BlockCols, RowCount};

    /// The gemma-4 tanhsoftcap prefill geometry: vocab-wide cols (262144, many sticks) with few
    /// rows, so `n_operands · per_core_mb · out_per_time · 2` overflows the 1,677,721-B usable LX
    /// at `out_per_time == full` and the divisor search lands on `time > 1`.
    const ROWS: u32 = 35;
    const COLS: u32 = 262144;

    /// The failing op itself, verbatim: `tscdiv` = realdiv(x, cap), x full, cap `[1,1]` broadcast.
    fn tscdiv() -> EmittedOp {
        let x = rb(
            "tsc_x",
            RowCount::of_token_rows(ROWS),
            BlockCols::of_feature_cols(COLS),
        );
        let cap = rb(
            "tsc_cap",
            RowCount::of_broadcast_row(),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let out = rb(
            "tsc_o",
            RowCount::of_token_rows(ROWS),
            BlockCols::of_feature_cols(COLS),
        );
        assemble_pointwise_broadcast_off(
            PointwiseFacts {
                op_name: "tscdiv_t2326",
                op_func: "realdiv",
                rows: RowCount::of_token_rows(ROWS),
                cols: BlockCols::of_feature_cols(COLS),
                inputs: &[In::full(&x).ew(), In::scalar(&cap).ew()],
                o: &out,
                out_offset: crate::addr::DevOff::ZERO.into(),
            },
            &mut 0i64,
            None,
        )
    }

    /// The first start address of operand `lds` in trip `t`, as `concrete_trips` expanded it.
    fn trip_start(e: &EmittedOp, t: usize, lds: u32) -> u64 {
        let trips = e.concrete_trips();
        let node = trips.get(t).expect("a trip per time").dscs_[0]
            .values()
            .next()
            .expect("one dsc")
            .scheduleTree_
            .iter()
            .find(|n| n.nodeType_ == "allocate" && n.ldsIdx_ == lds)
            .expect("an allocate node per operand");
        node.startAddressCoreCorelet_
            .data_
            .values()
            .next()
            .expect("a per-core address")
            .parse()
            .expect("a decimal address")
    }

    #[test]
    fn the_op_time_tiles_and_keeps_the_broadcast_operand() {
        let e = tscdiv();
        assert!(
            e.time > 1,
            "vocab-wide softcap must time-tile, got time={}",
            e.time
        );
        assert_eq!(e.arg_bindings.len(), 3, "x, cap, out — in door order");
        assert_eq!(
            e.arg_bindings[1].footprint_bytes,
            Fp16::ELEMS_PER_STICK as u64 * 2,
            "the cap's materialized footprint is ONE stick — the broadcast survived the door"
        );
    }

    #[test]
    fn the_broadcast_operand_takes_no_stride_but_the_full_operand_does() {
        let e = tscdiv();
        assert!(
            e.affine_strides[1].is_empty(),
            "the cap (out-broadcast) must have NO per-trip stride — concrete_trips keeps its base"
        );
        assert!(
            e.affine_strides.iter().any(|m| !m.is_empty()),
            "control: at least one operand (x and/or out) IS tiled and carries a stride"
        );
    }

    #[test]
    fn across_trips_the_broadcast_address_is_fixed_and_the_full_operand_advances() {
        let e = tscdiv();
        // The cap (lds 1): identical address in EVERY trip.
        let cap0 = trip_start(&e, 0, 1);
        for t in 1..e.time as usize {
            assert_eq!(
                trip_start(&e, t, 1),
                cap0,
                "trip {t} moved the broadcast operand's address — every trip must read the same value"
            );
        }
        // x (lds 0) and out (lds 2): whichever carries a stride must actually advance.
        for lds in [0u32, 2] {
            if e.affine_strides[lds as usize].is_empty() {
                continue; // only the strided operand is under test here
            }
            let a0 = trip_start(&e, 0, lds);
            let a1 = trip_start(&e, 1, lds);
            assert!(
                a1 > a0,
                "strided operand {lds} did not advance between trips 0→1 ({a0:#x} → {a1:#x}) — \
                 a zero stride would re-read trip 0's window"
            );
        }
    }
}

/// ⛔ THE COMPLEMENT OF THE BROADCAST LAW: an operand collapsed along a dim OTHER than
/// the tiled one still RANGES the tiled dim and MUST ADVANCE per trip.
///
/// `In::mb` — a `[1, cols]` row vector broadcast over rows (the rmsnorm gamma's own
/// form; `EwOperand::scale()` maps it to `mb` = RedNonStick with `out` staying ACTIVE) —
/// is NOT one of the address-stable broadcasts: in an op time-tiled over `out`, each
/// trip's window must read THAT trip's slice of the row, so the operand takes the
/// per-trip affine stride and its start address moves. Only the scale AT the tiled dim
/// decides exemption; a RedNonStick on `mb` exempts nothing when tiling `out`. Pinned on
/// the same door and geometry as the address-stable module, alongside the input-side
/// no-advance guard (2b) that makes a strideless ranging operand a build Err.
#[cfg(test)]
mod time_tiled_mb_operand_advances_per_trip {
    use super::*;
    use crate::sdsc_abstract::{BlockCols, RowCount};

    /// The same gemma-4 tanhsoftcap prefill geometry as the address-stable module:
    /// vocab-wide cols, few rows — the tiler lands on `time > 1`.
    const ROWS: u32 = 35;
    const COLS: u32 = 262144;

    /// realdiv(x, gamma): x full `[rows, cols]`, gamma an `In::mb` `[1, cols]` row
    /// vector — broadcast over ROWS, ACTIVE over cols (the rmsnorm-gamma form).
    fn mbdiv() -> EmittedOp {
        let x = rb(
            "mb_x",
            RowCount::of_token_rows(ROWS),
            BlockCols::of_feature_cols(COLS),
        );
        let gamma = rb(
            "mb_gamma",
            RowCount::of_broadcast_row(),
            BlockCols::of_feature_cols(COLS),
        );
        let out = rb(
            "mb_o",
            RowCount::of_token_rows(ROWS),
            BlockCols::of_feature_cols(COLS),
        );
        assemble_pointwise_broadcast_off(
            PointwiseFacts {
                op_name: "mbdiv_t2326",
                op_func: "realdiv",
                rows: RowCount::of_token_rows(ROWS),
                cols: BlockCols::of_feature_cols(COLS),
                inputs: &[In::full(&x).ew(), In::mb(&gamma).ew()],
                o: &out,
                out_offset: crate::addr::DevOff::ZERO.into(),
            },
            &mut 0i64,
            None,
        )
    }

    /// The first start address of operand `lds` in trip `t`, as `concrete_trips` expanded it.
    fn trip_start(e: &EmittedOp, t: usize, lds: u32) -> u64 {
        let trips = e.concrete_trips();
        let node = trips.get(t).expect("a trip per time").dscs_[0]
            .values()
            .next()
            .expect("one dsc")
            .scheduleTree_
            .iter()
            .find(|n| n.nodeType_ == "allocate" && n.ldsIdx_ == lds)
            .expect("an allocate node per operand");
        node.startAddressCoreCorelet_
            .data_
            .values()
            .next()
            .expect("a per-core address")
            .parse()
            .expect("a decimal address")
    }

    #[test]
    fn the_op_time_tiles_and_keeps_the_row_vector_at_full_width() {
        let e = mbdiv();
        assert!(
            e.time > 1,
            "vocab-wide op must time-tile, got time={}",
            e.time
        );
        assert_eq!(e.arg_bindings.len(), 3, "x, gamma, out — in door order");
        assert_eq!(
            e.arg_bindings[1].footprint_bytes,
            COLS as u64 * 2,
            "the mb operand's materialized footprint is the FULL row (one stick would mean \
             it was collapsed along `out` — the scalar form, not the gamma form)"
        );
    }

    #[test]
    fn the_mb_operand_takes_a_nonzero_stride_along_out() {
        let e = mbdiv();
        assert!(
            e.affine_strides[1].get("out").copied().unwrap_or(0) > 0,
            "the mb operand RANGES `out` at Active scale — it must carry a per-trip stride \
             (empty like the scalar cap's would re-read trip 0's window of the row)"
        );
    }

    #[test]
    fn across_trips_the_mb_operand_advances() {
        let e = mbdiv();
        let g0 = trip_start(&e, 0, 1);
        let g1 = trip_start(&e, 1, 1);
        assert!(
            g1 > g0,
            "the mb operand moved {g0:#x} → {g1:#x}: it must ADVANCE — each trip reads its \
             own out-window of the row, unlike the address-stable scalar broadcast"
        );
    }
}

/// ⭐ THE FactsMatchBindings WITNESS, PINNED AT THE STAMP: `EmittedOp::new` refuses a
/// facts/bindings length mismatch BY NAME before stamping anything. The door resolves
/// one fact per spec operand; a parallel-array mismatch here would silently stamp one
/// operand's resolution onto another's binding (the exact drift class records 53 and
/// 54 paid for), so it is a build-time panic, never an unchecked zip.
#[cfg(test)]
mod facts_match_bindings_witness {
    use super::*;

    /// Emit a real three-operand matmul through the door, then re-stamp it with a
    /// TRUNCATED facts list — the shape only an internal defect can produce.
    #[test]
    #[should_panic(
        expected = "EmittedOp factless: the door resolved 1 operands but the op binds 3 \
                    (FactsMatchBindings witness)"
    )]
    fn a_facts_bindings_length_mismatch_panics_naming_the_op() {
        let op =
            crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec(64, 64, 64, 1, "tA", "tW", "tO")
                .expect("a small matmul is a legal OpSpec");
        let folds = crate::superdsc_opspec::SdscFoldSet::new(op.iter.cores_used());
        let (inner, facts) =
            emit_sdsc("factless", &op, &folds, None).expect("the small matmul emits");
        assert_eq!(
            facts.len(),
            op.args.len(),
            "the fixture premise: the door resolved one fact per operand"
        );
        // The mismatch: 1 fact for 3 operands. The panic must name the OP and state
        // both counts before any binding is stamped.
        let _ = EmittedOp::new(
            "factless".to_string(),
            inner,
            &op.args,
            &op.iter,
            1,
            vec![std::collections::BTreeMap::new(); op.args.len()],
            &facts[..1],
        );
    }
}
