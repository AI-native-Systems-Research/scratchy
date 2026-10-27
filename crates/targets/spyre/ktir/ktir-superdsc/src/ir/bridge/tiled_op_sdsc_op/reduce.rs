//! Bridge 3 (`TiledOp -> SdscOp`), reduce family: the PRIMITIVE (one `TileOp` in, one
//! `OpSpec`/`EmittedOp` out, no internal decomposition) TileIR→SdscOp entry points for stick
//! reduces (sum/mean/max). See `super`'s module doc for why this is its own small file rather
//! than folded into a bigger one.

use crate::addr::{DevOff, Win};
use crate::emit::{EmittedOp, op_func_from_str, sen169_bits};
use crate::ir::island::tile_op::{TileOp, TileOpKind};
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{BlockCols, KindTag, RowCount, StickKind, Stk};
use crate::superdsc_opspec::{
    Allocation, AnyTensorArg, DataFormat, Df, Fp16, Fp32, MAX_CORES, MaxCores, OpFunc, OpInfo,
    OpSpec, Role, SamvCoordinateMask, Scale, SdscFoldSet, TensorArg, assert_df_stick_multiple,
};
use crate::work::distribute_cores;

/// ⛔ A REDUCE'S TileOp KIND IS THE DDL's OWN INPUT ARITY, NOT A LITERAL OPERAND COUNT. The
/// FUSED unseeded reduce (`quantscalepertokenfp8`, `Some(1)`) keeps its accumulator ON-CHIP
/// as a `ddl.internal_tensor` (`quant_scale_per_token.ddl:41`), so its per-core LX residency
/// is ONE `[per_core_mb, k]` operand — `PointwiseOrReduce { n_operands: 1 }`, the charge that
/// was always correct for it. The SEEDED convention (`sum`/`max`/`mean`,
/// `ddl_input_arity() = None`) binds data + accumulator as `inputLabeledDs[0..1]` — two
/// HBM-resident operands — and its accumulator is ONE STICK PER ROW whatever the trip, so its
/// kind is `SeededReduce`: the data at its per-trip width plus the accum at its true
/// one-stick size. A literal `n_operands = 2` modeled the accum as data-width, an
/// OVER-CHARGE that split wide reduces into time-tiled forms whose per-trip outputs ALIAS —
/// the accum is `Scale::RedStick`, stride-exempt in `rewrite_op_for_time_tile` — exactly the
/// collision the #50 `tiled_trips_alias` guard refuses. Two measured bites of the same law:
/// a PRIME prefill rung where `core_split` keeps the whole row count per core (the fused
/// arity fix: granite-3.1-2b mq=71 k=8192 billed 2,324,224 B; gemma-4-12b mq=43 k=15360
/// billed 2,641,920 B vs the 1,677,721-B budget). Anything else (`None`/`Some(2)+`) is the
/// seeded convention.
fn reduce_tile_kind(op: OpFunc) -> TileOpKind {
    // ⭐ WALL 12: the nonstick twins are a DISTINCT tiling kind — their accumulator is the
    // survivor vector, not one stick per row, so the seeded residency formula would charge the
    // wrong shape. Everything else keeps the arity-picked kind.
    if matches!(op, OpFunc::SumNonStick | OpFunc::MaxNonStick) {
        return TileOpKind::NonstickReduce;
    }
    match op.ddl_input_arity() {
        Some(1) => TileOpKind::PointwiseOrReduce { n_operands: 1 },
        _ => TileOpKind::SeededReduce,
    }
}

/// The stick-padded extent of `cols` under `df` — the reduction axis's TRUE on-device width.
/// Unlike a matmul's N/K extents ([`crate::superdsc_opspec::StickExtent::new`] REFUSES
/// sub-stick widths outright — a non-multiple-of-64 tile faults the scheduler,
/// L3DlOpsScheduler:1040), a reduce PADS UP to the next whole stick and masks the pad lanes
/// with the reduce's identity ([`reduce_coordinate_mask`]): the padded extent is still a
/// whole-stick dim the scheduler accepts, and the mask makes the padding numerically inert.
/// This is the vendor's own shape — `_get_padded_iteration_space` (codegen/superdsc.py:477-507)
/// pads every iteration space to stick alignment, and their `[mb=32, 514]` max fixture walks
/// `out_ = 576` with `coordinateMasking_: {"out": [[514, 62]]}` (sdsc_max.json).
fn reduce_stick_elems(cols: u32, df: Df) -> Result<u32, String> {
    if cols == 0 {
        return Err(
            "reduce width 0: a reduce needs at least one element along its reduced axis"
                .to_string(),
        );
    }
    let eps = if matches!(df, Df::Fp32) {
        <Fp32 as DataFormat>::ELEMS_PER_STICK
    } else {
        <Fp16 as DataFormat>::ELEMS_PER_STICK
    };
    Ok(cols.div_ceil(eps) * eps)
}

/// The reduce's SHARED `TileOp` front matter — the `[mb, out(stick), y]` iteration domain
/// (the stick-padded `cols` as the reduction stick axis) plus the arity-picked kind — ONE
/// constructor behind both `reduce_opspec_df` and `reduce_opspec_off`, so the two variants
/// cannot drift in what they declare to the tiler. `Err` only where the emission itself
/// would refuse (a `StickExtent` no stick-legal reduce can state). Returns the padded stick
/// extent alongside, the number the coordinate mask is computed from.
fn reduce_dims_kind(
    op: OpFunc,
    rows: u32,
    cols: u32,
    df: Df,
) -> Result<(Vec<crate::superdsc_opspec::ItDim>, TileOpKind, u32), String> {
    let stick_elems = reduce_stick_elems(cols, df)?;
    let dims = vec![
        crate::superdsc_opspec::ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df,
        },
        crate::superdsc_opspec::ItDim {
            name: "out",
            size: stick_elems,
            is_reduction: true,
            is_stick: true,
            df,
        },
        crate::superdsc_opspec::ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df,
        },
    ];
    Ok((dims, reduce_tile_kind(op), stick_elems))
}

/// ⭐ THE SAMV COORDINATE MASK on a padded reduce stick — the vendor's own answer to an
/// unaligned reduced width. A reduce over `cols` that is not a stick multiple still walks the
/// STICK-PADDED extent (the allocator hands out whole sticks), so the pad lanes ride the
/// reduction as whatever the buffer holds — for `max` over non-positive data that is a
/// zeros-read (`max(−65504, 0, x) = 0`, the measured `mxp=0` signature). The device's fix is
/// `coordinateMasking_` plus a `samv-maskvalue` constant: every coordinate past `valid` is fed
/// the mask word instead of a memory read, and the mask word is the reduce's IDENTITY — the
/// padding then contributes nothing. torch-spyre builds exactly this pair whenever the reduced
/// stick dim padded (`_get_coordinate_mask` + `constants["samv-maskvalue"] =
/// _get_mask_value(op)`, codegen/superdsc.py:2357-2365 and :281-319), and the backend's own
/// multi-row max fixture pins the wire form (`ddc/ddl_templates/test/sdsc_max.json`:
/// `coordinateMasking_: {"out": [[514, 62]]}`, const word `"65534"`).
///
/// `mask_bits` is the RAW SEN169 word, NOT `sen169_bits` of a value: the fixture's `65534` =
/// `0xFFFE` is the DDL's own `negInf` pattern (`summeanmaxexx2.ddl:69-72` defines it as a
/// `define_constant`), not the value-true encoding of −65504 (`0xDE00`). `max` masks with
/// negInf; every other seeded kind (sum/mean/exx2) masks with 0 — the additive identity.
///
/// fp32 is REFUSED, not masked: SAMV masking is a ≤16-bit-element mechanism (the vendor's own
/// element-bit guard in `_get_coordinate_mask`), so an fp32 padded reduce has no faithful
/// descriptor — it would fold the padding into the result. That is a build `Err` here (garbage
/// refuses at bake), never a silent wrong answer.
fn reduce_coordinate_mask(
    op: OpFunc,
    cols: u32,
    stick_elems: u32,
    df: Df,
    accum_name: &str,
) -> Result<Option<SamvCoordinateMask>, String> {
    let pad = stick_elems - cols;
    if pad == 0 {
        return Ok(None);
    }
    if matches!(df, Df::Fp32) {
        return Err(format!(
            "reduce `{accum_name}`: the reduced width {cols} is not a stick multiple (stick \
             extent {stick_elems}, pad {pad}) and SAMV coordinate masking is fp16-only (the \
             vendor's own ≤16-bit element limit) — an fp32 reduce here would fold the padding \
             into the result. Keep the fp32 reduced width a stick multiple."
        ));
    }
    Ok(Some(SamvCoordinateMask {
        dim: "out",
        valid: cols,
        pad,
        mask_bits: if matches!(op, OpFunc::Max) { 0xFFFE } else { 0 },
    }))
}

/// ⭐ THE REDUCE FAMILY'S MINT — the DATA operand's [`Win`], the DEVICE ACCESS window this family's
/// own descriptor declares. The reduce walks the STICK-PADDED width (`reduce_stick_elems`: the pad
/// lanes ride the reduction masked by [`reduce_coordinate_mask`]), so the window's columns are the
/// padded extent — minting at the logical `cols` would under-bound the very read the descriptor
/// bills. `rows`/`cols` are the op's own typed dims, so a caller cannot state the extents by hand.
///
/// The authority is whichever the caller holds: a [`BundleLayout`] mints through
/// [`BundleLayout::win`] (both doors — the handle's declared shape and the placement footprint,
/// resolved by name exactly as the bake-time guard resolves it); without one there IS no placement
/// authority — the same absence `resolve_seg_base` has, where it checks nothing — so the handle's
/// own declared shape is the bound. The extents are stated from the typed dims either way; the
/// [`Win`] binds identity, offset and extents as ONE value, so a name and an offset from different
/// laws no longer type-check as a pair.
pub fn win_reduce_data<'a, K: KindTag>(
    layout: Option<&BundleLayout>,
    stk: &'a Stk<K>,
    off: DevOff,
    rows: RowCount,
    cols: BlockCols,
    what: &str,
) -> Result<Win<'a, K>, String> {
    let stick_elems = reduce_stick_elems(cols.get(), Df::Fp16)?;
    mint_reduce_win(layout, stk, off, rows.get(), stick_elems, what)
}

/// ⭐ THE REDUCE FAMILY'S MINT — the ACCUMULATOR's [`Win`]: ONE STICK PER ROW whatever the trip
/// (the `Scale::RedStick` collapse — the seeded convention's own law, `reduce_tile_kind`'s doc).
/// The handle passed here must carry the buffer's true shape (a `[rows, 64]` stick-wide buffer),
/// not the `1×64` annotation the family's `hm` sites historically used: the mint is the first
/// reader of a handle's layout as a BOUND, and the bound is only as true as the declaration.
pub fn win_reduce_accum<'a, K: KindTag>(
    layout: Option<&BundleLayout>,
    stk: &'a Stk<K>,
    off: DevOff,
    rows: RowCount,
    what: &str,
) -> Result<Win<'a, K>, String> {
    mint_reduce_win(
        layout,
        stk,
        off,
        rows.get(),
        <Fp16 as DataFormat>::ELEMS_PER_STICK,
        what,
    )
}

/// The two family mints' shared door: the placement authority when the caller holds one, the
/// handle's declared shape when it does not. A refusal is an expansion-time `Err` — a `cargo
/// build` failure — and names `what` (the op and operand role), the offset, the extents and the
/// bound it exceeded.
fn mint_reduce_win<'a, K: KindTag>(
    layout: Option<&BundleLayout>,
    stk: &'a Stk<K>,
    off: DevOff,
    rows: u32,
    cols: u32,
    what: &str,
) -> Result<Win<'a, K>, String> {
    match layout {
        Some(l) => l.win(stk, off, rows, cols, what),
        None => {
            // No layout was handed to the mint, so there is no placement to consult — the same
            // absence the bake-time guard has (its `None` arm checks nothing). The handle's own
            // declared shape is the bound: the SAME shape [`BundleLayout::win`]'s first door
            // checks, so a mint that passes here cannot refuse there.
            let declared = stk.layout();
            let shape_elems = declared.rows as u64 * declared.cols as u64;
            let off_elems = off.into_raw_elems() as u64;
            let end_elems = off_elems + rows as u64 * cols as u64;
            if end_elems > shape_elems {
                return Err(format!(
                    "{what}: the operand window [{off_elems}, {end_elems}) (elements) of '{}' \
                     runs past the end its handle declares — {rows}×{cols} elements over a handle \
                     that declares {}×{} at {:?}. An op that sweeps past its own tensor is a \
                     defect in the computation that produced this offset; it cannot compile.",
                    stk.name(),
                    declared.rows,
                    declared.cols,
                    declared.df,
                ));
            }
            Ok(Win::minted(stk, off, rows, cols))
        }
    }
}

/// `cols` axis. Two operands: `data` (all active) + `accum` (the accumulator,
/// both `inputLabeledDs[1]` and the sole output, `out`-axis stick-reduction
/// `[Active, RedStick, Active]`).
/// fp16 wrapper (every existing caller). The torch-spyre fp32 `mean(x²)` uses [`reduce_opspec_df`].
pub fn reduce_opspec(
    op: OpFunc,
    rows: u32,
    cols: u32,
    data_name: &str,
    accum_name: &str,
    head_major: bool,
) -> Result<OpSpec, String> {
    reduce_opspec_df(op, rows, cols, data_name, accum_name, head_major, Df::Fp16)
}

pub fn reduce_opspec_df(
    op: OpFunc,
    rows: u32,
    cols: u32,
    data_name: &str,
    accum_name: &str,
    head_major: bool,
    df: Df,
) -> Result<OpSpec, String> {
    // LOWER to TileIR: this reduce's iteration domain + kind, ONE `TileOp` declaration instead of
    // the inline `WorkPlan::divide` + hand-picked `pointwise_lx_resident` closure — reached from the
    // REAL `lower_one_node` SubtileTape dispatch (every reduce: rmsnorm's mean, softmax's max/sum,
    // fp8's amax), same as `pointwise_opspec`. The kind comes off the DDL's own input arity, and the
    // seeded accum is charged at its TRUE one-stick-per-row size — see [`reduce_tile_kind`].
    let (dims, kind, stick_elems) = reduce_dims_kind(op, rows, cols, df)?;
    // The pad-lane mask, stated before anything else so an unmaskable shape (fp32 + pad) is a
    // build Err here, not a descriptor that silently folds its padding into the result.
    let coordinate_mask = reduce_coordinate_mask(op, cols, stick_elems, df, accum_name)?;
    let tile_op = TileOp {
        kind,
        dims: dims.clone(),
        df,
    };
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, distribute_cores, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    // ── STICK-MAJOR RESIDUAL ── flip ONLY the DATA input to rank-2
    // stick-major [mb,out] so it matches a stick-major pointwise producer (e.g. rmsnorm's x²); at
    // rows>1 AND cols>64 the flat read scrambles per-row sums → inf. The ACCUM (the reduced single
    // 64-stick output) is UNCHANGED (already one stick). head_major (per-head attn reduce) stays rank-3.
    // `cols > 64` is this fn's OWN stated precondition (see the comment above) but was never actually
    // checked — dropping to rank-2 for EVERY rows>1 reduce regardless of width omits the `y` dim from
    // `layoutDimOrder_` entirely (not just size-1s it), which a real pod dxp_standalone crash
    // (`vector::_M_range_check`) confirmed breaks dxp's own dim-handling for a `cols<=64` op (the
    // attention softmax sum reduces, `attn_dp`/`attn_dn`, at prefill). Restricting to `cols > 64`
    // keeps the real fix for wide reduces intact.
    let stickmajor = !head_major && rows > 1 && cols > Fp16::ELEMS_PER_STICK;
    let device_dims_r2 = plan.iter_syms(["mb", "out"]);

    let data = if stickmajor {
        AnyTensorArg::R2(
            TensorArg::<2>::new(
                true,
                data_name.to_string(),
                Role::Output,
                [Scale::Active, Scale::Active],
                device_dims_r2,
                ["mb", "out"],
                "out",
                Allocation::Hbm,
            )
            .map_err(|e| e.0)?
            .with_df(df)
            .with_row_blocked(matches!(df, Df::Fp32)),
        )
    } else {
        AnyTensorArg::R3(
            TensorArg::<3>::new(
                true,
                data_name.to_string(),
                Role::Output,
                [Scale::Active, Scale::Active, Scale::Active],
                device_dims,
                ["mb", "out", "y"],
                "out",
                Allocation::Hbm,
            )
            .map_err(|e| e.0)?
            .with_df(df)
            .with_row_blocked(matches!(df, Df::Fp32)),
        )
    };
    // The accum MUST match the DATA's rank: a rank-2 stick-major data with a rank-3 accum is a MIXED-RANK
    // op — the shared `primaryDsInfo` is rank-2 `[mb,out]`, so dxp interprets the rank-3 accum against it,
    // searches for a dim it can't find, gets -1, and range-checks it against the 3-dim vector
    // (`vector::_M_range_check __n=18446744073709551615 >= size 3`, dxp out_of_range = a BUILD failure).
    // So in the stick-major path the accum is ALSO rank-2 `[mb,out]` (out = RedStick, the reduced single
    // stick); the rank-3 `[mb,out,y]` form stays only for the head-major (per-head attn) reduce.
    let accum = if stickmajor {
        AnyTensorArg::R2(
            TensorArg::<2>::new(
                false,
                accum_name.to_string(),
                Role::Output,
                [Scale::Active, Scale::RedStick],
                device_dims_r2,
                ["mb", "out"],
                "out",
                Allocation::Hbm,
            )
            .map_err(|e| e.0)?
            .with_df(df)
            .with_row_blocked(matches!(df, Df::Fp32)),
        )
    } else {
        AnyTensorArg::R3(
            TensorArg::<3>::new(
                false,
                accum_name.to_string(),
                Role::Output,
                [Scale::Active, Scale::RedStick, Scale::Active],
                device_dims,
                ["mb", "out", "y"],
                "out",
                Allocation::Hbm,
            )
            .map_err(|e| e.0)?
            .with_df(df)
            .with_row_blocked(matches!(df, Df::Fp32)),
        )
    };

    let args = vec![data, accum];
    // BUILD-TIME GUARD (guard-every-crash-at-build-time): DATA and ACCUM are both `Role::Output`, so they
    // SHARE one `primaryDsInfo` LayoutInfo of rank N. dxp interprets EACH operand's AllocNode dims against
    // that single rank-N layout — a mismatched-rank operand makes dxp search for a dim that isn't there,
    // get -1, and range-check it against the N-dim vector (`vector::_M_range_check __n=…MAX >= size N`, an
    // out_of_range abort = a dxp BUILD failure, seen when the stick-major data was rank-2 but the accum
    // rank-3). This runs in the `#[forward]` proc-macro, so returning `Err` here makes the INVALID
    // mixed-rank reduce a `cargo build` error instead — it can't compile if it's invalid.
    let ranks: Vec<usize> = args.iter().map(|a| a.view().layout.len()).collect();
    if let Some(&bad) = ranks.iter().find(|&&r| r != ranks[0]) {
        return Err(format!(
            "reduce `{accum_name}`: operands have MIXED ranks {ranks:?} (rank {} vs {bad}) — data and \
             accum share ONE primaryDsInfo, so every operand MUST be the same rank or dxp aborts \
             out_of_range. Build both from the same rank branch.",
            ranks[0]
        ));
    }
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    // The sum/mean/max reduce template (summeanmaxexx2.ddl) requires a
    // `scaling_factor` external const (= 1/N for mean, 1.0 for sum/max — packed
    // fp16). `cols` is the reduced extent N. `exx2` divides by the SAME 1/N — its
    // `exx2scale` const is the E[x²] divisor (the template's own `@outDivisor@`).
    let scale = match op {
        OpFunc::Mean | OpFunc::Exx2 => 1.0f32 / cols as f32,
        _ => 1.0f32, // sum / max: identity scaling
    };
    // The const is read by the device as SEN169_FP16 (1-6-9), NOT IEEE f16 (1-5-10) — see
    // `sen169_bits`'s own doc for why a plain `half::f16` encoding silently mis-scales every reduce.
    // The const's element type MUST match the OP's data_format (torch-spyre
    // `encode_constant(value, data_format)`, compute_ops.py:127): the fp32 reduce (mq>1 rmsnorm
    // mean(x²)) needs an IEEE_FP32 const (raw f32 bits), else a SEN169_FP16 const feeding an fp32 op
    // is a MIXED [fp16,fp32] op → DD2 "Unsupported result precision conversion".
    let op_info = if op == OpFunc::Exx2 {
        // BUILD-TIME GUARD (guard-every-crash-at-build-time): the exx2 bind in summeanmaxexx2.ddl
        // declares %type_fp16 ONLY — an fp32 exx2 does not exist on the device, and emitting one
        // would fail at dxp with an op-mapping refusal that names neither the dtype nor the op.
        // The fp32 rmsnorm island keeps its elementary mean(x²) chain instead.
        if matches!(df, Df::Fp32) {
            return Err(format!(
                "reduce `{accum_name}`: exx2 is fp16-only (summeanmaxexx2.ddl binds %type_fp16 \
                 exclusively) — the fp32 mean(x²) must ride the elementary chain"
            ));
        }
        // The pair `exx2scale` + `useZeroMean` (the RAW 0 — at 1 ddc renames the op to
        // exx2_zeromean, which the runtime smc table cannot resolve; see OpInfo::Exx2Scale's doc).
        OpInfo::Exx2Scale(sen169_bits(scale) as u32)
    } else if matches!(df, Df::Fp32) {
        OpInfo::ReduceScalingFp32(scale.to_bits())
    } else {
        OpInfo::ReduceScaling(sen169_bits(scale) as u32)
    };
    Ok(OpSpec {
        op,
        is_reduction: true,
        iter: plan,
        args,
        op_info,
        tiled_symbols,
        time_tile,
        indirect: None,
        coordinate_mask,
    })
}

/// fp32 REDUCE (the torch-spyre RMSNorm `mean(x²)`): native fp32 SFP reduce over `cols`, keeping all
/// `rows` (NOT matmul — PSUM is fp16-only). `data` must be an fp32 tensor from a `dl16tofp32` convert.
#[allow(clippy::too_many_arguments)]
pub fn assemble_reduce_df(
    op_name: &str,
    op_func: &'static str,
    rows: u32,
    cols: u32,
    data_name: &str,
    accum_name: &str,
    df: Df,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // head_major=TRUE ⇒ rank-3 flat, matching the flat fp32 island (see assemble_pointwise_broadcast_df).
    let op = reduce_opspec_df(
        op_func_from_str(op_func),
        rows,
        cols,
        data_name,
        accum_name,
        true,
        df,
    )
    .unwrap_or_else(|e| panic!("assemble_reduce_df {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_reduce_df {op_name}: {e}"))
}

/// Assemble a complete SuperDSC for ONE REDUCTION over `[rows, cols]` reducing
/// the `cols` axis → `[rows, 1]`: `op_func` ∈ {"sum","max","mean"} (sfp unit),
/// via the typed [`reduce_opspec`] builder + `emit_sdsc`.
pub fn assemble_reduce<D: KindTag, A: KindTag>(
    op_name: &str,
    op_func: &'static str,
    rows: u32,
    cols: u32,
    data: &Stk<D>,
    accum: &Stk<A>,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let mut sym_id_base: i64 = 0;
    assemble_reduce_seeded(
        op_name,
        op_func,
        rows,
        cols,
        data,
        accum,
        &mut sym_id_base,
        layout,
    )
}

/// [`assemble_reduce`] with an explicit running symbol-id counter (risk #1).
#[allow(clippy::too_many_arguments)]
pub fn assemble_reduce_seeded<D: KindTag, A: KindTag>(
    op_name: &str,
    op_func: &'static str,
    rows: u32,
    cols: u32,
    data: &Stk<D>,
    accum: &Stk<A>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // DATA's kind now DRIVES head_major (was hardcoded `false`, forcing every caller's data operand
    // through the RowBlocked/stickmajor-eligible path regardless of what actually consumes it next).
    // The stick-major optimization exists ONLY so a whole-tensor pointwise producer's output agrees
    // with a SUBSEQUENT STICK-MAJOR-READING MATMUL consumer (see pointwise.rs's own doc) — it was never
    // meant to apply to a tensor whose ONLY consumer is a reduce. RmsNorm's `sq16` (x²) is exactly that
    // case: its sole consumer is THIS mean-reduce, never a matmul, yet it was picking up the
    // stickmajor rank-2 form (rows>1 && cols>64) because `assemble_reduce_seeded` always told
    // `reduce_opspec` `head_major=false`, keeping the door open for it. A real pod crash (fp8-dynamic
    // granite prefill: RmsNorm's mean(x²) producing inf from a perfectly finite embedding) traced to
    // this — the reduce's data-side coordInfo, walking the shared rank-2 `primaryDsInfo`, does not
    // correctly reduce a stick-major activation along `cols` (sums across the WRONG rows; the native
    // reduce mechanism itself is fine, matching torch-spyre's own `torch.mean` — this was scratchy's own
    // addressing gap for that specific rank-2 combination). Callers that pass `Stk<FlatTag>` (e.g.
    // rmsnorm's `sq16`) now correctly force `head_major=true`, keeping data flat/rank-3; existing
    // `Stk<RowBlockedTag>` callers (fp8 amax, etc.) are UNCHANGED (`RowBlockedTag::kind() !=
    // StickKind::Flat`, so `head_major` still resolves to `false` exactly as before — byte-identical).
    let head_major = D::kind() == StickKind::Flat;
    let op = reduce_opspec(
        op_func_from_str(op_func),
        rows,
        cols,
        data.name(),
        accum.name(),
        head_major,
    )
    .unwrap_or_else(|e| panic!("assemble_reduce {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_reduce {op_name}: {e}"))
}

/// ⭐⭐⭐⭐⭐ WALL 12 — the NONSTICK reduce: a leading-axis reduce `[reduced_rows, survivor_cols] →
/// [survivor_cols]`, the DDL's own nonstick family (`summeanmaxexx2.ddl:39/:45`, `sumnonstick` /
/// `maxnonstick`). The STICK form above reduces the innermost stick axis (one value per row);
/// this form reduces the OUTER axis, so the SURVIVORS are the lanes — the accumulator is the
/// survivor vector, not one stick per row. Row-major flat is `reduced · survivor_cols + survivor`
/// exactly: survivors stride 1, the reduced axis slow, ZERO data movement — which is what makes a
/// leading prefix the nonstick form's own geometry (the door's `reduce_axis_block` law).
///
/// # THE TYPED SURFACE (what matchDdl2Dsc binds, and why it cannot mis-walk)
///
/// * **ItDims `[out, y]`** — `out` = the reduced extent `{is_reduction: true, is_stick: false}`
///   (the FIRST reduction-non-stick ItDim in this crate; the DDL's `%reduce_dim` "OUT dimension"
///   reduced through a non-stick lane), `y` = the survivors `{is_stick: true}`. `y` is LAST in
///   the dim vector, so the survivor stick is the INNERMOST loop dim — the vendor fixture's own
///   nest (`sdsc_maxnonstick.json`: `x → mb → i → out`, survivor stick innermost) — by the
///   tiler's dim order, not by a hand-written walk.
/// * **data `[Active, Active]` R2, accum `[RedNonStick, Active]` R2** over
///   `layoutDimOrder ["out", "y"]`, stick `y` — the vendor scale law
///   (`sdsc_maxnonstick.json:711/:738`: data `scale_ [1,1,1,1]`, accum `[1,-1,1,1]`, the −1 on
///   the REDUCED dim) with the survivor stick carrying `Active` (the `MaterializedStick`
///   invariant `TensorArg::new` enforces). The DDL's dim↔symbol binding is a CONSTRAINT SOLVER
///   over these scales (`ddl_conversion.cpp:2360-2460`): `%outer_dim#0` — the SURVIVOR lane axis
///   — binds to the stick dim `y`, and the reduce symbols bind to the input-only reduced `out`.
///   This is the same solver path topk ships; there is no declared walk to get wrong.
/// * **`out` NEVER core-split** — `distribute_cores` splits output dims only and never a
///   reduction dim, so the cores split the SURVIVOR stick `y` by stick count (the fixture's own
///   division: `numWkSlicesPerDim_ {"mb": 8}`, the reduced `i` unsplit). The reduced axis is the
///   accumulation axis; splitting it would need the psum ring, which the fixture proves the
///   vendor's own nonstick emission does not bind (ringless).
/// * **NO coordinate mask** — the door's `nonstick_geometry` law admits this form only at a
///   whole-stick survivor width with `reduced_rows · survivor_cols == in_numel` EXACTLY, so no
///   pad lane exists anywhere in the op (contrast [`reduce_coordinate_mask`], the STICK form's
///   padded-width answer).
/// * **NO SFP REDUCE mode** — the DDL's SFP REDUCE modes and the PE `nfwd` block gate on the
///   STICK binds directly (`summeanmaxexx2.ddl:485-517`, `:351-438`), not on the
///   `condition_or` unions; the nonstick form's reduced axis is an outer dim the loop nest
///   walks, its lanes hold different survivors, and the per-lane PE accumulators are final
///   after the nest. What serves both forms — subchunk accumulation, the across-subchunk
///   combine, the psum ring arms — is union-gated and entered through the same
///   `genericpartialreduction` bind the stick form uses.
/// * **interimLabeledDs []** — the DDL's nonstick binds carry no interim tensor (the stick
///   `exx2` does, `%output_tensor_x2`); the vendor fixture states the empty list verbatim
///   (`sdsc_maxnonstick.json:775`), and this crate's wire emission derives the empty interim
///   list from the operand list the same way it does for `sum`/`max`.
///
/// # ⭐⭐⭐⭐⭐ WALL 13 — THE FP32 TWIN (the df-true stick basis)
///
/// The whole typed surface above is df-parameterized: at `Df::Fp32` the survivor stick is 32
/// lanes (a 128-byte stick is 64 f16 lanes OR 32 f32 lanes), the dims/`TileOp`/operand
/// `dataFormat`s carry IEEE_FP32, and the op rides the vendor's OWN fp32 nonstick family —
/// `summeanmaxexx2_fp32.ddl` binds `sumnonstick` (`:40`), `meannonstick` (`:43`),
/// `maxnonstick` (`:46`), `minnonstick` (`:49`), `absmaxnonstick` (`:55`), `prodnonstick`
/// (`:57`) at `%type_fp32`, on the SAME nonstick layouts as the f16 template (only the type
/// differs; the dataflow reads the nonstick input through the same lxlu→sfp transfer, `:232-248`).
/// This is the online-softmax `m`/`l` merge's own geometry: 32 survivors = EXACTLY one whole
/// fp32 stick at B=1 (`S = B·NKVH·GQA = 32·B`, whole sticks at every admissible batch width),
/// with the merge input f32 in the kernel itself (`m` is `tl.float32`; the f16 cast is after
/// the merge) — ZERO data movement, the `[J,32]` f32 row-major IS the nonstick layout.
///
/// ⛔ NO `row_blocked` on the fp32 NONSTICK form (the addendum law, vendor-verified): the fp32
/// template's nonstick binds are layout-identical to the f16 family's — `row_blocked` is a
/// property of the fp32 STICK form only (`reduce_opspec_df`'s `with_row_blocked`). The scaling
/// const DOES follow the fp32 law: the const's element type MUST match the op's data_format
/// (the reduce.rs:400-405 law — a SEN169_FP16 const feeding an fp32 op is a MIXED [fp16,fp32]
/// op → DD2 "Unsupported result precision conversion"), so the fp32 nonstick ships
/// `OpInfo::ReduceScalingFp32` with raw IEEE_FP32 bits, exactly as the fp32 stick form does.
///
/// # TRAP (e) — WHY THIS CANNOT REPRODUCE THE OLD HAND-WRITTEN CARD FAILURE
///
/// `attn.rs:2771-2790` records a prior hand-written `maxnonstick` that failed on card
/// ("walked only part of each core's rows"): those row-major operands carried a DECLARED WALK
/// hand-picked beside the layout. This path declares NO walk — the `TensorArg` dims/scales ARE
/// the layout, the DDL solver binds the symbols to them, and the agreement asserts in
/// [`assemble_reduce_nonstick`] bind the operand handles to the op's own typed geometry, so a
/// mismatch is a named refusal at assembly (a `cargo build` failure), never a mis-walk.
///
/// # THE TIME-TILE LADDER IS THE STICK LADDER — A NAMED REFUSAL, NOT A MIS-WALK
///
/// `time_tile_for_lx` searches divisors of the tiled dim's STICK count. This op's tiled dim is
/// the reduced `out` (the template's own reduce-dim chunk trips — `summeanmaxexx2.ddl:159-196`
/// reads the HBM accumulator only on second-and-later reduce-dim chunk trips), which is
/// NON-STICK: a shape whose per-core residency overflows LX at a non-stick-multiple reduced
/// extent refuses at the stick check with the dim named — fail-closed. The template's own
/// reduce-dim subchunk cap (`:94`, max 64 over the reduce dims) is the future ladder; wall 12
/// fires on no such shape (the epilogue merges are stick-wide or sub-stick), and a refusal is
/// never a silent wrong answer.
///
/// **Name collision, stated**: this crate's reduce vocabulary names the REDUCED dim `out` (the
/// DDL's own `%reduce_dim` "OUT dimension"); the vendor FIXTURE's `out` is its SURVIVOR stick.
/// The names never reach matchDdl2Dsc (it binds by layout and `opFuncName`, not by dim name) —
/// the roles are what bind, and they are pinned by the scales above.
pub fn reduce_nonstick_opspec(
    op: OpFunc,
    reduced_rows: u32,
    survivor_cols: u32,
    df: Df,
    data_name: &str,
    accum_name: &str,
) -> Result<OpSpec, String> {
    // The survivor width is a whole stick AT THE DATA'S OWN df BY CONSTRUCTION (the door's
    // `nonstick_geometry` law, df-parameterized since wall 13, admits nothing else);
    // `assert_df_stick_multiple` re-states it through the SAME typed df-dispatched witness the
    // spatial split uses, so a direct caller cannot smuggle a sub-stick width past this fn.
    assert_df_stick_multiple(survivor_cols, df)
        .map_err(|e| format!("reduce `{accum_name}` (nonstick): {e}"))?;
    let dims = vec![
        crate::superdsc_opspec::ItDim {
            name: "out",
            size: reduced_rows,
            is_reduction: true,
            is_stick: false,
            df,
        },
        crate::superdsc_opspec::ItDim {
            name: "y",
            size: survivor_cols,
            is_reduction: false,
            is_stick: true,
            df,
        },
    ];
    let tile_op = TileOp {
        kind: TileOpKind::NonstickReduce,
        dims: dims.clone(),
        df,
    };
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, distribute_cores, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["out", "y"]);
    let data = AnyTensorArg::R2(
        TensorArg::<2>::new(
            true,
            data_name.to_string(),
            Role::Output,
            [Scale::Active, Scale::Active],
            device_dims,
            ["out", "y"],
            "y",
            Allocation::Hbm,
        )
        .map_err(|e| e.0)?
        // The operand's OWN df — at Fp32 the stick is 32 lanes, wordLength 4, dataFormat
        // IEEE_FP32 (the same per-operand law `reduce_opspec_df` states; NO row_blocked here,
        // the nonstick form's own law in this doc's wall-13 section).
        .with_df(df),
    );
    let accum = AnyTensorArg::R2(
        TensorArg::<2>::new(
            false,
            accum_name.to_string(),
            Role::Output,
            // The REDUCED `out` carries RedNonStick (the phantom −1 — legal on a NON-stick dim);
            // the survivor stick `y` carries Active, the MaterializedStick invariant.
            [Scale::RedNonStick, Scale::Active],
            device_dims,
            ["out", "y"],
            "y",
            Allocation::Hbm,
        )
        .map_err(|e| e.0)?
        .with_df(df),
    );
    let args = vec![data, accum];
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    // The stick form's own sum/max const: `scaling_factor` at 1.0. The DDL gates the scaling
    // transfer on the MEAN branch only (`summeanmaxexx2.ddl:133-137`), so sum/max never read
    // it — the stick form ships this exact `OpInfo` card-proven (PR #813's five `tt.reduce`
    // maxima), and the nonstick binds ride the same family machinery. At Fp32 the const is
    // IEEE_FP32 bits (the reduce.rs:400-405 law in this file: the const's element type MUST
    // match the op's data_format, else the op is MIXED [fp16,fp32] → DD2 "Unsupported result
    // precision conversion").
    let op_info = if matches!(df, Df::Fp32) {
        OpInfo::ReduceScalingFp32(1.0f32.to_bits())
    } else {
        OpInfo::ReduceScaling(sen169_bits(1.0f32) as u32)
    };
    Ok(OpSpec {
        op,
        is_reduction: true,
        iter: plan,
        args,
        op_info,
        tiled_symbols,
        time_tile,
        indirect: None,
        // None BY LAW (see the doc above): the geometry admits no pad lane anywhere.
        coordinate_mask: None,
    })
}

/// [`reduce_nonstick_opspec`]'s assemble wrapper — the names-form twin of
/// [`assemble_reduce_seeded`], carrying the AGREEMENT-ASSERT that binds the operand handles to
/// the op's typed geometry: the data handle's declared element count must be
/// `reduced_rows · survivor_cols` EXACTLY and the accumulator's `survivor_cols` — the same
/// frame-invariant numel law the door threaded (wall 11), so a handle minted from any other
/// frame's rows/cols is a named refusal here, never a descriptor that walks the wrong bytes.
#[allow(clippy::too_many_arguments)]
pub fn assemble_reduce_nonstick<D: KindTag, A: KindTag>(
    op_name: &str,
    op_func: &'static str,
    reduced_rows: u32,
    survivor_cols: u32,
    df: Df,
    data: &Stk<D>,
    accum: &Stk<A>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let data_l = data.layout();
    let data_numel = data_l.rows as u64 * data_l.cols as u64;
    assert_eq!(
        data_numel,
        reduced_rows as u64 * survivor_cols as u64,
        "{op_name}: the data handle '{}' declares {data_numel} elements but the nonstick \
         geometry states {reduced_rows}·{survivor_cols} — the handle and the geometry must \
         come from the SAME law (the op's own numels, frame-invariant)",
        data.name(),
    );
    let accum_l = accum.layout();
    let accum_numel = accum_l.rows as u64 * accum_l.cols as u64;
    assert_eq!(
        accum_numel,
        survivor_cols as u64,
        "{op_name}: the accumulator handle '{}' declares {accum_numel} elements but the \
         nonstick form writes the {survivor_cols}-element survivor vector — one lane per \
         survivor, not one stick per row",
        accum.name(),
    );
    let op = reduce_nonstick_opspec(
        op_func_from_str(op_func),
        reduced_rows,
        survivor_cols,
        df,
        data.name(),
        accum.name(),
    )
    .unwrap_or_else(|e| panic!("assemble_reduce_nonstick {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_reduce_nonstick {op_name}: {e}"))
}

/// [`reduce_opspec`] with per-operand ELEMENT offsets — for per-head/offset reduce slices.
///
/// # ⛔ THIS DOC'S TWO PREVIOUS ATTRIBUTIONS OF THE reduce-MAX DEFECT WERE BOTH WRONG
///
/// v1 said the on-card reduce-**MAX** returns 0 (the seed) "when `rows>1`" and is correct only at
/// `rows=1`, naming the remedy as splitting the softmax max into nqh single-row reduces. v2
/// corrected that to "the conjunction `rows>1` **AND** a reduced extent past one stick" — and a
/// KTIR-door guard was built on v2 that refused the wide multi-row max by name. Each version had
/// shipping counter-examples against the other's axis (`attn_bmax` reduces MAX at `rows=nqh>1`
/// with width one stick on card at 31 tok/s; `fq_amax_op` reduces MAX over the whole hidden width
/// at `m==1` on card at 41 tok/s) — and both attributed to the REDUCE what the template's own DDL
/// shows was never the reduce's:
///
/// * `summeanmaxexx2.ddl` seeds the identity ITSELF — `ddl.define_constant`s at `:69-72` that
///   the PE accumulator LRFs are initialized from (negInf for max at `:209`/`:235`, zero for sum),
///   with the accumulator's HBM content read only on second-and-later reduce-dim chunk trips
///   (`:159-196`, `:583-585`). "Hands back the seed" was mechanically wrong, and a seed-win
///   reads −65504, not 0.
/// * The one measured diagnostic (`mxp = 0` over `[nqh, cap]`, 2026-07-28) is the ZEROS-READ
///   signature — the unaligned `cap` walking its stick padding, which read a fresh buffer's
///   zeros (`max(−65504, 0, x) = 0`) — and it was measured BEFORE the row-mixing fixes:
///   `wire.rs`'s `group_stride` law records the same-era SUM signature ("summed a single stick
///   across 32 consecutive rows (+ zero padding) instead of one row's 32 sticks → Σx²≈0 →
///   amax≈0"), and that fix landed 2026-09-16 (`d878aceae`), with the `head_major` fix in the
///   same window. The cell carried OUR addressing defect and OUR unmasked padding at once.
///
/// Both causes are closed: the addressing by `group_stride`/`head_major` (sum at
/// `[rows>1, cols>64]` card-proven since — the rmsnorm prefill parity runs; the template's sum
/// and max paths differ only in compute type and internal seed) and the padding by
/// [`reduce_coordinate_mask`] above. The wide multi-row max LOWERS — the door's law doc
/// ([`crate::emit::lower_ktir_to_superdsc::reduce`]) carries the card gate that arbitrates it,
/// because `dxp_standalone` executes no arithmetic. Splitting rows remains wrong for the reason
/// it always was: the reverted per-row experiment bought nothing and costs `rows` descriptors
/// per softmax.
///
/// None of this changes what THIS function is for: per-operand element offsets, so a per-head or
/// otherwise offset slice of a shared buffer can be reduced in place. The offsets have callers of their
/// own (`attn_dp`/`attn_dn`) and are unrelated to the defect history above. Each operand now
/// arrives as ONE [`Win`] minted by [`win_reduce_data`]/[`win_reduce_accum`] — name, offset and
/// extents bound as a single value, checked at the mint — instead of a name and a raw `u32` that
/// had to be re-paired by convention here.
pub fn reduce_opspec_off<D: KindTag, A: KindTag>(
    op: OpFunc,
    rows: u32,
    cols: u32,
    data: Win<'_, D>,
    accum: Win<'_, A>,
) -> Result<OpSpec, String> {
    reduce_opspec_off_divided(ReduceOffSpec {
        op,
        rows,
        cols,
        data,
        accum,
        division: ReduceDivision::Rows,
    })
}

/// ⭐ HOW A SEEDED REDUCE's CORES ARE SPENT — the reduce family's twin of
/// [`crate::work::EwDivision`]. Every reduce takes [`Self::Rows`] unless its caller states otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceDivision {
    /// [`distribute_cores`]: the output dims only (`mb` first); the reduced axis is never split, so
    /// every core reduces its own rows' whole extent. The division every reduce has always had.
    Rows,
    /// ⭐ THE REDUCED AXIS ACROSS THE CORES, rows whole: every core reduces a whole-stick run of every
    /// row, and the op's template rings the per-core partials into the one accum
    /// ([`OpFunc::template_combines_a_core_split_reduction`], which the emitter's shared-output guard
    /// holds the op to). For a stick-major `[rows, cols]` block, core `c` reads ONE contiguous run of
    /// sticks; [`Self::Rows`] hands each core its rows as `cols/64` strided one-row bursts.
    ReducedAxis,
}

/// The parameter list of [`reduce_opspec_off_divided`], carried as a struct. Every field but
/// `division` is [`reduce_opspec_off`]'s argument of the same name, with the same contract.
/// `head_major` is no longer a field: it is DERIVED from the data operand's kind
/// (`D::kind() == StickKind::Flat` — a `Stk<FlatTag>` data input IS a head-major per-head
/// attention reduce), which is the derivation the assemble layer always made; as a raw `bool` it
/// could have disagreed with the handle it arrived beside.
pub struct ReduceOffSpec<'a, D: KindTag, A: KindTag> {
    pub op: OpFunc,
    pub rows: u32,
    pub cols: u32,
    pub data: Win<'a, D>,
    pub accum: Win<'a, A>,
    /// How the cores are spent. [`ReduceDivision::Rows`] is [`reduce_opspec_off`] byte for byte.
    pub division: ReduceDivision,
}

/// [`reduce_opspec_off`] under a STATED [`ReduceDivision`]. A [`ReduceDivision::ReducedAxis`] reduce
/// of an op whose template cannot combine the cores' partials is a build `Err` here, before the
/// emitter's own guard would refuse its shared output.
pub fn reduce_opspec_off_divided<D: KindTag, A: KindTag>(
    r: ReduceOffSpec<'_, D, A>,
) -> Result<OpSpec, String> {
    let ReduceOffSpec {
        op,
        rows,
        cols,
        data,
        accum,
        division,
    } = r;
    // TYPED: `head_major` is DERIVED from the DATA operand's kind (see `ReduceOffSpec`'s doc) —
    // the Win carries the handle, so the kind crosses into this layer by type.
    let head_major = D::kind() == StickKind::Flat;
    let data_name = data.name();
    let accum_name = accum.name();
    let data_off = data.off().into_raw_elems();
    let accum_off = accum.off().into_raw_elems();
    // ── THE AGREEMENT-ASSERT — the mint/guard flip ──
    // The wins are the DEVICE ACCESS windows this descriptor itself declares (what the bake-time
    // footprint guard bills as `materialized_bytes`): data `[rows, stick-padded cols]`, accum
    // `[rows, one stick]`. A win minted at any other extents is a caller whose window and whose
    // iteration dims come from different laws — the exact pairing the raw `(name, u32)` seam used
    // to allow. Refused here, naming the op and both extents.
    let stick_elems = reduce_stick_elems(cols, Df::Fp16)?;
    let eps = <Fp16 as DataFormat>::ELEMS_PER_STICK;
    if data.rows() != rows || data.cols() != stick_elems {
        return Err(format!(
            "reduce `{accum_name}`: the data operand's window is {}×{} but the op sweeps \
             {rows}×{stick_elems} (rows × stick-padded width of logical {cols}) — the window and \
             the iteration dims must come from the SAME law. Mint the data window with \
             `win_reduce_data`, which states the padded extent from the op's own typed dims.",
            data.rows(),
            data.cols(),
        ));
    }
    if accum.rows() != rows || accum.cols() != eps {
        return Err(format!(
            "reduce `{accum_name}`: the accumulator's window is {}×{} but the op writes {rows} \
             rows at ONE stick ({eps} elements) per row — the accumulator is `Scale::RedStick`, \
             one stick per row whatever the trip. Mint it with `win_reduce_accum`.",
            accum.rows(),
            accum.cols(),
        ));
    }
    // LOWER to TileIR: same pattern as `reduce_opspec_df` — the per-head/offset reduce variant
    // over the SAME shared declaration, kind picked off the DDL arity ([`reduce_tile_kind`]).
    let (dims, kind, stick_elems) = reduce_dims_kind(op, rows, cols, Df::Fp16)?;
    // The pad-lane mask (see `reduce_opspec_df`'s identical call): stated first so an
    // unmaskable shape is a build Err, never a silently-wrong descriptor.
    let coordinate_mask = reduce_coordinate_mask(op, cols, stick_elems, Df::Fp16, accum_name)?;
    let tile_op = TileOp {
        kind,
        dims: dims.clone(),
        df: Df::Fp16,
    };
    let tiled = match division {
        ReduceDivision::Rows => tile_op.tile(MaxCores::<MAX_CORES>, distribute_cores, "out"),
        ReduceDivision::ReducedAxis => {
            if !op.template_combines_a_core_split_reduction() {
                return Err(format!(
                    "reduce `{accum_name}` ({}): a split of the reduced axis needs a template that \
                     combines the cores' partials (`OpFunc::template_combines_a_core_split_reduction`), \
                     and this op's does not — 32 cores would write one accum",
                    op.name()
                ));
            }
            tile_op.tile(
                MaxCores::<MAX_CORES>,
                crate::work::distribute_cores_reduced_axis,
                "out",
            )
        }
    }
    .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    // ── STICK-MAJOR RESIDUAL ── flip ONLY the DATA input to rank-2
    // stick-major to match a stick-major producer (see `reduce_opspec`); a per-head OFFSET slice
    // (non-block-aligned data_off/accum_off) is head-structured and stays rank-3. head_major forces
    // rank-3 too.
    // `cols > 64` matches `reduce_opspec`'s identical fix: rank-2 drops `y` from `layoutDimOrder_`
    // entirely, which only needs to happen for the wide (`cols>64`) case this optimization exists
    // for — a real pod crash confirmed a `cols<=64` per-head reduce (attn_dp/attn_dn) breaks
    // otherwise.
    //
    // ⛔ OFFSETS: only a WHOLE-BLOCK shift of this op's own `[rows, cols]` slab keeps rank-2 — the
    // SAME rule `pointwise.rs` documents for its `out_offset`. A RowBlocked stick-group is
    // `rows·64` elements, so a sub-block read at element base `k·(rows·cols)` (a whole number of
    // the op's own blocks, each block a whole number of stick-groups since `cols` is a stick
    // multiple) addresses through the identical law: the full tensor's `(r, k·cols + c)` sits at
    // exactly `k·rows·cols + ((c/64)·rows·64 + r·64 + c%64)`. A NON-block-aligned offset (a flat
    // column slice, a per-head head offset) is NOT a whole-block shift — a column slice of a
    // stick-major tensor is NOT contiguous — so it stays rank-3 flat, exactly as before. Zero is
    // block-aligned, so every existing caller (attn_dp/attn_dn at `DevOff::ZERO`) is byte-identical.
    let block = rows.saturating_mul(cols).max(1);
    let stickmajor = !head_major
        && rows > 1
        && cols > Fp16::ELEMS_PER_STICK
        && data_off.is_multiple_of(block)
        && accum_off.is_multiple_of(block);
    let device_dims_r2 = plan.iter_syms(["mb", "out"]);
    let data = if stickmajor {
        AnyTensorArg::R2(
            TensorArg::<2>::new(
                true,
                data_name.to_string(),
                Role::Output,
                [Scale::Active, Scale::Active],
                device_dims_r2,
                ["mb", "out"],
                "out",
                Allocation::Hbm,
            )
            .map_err(|e| e.0)?
            .with_offset(data_off),
        )
    } else {
        AnyTensorArg::R3(
            TensorArg::<3>::new(
                true,
                data_name.to_string(),
                Role::Output,
                [Scale::Active, Scale::Active, Scale::Active],
                device_dims,
                ["mb", "out", "y"],
                "out",
                Allocation::Hbm,
            )
            .map_err(|e| e.0)?
            .with_offset(data_off),
        )
    };
    // The accum MUST match the DATA's rank (see `reduce_opspec_df`'s identical guard/fix): a rank-2
    // stick-major data with a rank-3 accum is a MIXED-RANK op — the shared `primaryDsInfo` is rank-2
    // `[mb,out]`, so dxp interprets the rank-3 accum against it, searches for a dim it can't find,
    // gets -1, and range-checks it against the 3-dim vector (`vector::_M_range_check
    // __n=18446744073709551615 >= size 3`, dxp out_of_range = a BUILD failure) — this variant
    // (`reduce_opspec_off`, the per-head/offset reduce attn_dp/attn_dn use) hardcoded accum to R3
    // unconditionally, so it broke the instant `stickmajor` could fire (once the `cols>64` fix let it
    // trigger for a real active_cap/mq_pad width) while `reduce_opspec_df` never did.
    let accum = if stickmajor {
        AnyTensorArg::R2(
            TensorArg::<2>::new(
                false,
                accum_name.to_string(),
                Role::Output,
                [Scale::Active, Scale::RedStick],
                device_dims_r2,
                ["mb", "out"],
                "out",
                Allocation::Hbm,
            )
            .map_err(|e| e.0)?
            .with_offset(accum_off),
        )
    } else {
        AnyTensorArg::R3(
            TensorArg::<3>::new(
                false,
                accum_name.to_string(),
                Role::Output,
                [Scale::Active, Scale::RedStick, Scale::Active],
                device_dims,
                ["mb", "out", "y"],
                "out",
                Allocation::Hbm,
            )
            .map_err(|e| e.0)?
            .with_offset(accum_off),
        )
    };
    let args = vec![data, accum];
    // Mirror reduce_opspec_df's build-time guard: a mixed-rank data/accum pair is unconditionally
    // wrong (dxp abort), so catch it here too rather than let a future edit reintroduce it silently.
    let ranks: Vec<usize> = args.iter().map(|a| a.view().layout.len()).collect();
    if let Some(&bad) = ranks.iter().find(|&&r| r != ranks[0]) {
        return Err(format!(
            "reduce `{accum_name}`: operands have MIXED ranks {ranks:?} (rank {} vs {bad}) — data and \
             accum share ONE primaryDsInfo, so every operand MUST be the same rank or dxp aborts \
             out_of_range. Build both from the same rank branch.",
            ranks[0]
        ));
    }
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    let scale = match op {
        OpFunc::Mean => 1.0f32 / cols as f32,
        _ => 1.0f32,
    };
    let scale_bits = sen169_bits(scale) as u32;
    Ok(OpSpec {
        op,
        is_reduction: true,
        iter: plan,
        args,
        op_info: OpInfo::ReduceScaling(scale_bits),
        tiled_symbols,
        time_tile,
        indirect: None,
        coordinate_mask,
    })
}

#[allow(clippy::too_many_arguments)]
// The row/width slots are TYPED: `rows` arrives only through `RowCount`'s named doors (the
// whole-batch `nqh*mq` shared-buffer extent or the per-request `nqh` — a head count, `mq_pad` or a
// lane count no longer type-checks there), and `cols` names which of the numerically-coincident
// widths (one stick / `mq_pad` / head dim) the sweep is. The operands arrive as `Win`s minted by
// [`win_reduce_data`]/[`win_reduce_accum`] — checked where their offsets are BORN — in place of
// the `(Stk, DevOff)` pairs whose name and offset could come from different laws (the #717 class).
pub fn assemble_reduce_off<D: KindTag, A: KindTag>(
    op_name: &str,
    op_func: &'static str,
    rows: crate::sdsc_abstract::RowCount,
    cols: crate::sdsc_abstract::BlockCols,
    data: Win<'_, D>,
    accum: Win<'_, A>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op = reduce_opspec_off(
        op_func_from_str(op_func),
        rows.get(),
        cols.get(),
        data,
        accum,
    )
    .unwrap_or_else(|e| panic!("assemble_reduce_off {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_reduce_off {op_name}: {e}"))
}

/// The parameter list of [`assemble_reduce_off_divided`], carried as a struct (the `PointwiseDivided`
/// shape). Every field but `division` is [`assemble_reduce_off`]'s, with the same contract.
pub struct ReduceDivided<'a, D: KindTag, A: KindTag> {
    pub op_name: &'a str,
    pub op_func: &'static str,
    pub rows: crate::sdsc_abstract::RowCount,
    pub cols: crate::sdsc_abstract::BlockCols,
    pub data: Win<'a, D>,
    pub accum: Win<'a, A>,
    /// How the cores are spent. [`ReduceDivision::Rows`] is [`assemble_reduce_off`] byte for byte.
    pub division: ReduceDivision,
}

/// [`assemble_reduce_off`] under a STATED [`ReduceDivision`] — the decode batch chunk's row max and row
/// sum take [`ReduceDivision::ReducedAxis`] (`attn`'s `ChunkBufs::red_division_of` says why).
pub fn assemble_reduce_off_divided<D: KindTag, A: KindTag>(
    r: ReduceDivided<'_, D, A>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let ReduceDivided {
        op_name,
        op_func,
        rows,
        cols,
        data,
        accum,
        division,
    } = r;
    let op = reduce_opspec_off_divided(ReduceOffSpec {
        op: op_func_from_str(op_func),
        rows: rows.get(),
        cols: cols.get(),
        data,
        accum,
        division,
    })
    .unwrap_or_else(|e| panic!("assemble_reduce_off_divided {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_reduce_off_divided {op_name}: {e}"))
}

#[cfg(test)]
mod seeded_reduce_refusals {
    use super::*;

    /// ⛔ A TIME-TILED SEEDED REDUCE REFUSES AT THE EMISSION — never an emitted op whose
    /// trips alias on one accum stick (the #50 `tiled_trips_alias` collision). With the
    /// partials builder gone, the single-trip row is the only seeded form; a caller whose
    /// row genuinely overruns the usable LX must split the reduce itself (the fp8 qspt
    /// path's per-slice single-shot + combine pattern in `emit/ktir_matmul_fp8.rs`).
    #[test]
    fn a_time_tiled_seeded_reduce_refuses_naming_the_single_trip_law() {
        let op = reduce_opspec_df(OpFunc::Max, 1, 524_288, "w", "j", false, Df::Fp32)
            .expect("the bare opspec assembles");
        assert_eq!(op.time(), 2, "the past-the-budget row genuinely tiles");
        let folds = SdscFoldSet::new(op.iter.cores_used());
        let err = match crate::emit::emit_sdsc_tiled("max32_bare", &op, &folds, &mut 0i64, None) {
            Err(e) => e,
            // `EmittedOp` is not `Debug`, so the Ok arm states its own refusal.
            Ok(_) => panic!("the bare two-trip seeded reduce must refuse at the emission"),
        };
        assert!(
            err.0.contains("not in the partials form"),
            "the refusal names the law: {}",
            err.0
        );
        assert!(
            err.0.contains("single trip"),
            "the refusal names the remedy: {}",
            err.0
        );
    }
}
