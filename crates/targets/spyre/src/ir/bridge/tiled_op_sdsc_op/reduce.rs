//! Bridge 3 (`TiledOp -> SdscOp`), reduce family: the PRIMITIVE (one `TileOp` in, one
//! `OpSpec`/`EmittedOp` out, no internal decomposition) TileIR→SdscOp entry points for stick
//! reduces (sum/mean/max). See `super`'s module doc for why this is its own small file rather
//! than folded into a bigger one.

use crate::ir::island::tile_op::TileOp;
use crate::lower_subtile_tape_to_superdsc::{
    BundleLayout, EmittedOp, distribute_cores, op_func_from_str, sen169_bits,
};
use scratchy_subtile::sdsc_abstract::{KindTag, StickKind, Stk};
use scratchy_subtile::superdsc_opspec::{
    Allocation, AnyTensorArg, DataFormat, Df, Fp16, Fp32, MAX_CORES, MaxCores, OpFunc, OpInfo,
    OpSpec, Role, Scale, SdscFoldSet, StickExtent, TensorArg,
};

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
    let stick_elems = if matches!(df, Df::Fp32) {
        StickExtent::<Fp32>::new(cols)?.elems()
    } else {
        StickExtent::<Fp16>::new(cols)?.elems()
    };
    let dims = vec![
        scratchy_subtile::superdsc_opspec::ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df,
        },
        scratchy_subtile::superdsc_opspec::ItDim {
            name: "out",
            size: stick_elems,
            is_reduction: true,
            is_stick: true,
            df,
        },
        scratchy_subtile::superdsc_opspec::ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df,
        },
    ];
    // LOWER to TileIR: this reduce's iteration domain + kind, ONE `TileOp` declaration instead of
    // the inline `WorkPlan::divide` + hand-picked `pointwise_lx_resident` closure — reached from the
    // REAL `lower_one_node` SubtileTape dispatch (every reduce: rmsnorm's mean, softmax's max/sum,
    // fp8's amax), same as `pointwise_opspec`.
    let tile_op = TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands: 2 },
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
    // fp16). `cols` is the reduced extent N.
    let scale = match op {
        OpFunc::Mean => 1.0f32 / cols as f32,
        _ => 1.0f32, // sum / max: identity scaling
    };
    // The const is read by the device as SEN169_FP16 (1-6-9), NOT IEEE f16 (1-5-10) — see
    // `sen169_bits`'s own doc for why a plain `half::f16` encoding silently mis-scales every reduce.
    // The const's element type MUST match the OP's data_format (torch-spyre
    // `encode_constant(value, data_format)`, compute_ops.py:127): the fp32 reduce (mq>1 rmsnorm
    // mean(x²)) needs an IEEE_FP32 const (raw f32 bits), else a SEN169_FP16 const feeding an fp32 op
    // is a MIXED [fp16,fp32] op → DD2 "Unsupported result precision conversion".
    let op_info = if matches!(df, Df::Fp32) {
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
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
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
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_reduce {op_name}: {e}"))
}

/// [`reduce_opspec`] with per-operand ELEMENT offsets — for PER-HEAD reduces (rows=1).
/// The on-card reduce-**MAX** returns 0 (the seed) when `rows>1` (PROVEN via attn diag:
/// mxp=0 over [nqh,cap]) but is CORRECT at rows=1 (the rmsnorm `rmamax` works). reduce-SUM
/// is fine multi-row (the score reduce gives sane scores), so only the softmax max-reduce
/// needs splitting into nqh single-row reduces.
#[allow(clippy::too_many_arguments)]
pub fn reduce_opspec_off(
    op: OpFunc,
    rows: u32,
    cols: u32,
    data_name: &str,
    accum_name: &str,
    data_off: u32,
    accum_off: u32,
    head_major: bool,
) -> Result<OpSpec, String> {
    let cols_ext = StickExtent::<Fp16>::new(cols)?;
    let dims = vec![
        scratchy_subtile::superdsc_opspec::ItDim {
            name: "mb",
            size: rows,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
        scratchy_subtile::superdsc_opspec::ItDim {
            name: "out",
            size: cols_ext.elems(),
            is_reduction: true,
            is_stick: true,
            df: Df::Fp16,
        },
        scratchy_subtile::superdsc_opspec::ItDim {
            name: "y",
            size: 1,
            is_reduction: false,
            is_stick: false,
            df: Df::Fp16,
        },
    ];
    // LOWER to TileIR: same pattern as reduce_opspec_df — the per-head/offset reduce variant.
    let tile_op = TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::PointwiseOrReduce { n_operands: 2 },
        dims: dims.clone(),
        df: Df::Fp16,
    };
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, distribute_cores, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    // ── STICK-MAJOR RESIDUAL ── flip ONLY the DATA input to rank-2
    // stick-major to match a stick-major producer (see `reduce_opspec`); a per-head OFFSET slice
    // (data_off/accum_off ≠ 0) is head-structured and stays rank-3. head_major forces rank-3 too.
    // `cols > 64` matches `reduce_opspec`'s identical fix: rank-2 drops `y` from `layoutDimOrder_`
    // entirely, which only needs to happen for the wide (`cols>64`) case this optimization exists
    // for — a real pod crash confirmed a `cols<=64` per-head reduce (attn_dp/attn_dn) breaks
    // otherwise.
    let stickmajor =
        !head_major && rows > 1 && cols > Fp16::ELEMS_PER_STICK && data_off == 0 && accum_off == 0;
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
    })
}

#[allow(clippy::too_many_arguments)]
// The row/width slots are TYPED: `rows` arrives only through `RowCount`'s named doors (the
// whole-batch `nqh*mq` shared-buffer extent or the per-request `nqh` — a head count, `mq_pad` or a
// lane count no longer type-checks there), and `cols` names which of the numerically-coincident
// widths (one stick / `mq_pad` / head dim) the sweep is.
pub fn assemble_reduce_off<D: KindTag, A: KindTag>(
    op_name: &str,
    op_func: &'static str,
    rows: scratchy_subtile::sdsc_abstract::RowCount,
    cols: scratchy_subtile::sdsc_abstract::BlockCols,
    data: &Stk<D>,
    data_off: scratchy_subtile::addr::DevOff,
    accum: &Stk<A>,
    accum_off: scratchy_subtile::addr::DevOff,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let rows = rows.get();
    let cols = cols.get();
    let data_off = data_off.into_raw_elems();
    let accum_off = accum_off.into_raw_elems();

    // TYPED: `head_major` is DERIVED from the DATA input's kind — a `Stk<FlatTag>` data input IS a
    // head-major (per-head attention) reduce; a `Stk<RowBlockedTag>` data is the residual stream. Both
    // operands carry only `.name()`, so the emit is byte-identical.
    let head_major = D::kind() == StickKind::Flat;
    let op = reduce_opspec_off(
        op_func_from_str(op_func),
        rows,
        cols,
        data.name(),
        accum.name(),
        data_off,
        accum_off,
        head_major,
    )
    .unwrap_or_else(|e| panic!("assemble_reduce_off {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_reduce_off {op_name}: {e}"))
}
