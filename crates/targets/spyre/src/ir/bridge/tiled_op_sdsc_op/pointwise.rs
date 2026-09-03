//! Bridge 3 (`TiledOp -> SdscOp`), pointwise/broadcast family: the PRIMITIVE (one `TileOp` in, one
//! `OpSpec`/`EmittedOp` out, no internal decomposition) TileIR→SdscOp entry points for elementwise
//! and broadcast pointwise ops. See `super`'s module doc for why this is its own small file rather
//! than folded into a bigger one.

use crate::ir::island::tile_op::TileOp;
use crate::lower_subtile_tape_to_superdsc::{
    BundleLayout, EmittedOp, EwOperand, distribute_cores, op_func_from_str,
};
use scratchy_subtile::sdsc_abstract::{KindTag, StickKind, Stk};
use scratchy_subtile::superdsc_opspec::{
    Allocation, AnyTensorArg, DataFormat, Df, Fp16, MAX_CORES, MaxCores, OpFunc, OpInfo, OpSpec,
    Role, Scale, SdscFoldSet, TensorArg,
};

pub fn pointwise_opspec_from_tile(
    tile_op: &TileOp,
    op: OpFunc,
    in_names: &[&str],
    o_name: &str,
    head_major: bool,
) -> Result<OpSpec, String> {
    let rows = tile_op
        .dims
        .iter()
        .find(|d| d.name == "mb")
        .map(|d| d.size)
        .unwrap_or(1);
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, distribute_cores, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    // ── STICK-MAJOR RESIDUAL ── a whole-tensor residual pointwise op writes FLAT (rank-3
    // [mb,out,y=1]) but the consuming matmul reads STICK-MAJOR (rank-2 [mb,out]); byte-identical at
    // rows==1, scrambled at rows>1 AND cols>64. Present rank-2 [mb,out] so producer + consumer
    // agree. head_major (per-head attn ops) stays rank-3.
    let stickmajor = !head_major && rows > 1;
    let device_dims_r2 = plan.iter_syms(["mb", "out"]);

    // rank-2 stick-major (`stickmajor`) or the legacy rank-3 flat operand (drops the size-1 `y`).
    let mk = |is_input: bool, name: &str| -> Result<AnyTensorArg, String> {
        if stickmajor {
            Ok(AnyTensorArg::R2(
                TensorArg::<2>::new(
                    is_input,
                    name.to_string(),
                    Role::Output,
                    [Scale::Active, Scale::Active],
                    device_dims_r2,
                    ["mb", "out"],
                    "out",
                    Allocation::Hbm,
                )
                .map_err(|e| e.0)?,
            ))
        } else {
            Ok(AnyTensorArg::R3(
                TensorArg::<3>::new(
                    is_input,
                    name.to_string(),
                    Role::Output,
                    [Scale::Active, Scale::Active, Scale::Active],
                    device_dims,
                    ["mb", "out", "y"],
                    "out",
                    Allocation::Hbm,
                )
                .map_err(|e| e.0)?,
            ))
        }
    };
    let mut args = Vec::with_capacity(in_names.len() + 1);
    for name in in_names {
        args.push(mk(true, name)?);
    }
    args.push(mk(false, o_name)?);

    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    Ok(OpSpec {
        op,
        is_reduction: false,
        iter: plan,
        args,
        op_info: if op.needs_sfp_const_table() {
            OpInfo::SfpConstTable
        } else {
            OpInfo::None
        },
        tiled_symbols,
        time_tile,
    })
}

/// TileIR → SdscOp for the [`EwOperand`]-broadcast pointwise family: takes an ALREADY-LOWERED
/// `TileOp` (from `pointwise_broadcast_opspec_df` in the live emitter, or directly from a live
/// per-node lowering via `node_to_single_tile_op`) and runs the tiler + TensorArg assembly.
/// `rows`/`cols` stay EXPLICIT parameters (not re-derived from `tile_op.dims`) because this
/// builder's `stickmajor`/block-alignment logic needs the untouched `rows·cols` product, not the
/// stick-rounded `out` extent `TileOp` carries.
#[allow(clippy::too_many_arguments)]
pub fn pointwise_broadcast_opspec_from_tile(
    tile_op: &TileOp,
    rows: u32,
    cols: u32,
    op: OpFunc,
    inputs: &[EwOperand<'_>],
    o_name: &str,
    out_offset: u32,
    head_major: bool,
) -> Result<OpSpec, String> {
    let df = tile_op.df;
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, distribute_cores, "out")
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    let device_dims = plan.iter_syms(["mb", "out", "y"]);
    // ── STICK-MAJOR RESIDUAL + PER-HEAD ATTN ── The m>1 seam fix: a residual whole-tensor
    // pointwise op writes FLAT (rank-3 [mb,out,y=1]) but the consuming matmul reads STICK-MAJOR
    // (rank-2 [mb,out]); byte-identical at rows==1, scrambled at rows>1 AND cols>64. Present the op
    // as rank-2 [mb,out] (stick-major) so producer + consumer agree. head_major stays rank-3.
    //
    // BLOCK-ALIGNED per-head offsets also qualify: a per-head attn softmax op processes head h's
    // whole [rows,cols] SUB-BLOCK at base offset h·(rows·cols) — a base shift of a STANDALONE
    // stick-major [rows,cols] tensor (`with_offset` bumps the flat base, `per_core_addr` then
    // addresses the block stick-major), byte-identical to the score matmul that WROTE the block
    // per-head. So an offset that is an exact multiple of rows·cols is stick-major-compatible. A
    // per-row-scalar (out-broadcast) operand collapses its column addressing to c=0, so its own
    // per-head row-block offset (e.g. rc's h·rows·stick) is compatible regardless of alignment to
    // rows·cols. A NON-block-aligned COLUMN slice (RoPE rotate-half `half`; an MLP column chunk
    // `cols_start`) is NOT a whole-block shift — a column slice of a stick-major tensor is NOT
    // contiguous — so it stays rank-3 flat.
    let block = (rows as u64) * (cols as u64);
    let block_aligned = |off: u32| block != 0 && (off as u64).is_multiple_of(block);
    // `cols > 64` (one fp16 stick) is the OTHER half of this fn's own documented precondition —
    // the comment above says outright "byte-identical at rows==1, scrambled at rows>1 AND
    // cols>64" — but the condition below never checked `cols`, so it dropped to rank-2 for EVERY
    // rows>1 op regardless of width, including `cols<=64` ops the flat/rank-3 form was already
    // correct for. That rank-2 form OMITS the `y` dim from `layoutDimOrder_` entirely (not just
    // sizes it 1) — confirmed against a real pod dxp_standalone crash (`vector::_M_range_check`,
    // fp8-dynamic granite prefill's `attn_onorm`/`attn_sp`, both `cols=head_dim=64`): dxp's own
    // dim-handling (`designSpaceConfig.cpp`/`dsc2.cpp` in the dxp source, `dxp-rs/deeptools`)
    // consistently assumes a 3-axis (mb/out/y) structure elsewhere, and a rank-2 op whose layout
    // genuinely lacks `y` is exactly what breaks that assumption. Restricting stickmajor to
    // `cols > 64` — the ONLY regime the introducing comment claims needs it — leaves every
    // `cols<=64` op (every per-head attention op here) at the already-proven rank-3 form, while
    // preserving the real fix for wide (`cols>64`) ops it was actually written for.
    let stickmajor = !head_major
        && rows > 1
        && cols > Fp16::ELEMS_PER_STICK
        && block_aligned(out_offset)
        && inputs
            .iter()
            .all(|i| i.out_broadcast() || block_aligned(i.col_offset()));
    let device_dims_r2 = plan.iter_syms(["mb", "out"]);

    if time_tile.is_some() && inputs.iter().any(|i| i.out_broadcast()) {
        return Err(format!(
            "pointwise_broadcast '{o_name}': a broadcast-over-`out` operand in a TIME-TILED op \
             is not yet supported — concrete_trips would wrongly advance the broadcast operand's \
             address per trip (silently-wrong). Refusing (build guard). [rows={rows} cols={cols}]"
        ));
    }

    // Build a rank-2 stick-major (`stickmajor`) or the legacy rank-3 flat operand. rank-2 drops the
    // size-1 `y` dim + its scale so `for_view_df` classifies it RowBlocked (stick-major), matching
    // the matmul I/O.
    let mk =
        |is_input: bool, name: &str, scale: [Scale; 3], off: u32| -> Result<AnyTensorArg, String> {
            if stickmajor {
                Ok(AnyTensorArg::R2(
                    TensorArg::<2>::new(
                        is_input,
                        name.to_string(),
                        Role::Output,
                        [scale[0], scale[1]],
                        device_dims_r2,
                        ["mb", "out"],
                        "out",
                        Allocation::Hbm,
                    )
                    .map_err(|e| e.0)?
                    .with_offset(off)
                    .with_df(df)
                    .with_row_blocked(matches!(df, Df::Fp32)),
                ))
            } else {
                Ok(AnyTensorArg::R3(
                    TensorArg::<3>::new(
                        is_input,
                        name.to_string(),
                        Role::Output,
                        scale,
                        device_dims,
                        ["mb", "out", "y"],
                        "out",
                        Allocation::Hbm,
                    )
                    .map_err(|e| e.0)?
                    .with_offset(off)
                    .with_df(df)
                    .with_row_blocked(matches!(df, Df::Fp32)),
                ))
            }
        };
    let mut args = Vec::with_capacity(inputs.len() + 1);
    for inp in inputs {
        args.push(mk(true, inp.name(), inp.scale(), inp.col_offset())?);
    }
    // The OUTPUT may write to a COLUMN SLICE of its tensor (`out_offset` cols in →
    // `out_offset·2` byte offset): RoPE's rotate-half writes each head-half to a
    // distinct half of the roped output. `per_core_addr` adds this on top of the
    // global placement (task #55). 0 = whole-tensor output (every existing caller;
    // and the `stickmajor` predicate requires out_offset==0, so rank-2 is whole-tensor).
    args.push(mk(
        false,
        o_name,
        [Scale::Active, Scale::Active, Scale::Active],
        out_offset,
    )?);
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    Ok(OpSpec {
        op,
        is_reduction: false,
        iter: plan,
        args,
        op_info: if op.needs_sfp_const_table() {
            OpInfo::SfpConstTable
        } else {
            OpInfo::None
        },
        tiled_symbols,
        time_tile,
    })
}

/// `assemble_pointwise_broadcast_off` TAKING AN ALREADY-LOWERED `TileOp` — the TileIR→SdscOp entry
/// point for a live per-node caller that built its `TileOp` via `node_to_single_tile_op` (straight
/// off the `SubtileNode`) instead of re-deriving `rows`/`cols` first. `rows`/`cols` stay explicit
/// params (needed for the SFP-transcendental guard and the stickmajor block-alignment calc — see
/// [`pointwise_broadcast_opspec_from_tile`]'s doc for why they are not re-derived from
/// `tile_op.dims`).
#[allow(clippy::too_many_arguments)]
pub fn assemble_pointwise_broadcast_off_from_tile<O: KindTag>(
    op_name: &str,
    tile_op: &TileOp,
    op_func: &'static str,
    rows: u32,
    cols: u32,
    inputs: &[EwOperand<'_>],
    o: &Stk<O>,
    out_offset: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let o_name = o.name();
    let head_major = O::kind() == StickKind::Flat;
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
        && inputs.iter().any(|i| i.out_broadcast())
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
    let op = pointwise_broadcast_opspec_from_tile(
        tile_op,
        rows,
        cols,
        op_func_from_str(op_func),
        inputs,
        o_name,
        out_offset,
        head_major,
    )
    .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_pointwise_broadcast {op_name}: {e}"))
}

/// `assemble_pointwise_seeded` TAKING AN ALREADY-LOWERED `TileOp` instead of raw `rows`/`cols` — the
/// real TileIR→SdscOp entry point for the live per-node path: a caller that already built its
/// `TileOp` via [`crate::subtile_tape_to_tile_ir::node_to_single_tile_op`] (straight off the
/// `SubtileNode`, no re-derivation of shape) hands it here instead of unpacking it back into
/// `rows`/`cols` first.
pub fn assemble_pointwise_seeded_from_tile(
    op_name: &str,
    tile_op: &TileOp,
    op_func: &'static str,
    in_names: &[&str],
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op =
        pointwise_opspec_from_tile(tile_op, op_func_from_str(op_func), in_names, o_name, false)
            .unwrap_or_else(|e| panic!("assemble_pointwise {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_pointwise {op_name}: {e}"))
}
