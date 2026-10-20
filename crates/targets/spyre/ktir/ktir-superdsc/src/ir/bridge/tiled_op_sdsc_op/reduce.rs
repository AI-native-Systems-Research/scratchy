//! Bridge 3 (`TiledOp -> SdscOp`), reduce family: the PRIMITIVE (one `TileOp` in, one
//! `OpSpec`/`EmittedOp` out, no internal decomposition) TileIR→SdscOp entry points for stick
//! reduces (sum/mean/max). See `super`'s module doc for why this is its own small file rather
//! than folded into a bigger one.

use crate::emit::{EmittedOp, op_func_from_str, sen169_bits};
use crate::ir::island::tile_op::{TileOp, TileOpKind};
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{KindTag, StickKind, Stk};
use crate::superdsc_opspec::{
    Allocation, AnyTensorArg, DataFormat, Df, Fp16, Fp32, MAX_CORES, MaxCores, OpFunc, OpInfo,
    OpSpec, Role, SamvCoordinateMask, Scale, SdscFoldSet, TensorArg,
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

/// [`reduce_opspec`] with per-operand ELEMENT offsets — for per-head/offset reduce slices.
///
/// # ⛔⛔⛔ THE reduce-MAX DEFECT THIS DOC USED TO MISSTATE: IT IS THE CONJUNCTION
///
/// This doc previously said the on-card reduce-**MAX** returns 0 (the seed) "when `rows>1`" and is
/// correct only at `rows=1`, and named the remedy as splitting the softmax max into nqh single-row
/// reduces. **THAT ATTRIBUTED THE FAILURE TO THE WRONG AXIS, and it was believed downstream** — a
/// KTIR-door guard was built on this sentence rather than on the code that had already corrected it,
/// and it refused a CORRECT decoder softmax by name.
///
/// The correction, which is measurable in this tree and not a matter of opinion: the broken cell is
/// `rows > 1` **AND** a reduced extent past ONE STICK. Exactly one diagnostic ever measured the defect
/// — `mxp = 0` over `[nqh, cap]` — and that shape moved the row count AND the width off their safe
/// values TOGETHER (rows = nqh > 1, cols = cap = many sticks), so on its own it cannot say which axis
/// broke it. Each single-axis reading is refuted by a different piece of shipping, hardware-proven
/// emission:
///
/// * **NOT `rows > 1`.** [`super::attn`] records a per-row split TRIED AND REVERTED (2026-07-28) after
///   direct comparison against the old proven flash-decode's `attn_bmax{b}_o{t}`: that code reduces MAX
///   at `rows = nqh` (> 1) with `width` one stick, on real hardware, at 31 tok/s — and the reverted
///   split was measured as zero behavioural change. `assemble_attn_block`'s `width` parameter states the
///   safe regime at the parameter itself.
/// * **NOT the width alone.** [`crate::emit::ktir_matmul_fp8`]'s `fq_amax_op` reduces MAX over the FULL
///   hidden width `k` (2048/4096 — many sticks) on every fp8 matmul of every layer; at decode `m == 1`
///   and it is proven on card at 41 tok/s.
///
/// The conjunction is what the two have in common, and it is already named twice as such: the
/// `stickmajor` branch in [`reduce_opspec_df`] and in this function fires at exactly
/// `rows > 1 && cols > 64`, and `ktir_matmul_fp8.rs` calls that regime "scrambled at rows>1 AND
/// cols>64" — which is the clue that the hazard there is ADDRESSING (a stick-major activation read as
/// flat), not reduce-MAX arithmetic as such.
///
/// So the remedy for a genuinely wide max is to TILE THE REDUCTION to one stick and combine the
/// partials (what `attention_flash.py` and `swiglu_mlp.py` pin `BLOCK_N = 64` for, as a correctness
/// constraint), **not** to split it into one reduce per row. reduce-SUM combines correctly multi-row at
/// any width and needs nothing.
///
/// None of this changes what THIS function is for: per-operand element offsets, so a per-head or
/// otherwise offset slice of a shared buffer can be reduced in place. The offsets have callers of their
/// own (`attn_dp`/`attn_dn`) and are unrelated to the defect above.
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
    reduce_opspec_off_divided(ReduceOffSpec {
        op,
        rows,
        cols,
        data_name,
        accum_name,
        data_off,
        accum_off,
        head_major,
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
pub struct ReduceOffSpec<'a> {
    pub op: OpFunc,
    pub rows: u32,
    pub cols: u32,
    pub data_name: &'a str,
    pub accum_name: &'a str,
    pub data_off: u32,
    pub accum_off: u32,
    pub head_major: bool,
    /// How the cores are spent. [`ReduceDivision::Rows`] is [`reduce_opspec_off`] byte for byte.
    pub division: ReduceDivision,
}

/// [`reduce_opspec_off`] under a STATED [`ReduceDivision`]. A [`ReduceDivision::ReducedAxis`] reduce
/// of an op whose template cannot combine the cores' partials is a build `Err` here, before the
/// emitter's own guard would refuse its shared output.
pub fn reduce_opspec_off_divided(r: ReduceOffSpec<'_>) -> Result<OpSpec, String> {
    let ReduceOffSpec {
        op,
        rows,
        cols,
        data_name,
        accum_name,
        data_off,
        accum_off,
        head_major,
        division,
    } = r;
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
// widths (one stick / `mq_pad` / head dim) the sweep is.
pub fn assemble_reduce_off<D: KindTag, A: KindTag>(
    op_name: &str,
    op_func: &'static str,
    rows: crate::sdsc_abstract::RowCount,
    cols: crate::sdsc_abstract::BlockCols,
    data: &Stk<D>,
    data_off: crate::addr::DevOff,
    accum: &Stk<A>,
    accum_off: crate::addr::DevOff,
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
    pub data: &'a Stk<D>,
    pub data_off: crate::addr::DevOff,
    pub accum: &'a Stk<A>,
    pub accum_off: crate::addr::DevOff,
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
        data_off,
        accum,
        accum_off,
        division,
    } = r;
    let head_major = D::kind() == StickKind::Flat;
    let op = reduce_opspec_off_divided(ReduceOffSpec {
        op: op_func_from_str(op_func),
        rows: rows.get(),
        cols: cols.get(),
        data_name: data.name(),
        accum_name: accum.name(),
        data_off: data_off.into_raw_elems(),
        accum_off: accum_off.into_raw_elems(),
        head_major,
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
