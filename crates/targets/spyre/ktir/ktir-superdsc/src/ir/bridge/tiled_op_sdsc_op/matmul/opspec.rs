//! Matmul family: the PRIMITIVE (one `TileOp`/tiling call in, one `OpSpec` out, no internal
//! decomposition) opspec builders. See `super`'s module doc.

use super::dims::{matmul_dims, matmul_split_map, matmul_split_map_for};
use super::walk::{
    InAxis, MatmulWrapperSite, MbAxis, OutAxis, RungRegime, SharedKernelBmmForm,
    SharedKernelBmmRegime, Walk2, Walk3, WalkAxis,
};
use crate::ir::island::tile_op::TileOp;
use crate::sdsc_abstract::{MatK, MatM, MatN, MatY, PhysM};
use crate::superdsc_opspec::{
    Allocation, AnyTensorArg, DataFormat, Df, Fp16, MAX_CORES, MaxCores, OpFunc, OpInfo, OpSpec,
    Role, Scale, StickExtent, TensorArg, WorkPlan,
};

/// Build the typed [`OpSpec`] for ONE matmul `A[m,k]·W[k,n]→O[m,n]` (batch `b`).
/// N (output stick) and K (reduction stick) are wrapped in [`StickExtent`] so a
/// sub-stick extent is a `cargo`-surfaced `Err` (witness (a)); the cost-model
/// split is wrapped in a validated `WorkPlan` (witnesses (b)(f)); the three
/// `TensorArg`s carry rank-pinned scale/dim/layout arrays (witness (d)) whose
/// `IterSym`s come only from the plan (witness (e)).
#[allow(clippy::too_many_arguments)]
pub fn matmul_opspec(
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a_name: &str,
    w_name: &str,
    o_name: &str,
) -> Result<OpSpec, String> {
    // The untyped wrapper boundary: this signature predates the typed run, and its callers are the
    // dense-projection lowerings whose dims are unambiguous token rows / feature widths — none of
    // them a batch of requests, so the proven form is stated here, not defaulted below.
    matmul_opspec_off::<Fp16>(
        MatM::of_token_rows(m),
        MatN::of_out_features(n),
        MatK::of_in_features(k),
        MatY::of_batch(batch),
        SharedKernelBmmForm::batch_inner_proven(MatmulWrapperSite::witness()),
        a_name,
        w_name,
        o_name,
        0,
        0,
        0,
    )
}

/// [`matmul_opspec`] with per-operand ELEMENT start-offsets (`*_off`), bumping each
/// arg's base by `off·wordLength` (the [`TensorArg::with_offset`] slice). The ONLY
/// caller that needs this is the per-GQA-group attention bmm: the AIU `batchmatmul`
/// KERNEL is SHARED across the batch (a 3-D kernel is dxp-"garbage"), so a single
/// `batch=nqh` bmm makes ALL query heads attend with head-0's K/V — WRONG for
/// GQA/MHA. The fix emits one bmm PER kv-head group (`batch=gqa`, the group's q-heads
/// legitimately SHARE their kv-head's K/V), each sliced to the group via `*_off`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_opspec_off<DF: DataFormat>(
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    // WHICH shared-kernel batched form this op takes (walk order + split policy) — threaded from
    // the caller that knows the bundle's row kind; see [`SharedKernelBmmForm`].
    form: SharedKernelBmmForm,
    a_name: &str,
    w_name: &str,
    o_name: &str,
    a_off: u32,
    w_off: u32,
    o_off: u32,
) -> Result<OpSpec, String> {
    // Default work-division = the monolith cost-model split. The Kani-verified tower drives its OWN
    // (`CoreSplit`) via `matmul_opspec_split`, reusing everything below the split.
    matmul_opspec_off_operands::<DF>(
        m,
        n,
        k,
        batch,
        form,
        a_name,
        w_name,
        o_name,
        a_off,
        w_off,
        o_off,
        DF::DF,
    )
}

/// [`matmul_opspec_off`] with an EXPLICIT operand residency, decoupled from the `DF` stick geometry.
/// Only fp8 W8A8 needs this: its geometry is `::<Fp16>` while its operands are 1 byte.
#[allow(clippy::too_many_arguments)]
pub fn matmul_opspec_off_operands<DF: DataFormat>(
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a_name: &str,
    w_name: &str,
    o_name: &str,
    a_off: u32,
    w_off: u32,
    o_off: u32,
    operand_df: Df,
) -> Result<OpSpec, String> {
    matmul_opspec_off_operands_phys::<DF>(
        m, n, k, batch, form, a_name, w_name, o_name, a_off, w_off, o_off, operand_df, None,
    )
}

/// [`matmul_opspec_off_operands`] for an op that sweeps only PART of the activation's rows — see
/// `phys_mb` on [`matmul_opspec_split`].
#[allow(clippy::too_many_arguments)]
pub fn matmul_opspec_off_operands_phys<DF: DataFormat>(
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a_name: &str,
    w_name: &str,
    o_name: &str,
    a_off: u32,
    w_off: u32,
    o_off: u32,
    operand_df: Df,
    phys_mb: Option<PhysM>,
) -> Result<OpSpec, String> {
    // The floor of the typed run: `matmul_opspec_split` is the raw-extent seam the Kani tower also
    // drives, so the quantities give up their names here — after the slots have been filled. The
    // form picks the LIVE splitter in the same breath as it will pick the walks below, so the
    // split policy and the declared order come from one value.
    matmul_opspec_split::<DF, _>(
        m.get(),
        n.get(),
        k.get(),
        batch.get(),
        a_name,
        w_name,
        o_name,
        a_off,
        w_off,
        o_off,
        operand_df,
        phys_mb.map(PhysM::get),
        // No K-trip: every typed door contracts the weight's whole reduction axis.
        None,
        form,
        // The strides ride ON the batch axis, so they arrive here already paired with the `y` extent
        // they describe — there is no slot for a caller to fill one without the other.
        batch.batch_strides(),
        matmul_split_map_for(form),
    )
}

/// [`matmul_opspec_off`] with an INJECTABLE work-division `splitter` — the SEAM the Kani-verified tower
/// (`scratchy-sdsc`) drives: it passes a `CoreSplit`-derived splitter so the emitted SDSC uses the PROVEN
/// 32-core partition (disjoint+covering, #50-free by proof), while this fn keeps owning the ABI OpSpec /
/// TensorArg assembly. `splitter` matches `WorkPlan::divide`'s: `(dims, max_cores) -> {name → >1 split}`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_opspec_split<DF: DataFormat, S>(
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a_name: &str,
    w_name: &str,
    o_name: &str,
    a_off: u32,
    w_off: u32,
    o_off: u32,
    operand_df: Df,
    // PHYSICAL `mb` extent when this op iterates only PART of the activation's rows. `None` (every
    // caller until the row-batched fold) means "the tensor has exactly the rows this op sweeps".
    //
    // The y-batched walk under the request-batch form is HEAD-OUTERMOST, so advancing `y` by one
    // moves one head PLANE and a plane is `mb·in` elements — the head stride comes from the mb
    // DEVICE extent. A pass that computes one request's row per head therefore needs `mb = 1` for
    // the work and `phys_mb` naming the tensor's own plane depth, or every head but the first is
    // addressed the wrong distance in.
    phys_mb: Option<u32>,
    // ⭐ THE KERNEL'S TRUE PHYSICAL `in` EXTENT when this op contracts only PART of the weight's
    // reduction axis — ONE K TRIP of a K-trip decomposition (see `super::ktrips`). `None` (every
    // caller before that decomposition) means "the weight is exactly as deep as this op contracts".
    //
    // The shared 2-D kernel walk is `[in, out]` stick `out`, whose address law is
    // `(n/stk)·(in·stk) + k·stk + (n%stk)` (`sdsc_abstract::dev_off_stk`, rank-2 stick-on-last), so
    // the `out` stick-GROUP stride is `in·stk` — the FULL weight depth. A trip that sweeps `k/T` and
    // leaves `in` at the swept extent therefore strides every `out` core `T`× too close and reads
    // another trip's slab: correct for out-core 0 and wrong for all 31 others, which is a fluent
    // wrong answer rather than a fault. `with_device_extent` is exactly the "iterate a window,
    // address the allocation" declaration (`DeviceExtents::of_view`: a declared extent WINS), and
    // `w_off = k0·stk` then places the trip inside that allocation.
    kernel_phys_in: Option<u32>,
    // WHICH walk pair the batch>1 arm declares — see [`SharedKernelBmmForm`]. The live callers pick
    // the splitter from this same value (`matmul_split_map_for`); the Kani seam injects its own
    // splitter and states the proven form.
    form: SharedKernelBmmForm,
    // ⛔ THE TWO BATCHED OPERANDS' REAL HEAD STRIDES, so this builder can refuse a walk that would
    // step into another head. `None` ONLY at `batch <= 1`, where there is no `y` and the question does
    // not exist — and the only producer of a `Some` is `MatY::of_gqa_group`, which cannot be called
    // without both placements. This raw seam takes them as a value because it takes every other extent
    // as a value too (the Kani tower drives it); the TYPED door above is where they are demanded.
    batch_strides: Option<crate::sdsc_abstract::BatchStrides>,
    splitter: S,
) -> Result<OpSpec, String>
where
    S: Fn(&[crate::superdsc_opspec::ItDim], u32) -> std::collections::BTreeMap<&'static str, u32>,
{
    // N and K are constructed as typed [`StickExtent<DF>`] — for `DF = Fp8` this is a
    // 128-multiple witness, so a non-128 fp8 extent is a Rust `Err` at emit, NEVER the
    // on-card `L3DlOpsScheduler:1070 multiple-of-stick` DtException. The `DF` also flows
    // into `matmul_dims`, stamping the stick dims with `DF::DF` so the work-division's
    // stick basis is the operand format's (fp8 ⇒ 128) by TYPE, not a hand-picked 64.
    let n_ext = StickExtent::<DF>::new(n)?;
    let k_ext = StickExtent::<DF>::new(k)?;
    let mut dims = matmul_dims::<DF>(m, &n_ext, &k_ext, batch);
    // ⭐⭐⭐ THE REDUCTION'S STICK BASIS IS THE **OPERAND'S**, PER AXIS — and this line is the whole
    // difference between "N and K share one format" and what the machine actually does.
    //
    // The fp8 path calls this with `DF = Fp16` ON PURPOSE (see `lower_matmul_node`'s fq_mm site): the
    // OUTPUT is fp16 and the packed fp8 KERNEL's N-stick is 64, so the `out` work-division must split on
    // the 64 basis. But the fp8 ACTIVATION's K-stick is 128, and `stick_basis` — which is what
    // `WorkPlan::divide`'s stick clause and the split search both measure against — reads the DIM's `df`.
    // Left at `DF::DF` the reduction is measured in fp16 sticks, so a per-core `in` slice of 960 elements
    // passes as "15 whole sticks" when it is SEVEN AND A HALF fp8 sticks.
    //
    // ⛔ AND THAT IS NOT HYPOTHETICAL — it is gemma-4's dxp wall, measured one op per compile on branch
    // `worktree-spyre-gemma4` (`a1cde93f`): `k=3840 n=15360` split `in=4 out=8` is refused with
    // `DtException: There must be at least one valid candidate.` (`L3DlOpsScheduler.cpp:1375`) while
    // `in=1 out=30` and `in=5 out=6` compile. granite never hit it because its cost model picks `in=1` for
    // every projection (`SCRATCHY_SDSC_OPHIST`'s stream lines show `"in": 1` on all seven), so the wrong
    // basis was inert on every model that had run — the same shape as every other coincidence in this file.
    //
    // The fq_mm site's own comment claims the reduction is "never split", and for granite's extents that
    // is TRUE BY ARITHMETIC rather than by construction. This makes the basis right whether it is split or
    // not, so the claim no longer has to hold for the emission to be legal.
    if let Some(d) = dims
        .iter_mut()
        .find(|d| d.name == super::walk::InAxis::NAME)
    {
        d.df = operand_df;
    }
    // LOWER to TileIR: this matmul's iteration domain + kind, ONE `TileOp` declaration instead of the
    // inline `WorkPlan::divide` + hand-picked residency call — reached from the REAL `lower_one_node`
    // SubtileTape dispatch (every `SubOp::MatmulTile`), same pattern as `pointwise_opspec`/
    // `reduce_opspec_df`. `DF::DF` bridges the compile-time format marker to the runtime `Df`
    // `TileOp::resident_bytes` dispatches on (fp8 ⇒ its 1-byte operand path, unchanged).
    let tile_op = TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::Matmul,
        dims: dims.clone(),
        df: operand_df,
    };
    // ── LX-fit TIME-TILING (the coarse_tile pass) ───────────────────────────
    // After the spatial work-division, decide whether the per-core resident set
    // (A + W + O, fp16=2 B, INCLUDING the batch/x factor — KERNEL `w` dominates
    // and is NOT divided by the mb split) fits the USABLE_LX scratchpad. If not,
    // tile the OUTPUT stick dim `out` in TIME. "doesn't fit even when tiled" is a
    // build `Err` (the DtException-1535 guard), NOT an on-card crash.
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, splitter, OutAxis::NAME)
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);

    // The CANONICAL matmul dim set is EXACTLY {mb(M), in(K), out(N), x(batch)} — NO
    // phantom `y` (verified against torch-spyre's own test_coarse_tiling.py:1330-1343:
    // INPUT dim_order ["mb","in","x"], OUTPUT ["mb","out","x"], iteration [x,mb,out,in];
    // matmul takes use_op_dims=False ⇒ args carry ONLY their own dims, ALL scale=1, and
    // the reduction is expressed PURELY by OMITTING `in` from OUTPUT). A spurious `y`
    // (scratchy's old 1535 workaround) breaks dxp's contraction inference → the matmul
    // computed an orthogonal product. in(=K) is the INPUT stick (Active keeps its ×64
    // stick factor — no 1535); out(=N) is the KERNEL/OUTPUT stick.
    // dxp-VALIDATED fixture exact match (l0_tethering MatMul_49):
    // INPUT  [mb,in,y]  stick=in   scale[1,1,1].
    // KERNEL [in,out]   stick=out  scale[1,1]   — BARE 2-D, shared weight (broadcast
    //                                             across mb+y); a 3-D kernel makes dxp
    //                                             treat the weight as per-batch → garbage.
    // OUTPUT [mb,out,y] stick=out  scale[1,1,1] (reduction = OMIT `in`).
    // KERNEL is always the bare 2-D shared weight [in,out] (broadcast across mb+batch).
    let kernel_walk = Walk2::kernel_shared();
    let kernel = TensorArg::<2>::new(
        true,
        w_name.to_string(),
        Role::Kernel,
        [Scale::Active, Scale::Active],
        plan.iter_syms(kernel_walk.order()),
        kernel_walk.order(),
        kernel_walk.stick(),
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(w_off);
    // The K-trip window (see `kernel_phys_in`). A phys SMALLER than the swept extent is not a
    // window but a mis-declaration — the walk would describe less memory than the op reads — so it
    // is refused here, the SAME rule `matmul_opspec_batched_off` enforces for its `y` override.
    let kernel = match kernel_phys_in {
        Some(phys) => {
            let swept = plan.extent(InAxis::NAME);
            if phys < swept {
                return Err(format!(
                    "matmul_opspec '{w_name}': kernel device_extent in={phys} is SMALLER than the \
                     swept `in` {swept}. device_extent declares a window into a LARGER allocation \
                     (phys >= iteration); a smaller value cannot be expressed this way."
                ));
            }
            kernel
                .with_device_extent(InAxis::NAME, phys)
                .map_err(|e| e.0)?
        }
        None => kernel,
    };

    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    // INPUT/OUTPUT rank tracks the op: a PLAIN 2-D matmul (batch==1) is [mb,in]/[mb,out] (opFuncName
    // "matmul", matching torch-spyre's generate_sdsc — the `y` phantom + batchmatmul was THE mq>1
    // prefill collapse). A true batched matmul (batch>1, attention bmm) keeps the `y`(=batch) dim.
    let args = if batch <= 1 {
        let input_walk = Walk2::input_rows_by_k();
        let input = TensorArg::<2>::new(
            true,
            a_name.to_string(),
            Role::Input,
            [Scale::Active, Scale::Active],
            plan.iter_syms(input_walk.order()),
            input_walk.order(),
            input_walk.stick(),
            Allocation::Hbm,
        )
        .map_err(|e| e.0)?
        .with_offset(a_off);
        let output_walk = Walk2::output_rows_by_n();
        let output = TensorArg::<2>::new(
            false,
            o_name.to_string(),
            Role::Output,
            [Scale::Active, Scale::Active],
            plan.iter_syms(output_walk.order()),
            output_walk.order(),
            output_walk.stick(),
            Allocation::Hbm,
        )
        .map_err(|e| e.0)?
        .with_offset(o_off);
        // ⭐⭐⭐⭐⭐ THE OUTPUT'S OWN ROW COUNT, WHEN IT IS A SLICE OF A TALLER BUFFER — and this is what
        // lets ONE op write a head's WHOLE `[mq, hd]` block instead of one op per slab.
        //
        // A rank-2 stick-blocked view addresses `(j/stk)*(dims[0]*stk) + i*stk + (j%stk)`, so its
        // stick-GROUP stride comes from `dims[0]` — the mb DEVICE extent. An op sweeping one head's
        // `mq` rows out of a `[nqh*mq, hd]` accumulator therefore derives `mq*stk` where the tensor's
        // group stride is `rows*stk`: correct for the FIRST stick group and `nqh`× too close for every
        // one after it. That — not the write width — is what forced `for s in 0..nslab`.
        //
        // ⛔ AND IT IS INERT AT ONE SLAB, which is why the loop looked like the only option: with a
        // single stick group `j/stk` is always 0 and the wrong factor multiplies nothing. Every hd=64
        // bundle emits byte-identically.
        //
        // `phys_mb` already carried exactly this statement for the batched input ("this op ITERATES a
        // window of a LARGER allocation"); it simply was never offered to the output.
        let output = match phys_mb {
            Some(phys) => {
                let swept = plan.extent(MbAxis::NAME);
                if phys < swept {
                    return Err(format!(
                        "matmul_opspec '{o_name}': phys_mb={phys} is SMALLER than the swept mb \
                         {swept}. device_extent declares a window into a LARGER allocation."
                    ));
                }
                output
                    .with_device_extent(MbAxis::NAME, phys)
                    .map_err(|e| e.0)?
            }
            None => output,
        };
        vec![
            AnyTensorArg::R2(input),
            AnyTensorArg::R2(kernel),
            AnyTensorArg::R2(output),
        ]
    } else {
        // WHICH rank-3 walk pair — the REGIME's associated order, carried as the regime's own
        // typed walks all the way into [`batched_walk_args`], so input and output orders arrive
        // from one type (see `RungRegime`):
        //
        //   * BatchInnerMq1Proven — `[mb, y, in]` / `[mb, y, out]`. The order every shipped bundle
        //     carries; on-hardware-proven for the mq==1 decode g-form, where the unit `mb` makes its
        //     stride assignment coincide with the head-major layout. At `mb > 1` a PINNED
        //     (`maxDimSizes_` = actual) walk of this order declares the strides SWAPPED — MEASURED
        //     at mq=23: `numWkSlicesPerDim_ {mb:23, y:1}`, `maxDimSizes_ [23,4,64]` — mb-stride 256
        //     / y-stride 64 against the tensor's real 64 / mq·hd.
        //   * HeadOutermostRequests — `[y, mb, in]` / `[y, mb, out]` (batch OUTERMOST), the
        //     rows-are-requests decode rungs. Row-major over this order gives `y` one head PLANE
        //     (`mb·in` — a head-major operand's head stride at ANY `mb`, with `mb` the DEVICE extent
        //     so a `phys_mb` override sets the true plane) and `mb` one row (`in`, the row pitch) —
        //     the layout every head-major attention tensor really has: `qs` head h at `h·mb·in`,
        //     `sc` head h at `h·mb·out`.
        //
        // The stick dim stays LAST in both (contiguous), so `stickDimOrder_` is unchanged. The one
        // order with NO constructor at all is `[mb, in, y]` — `y` innermost strides ONE ELEMENT
        // inside each stick. MEASURED at batch=4: qs read heads 1 element apart while the proven
        // per-head ops read them 64 apart (h4 @6797696, h5 @6797824) — a 64x mis-stride, and exactly
        // the garbage it produced.
        // ⛔⛔⛔ THE STRIDE THE WALK WILL USE MUST BE THE STRIDE THE OPERANDS HAVE — REFUSED AT BUILD.
        //
        // A `y`-batched op writes ONE base offset and strides to the remaining heads. What it strides
        // by is decided by the regime's declared order (`RungRegime::y_stride_elems`); where the heads
        // ACTUALLY are is each operand's own placement law. Nothing related the two, so the arm sat
        // behind a head-dim gate standing in for the relation, and it was WRONG the moment a head
        // stopped being one stick: with `out = hd` a head-major output derives `mb*hd` against a real
        // pitch of `mb*stick`, equal only at `hd == stick`. That is not a fault on-card — it reads and
        // writes inside another head's features and produces fluent, wrong output.
        //
        // `MatY::of_gqa_group` cannot be built without both placements, so the required strides are
        // always present here, and this compares them against what the walk derives. Returning `Err`
        // inside the `#[forward]` proc macro IS a `cargo build` failure naming the op — the head-dim
        // gate is replaced by the relation it was approximating.
        let pitches = batch_strides.map(|w| (w.activation_pitch().get(), w.output_pitch().get()));
        if let Some(want) = batch_strides {
            // ⛔⛔⛔ EACH OPERAND'S OWN PITCH. This was ONE `phys_mb.unwrap_or(m)` shared by both
            // checks, and that single value is why a one-stick contraction looked impossible: the score
            // leg's token-stream `qs` needs `mq*nslab` when `in` is one stick, while its head-major
            // `sc` needs `mq` — satisfying either broke the other. The pitches ride on the same
            // placements as the strides, so neither can be paired with the wrong operand.
            let a_mb_dev = want.activation_pitch().get().max(phys_mb.unwrap_or(m));
            let o_mb_dev = want.output_pitch().get().max(1);
            let check = |regime_stride: u32,
                         have: crate::addr::AxisStride,
                         role: &str,
                         axis: &str,
                         ext: u32|
             -> Result<(), String> {
                if regime_stride == have.elems() {
                    return Ok(());
                }
                Err(format!(
                    "matmul '{o_name}': the batched {role} walk strides `y` by {regime_stride} elems \
                     (derived={regime_stride}, {axis}={ext}), but this operand's placement law puts adjacent \
                     heads {} elems apart. A `y`-batch would step into another head. If the operand is \
                     head-major its pitch is `mb*stick`, so declare ONE STICK of `{axis}`; if it is a \
                     token stream its pitch is `mb*hd` and the full width is right. (This is the \
                     relation the head_dim==64 gate used to stand in for.)",
                    have.elems(),
                ))
            };
            match form.regime() {
                SharedKernelBmmRegime::Proven(r) => {
                    check(
                        r.y_stride_elems(a_mb_dev, k),
                        want.activation(),
                        "input",
                        "in",
                        k,
                    )?;
                    check(
                        r.y_stride_elems(o_mb_dev, n),
                        want.output(),
                        "output",
                        "out",
                        n,
                    )?;
                }
                SharedKernelBmmRegime::Requests(r) => {
                    check(
                        r.y_stride_elems(a_mb_dev, k),
                        want.activation(),
                        "input",
                        "in",
                        k,
                    )?;
                    check(
                        r.y_stride_elems(o_mb_dev, n),
                        want.output(),
                        "output",
                        "out",
                        n,
                    )?;
                }
            }
        }
        let (input, output) = match form.regime() {
            SharedKernelBmmRegime::Proven(regime) => batched_walk_args(
                regime, &plan, a_name, o_name, a_off, o_off, phys_mb, pitches,
            )?,
            SharedKernelBmmRegime::Requests(regime) => batched_walk_args(
                regime, &plan, a_name, o_name, a_off, o_off, phys_mb, pitches,
            )?,
        };
        vec![
            AnyTensorArg::R3(input),
            AnyTensorArg::R2(kernel),
            AnyTensorArg::R3(output),
        ]
    };
    // The operand RESIDENCY is the same `DF` type decision as the stick basis: stamp every
    // non-output arg (activation + weight) with `DF::DF` so `::<Fp8>` yields 1-byte packed
    // fp8 HBM reads (`wordLength`=1) AND the 128 stick basis ATOMICALLY. You cannot pick the
    // fp8 basis without the fp8 residency — this closes the "forgot `set_df` ⇒ silent 2-byte
    // fp16 weight read" reward-hack (see fp8-reward-hack-lessons). `DF = Fp16` is the default
    // ⇒ this is a no-op for every dense/attention matmul (byte-identical).
    let mut args = args;
    for arg in &mut args {
        if !matches!(arg.view().role, Role::Output) {
            arg.set_df(DF::DF);
        }
    }
    Ok(OpSpec {
        op: OpFunc::matmul(batch),
        is_reduction: true,
        iter: plan,
        args,
        op_info: OpInfo::None,
        tiled_symbols,
        time_tile,
    })
}

/// BOTH rank-3 args of one regime's shared-kernel batched matmul, from the regime's own typed walk
/// pair. The pair rides as `Walk3` values down to the descriptor constructors: the input slot
/// ([`batched_input_arg`]) admits only a stick-`in` walk and the output slot
/// ([`batched_output_arg`]) only a stick-`out` walk, so handing either half to the other's
/// descriptor is an `E0308` — there is no point between the regime and the descriptor write where
/// the two halves exist as interchangeable values.
#[allow(clippy::too_many_arguments)]
fn batched_walk_args<R: RungRegime>(
    regime: R,
    plan: &WorkPlan,
    a_name: &str,
    o_name: &str,
    a_off: u32,
    o_off: u32,
    phys_mb: Option<u32>,
    // ⭐ EACH OPERAND'S OWN DECLARED ROW PITCH, from its own placement law (see `BatchStrides`). The
    // two genuinely differ on the score leg: a token-stream activation stepped one stick at a time
    // needs `mq*nslab` while the head-major output needs `mq`. One shared `phys_mb` could satisfy
    // only one of them, which is what made a one-stick contraction look unrepresentable.
    pitches: Option<(u32, u32)>,
) -> Result<(TensorArg<3>, TensorArg<3>), String> {
    let input = batched_input_arg(a_name, regime.input_walk(), plan)?.with_offset(a_off);
    // A phys SMALLER than the swept extent is a mis-declaration, not a window (same rule the
    // kernel override enforces): the walk would describe less memory than the op reads.
    let input = match phys_mb {
        Some(phys) => {
            let swept = plan.extent(MbAxis::NAME);
            if phys < swept {
                return Err(format!(
                    "matmul_opspec '{a_name}': phys_mb={phys} is SMALLER than the swept mb                          {swept}. device_extent declares a window into a LARGER allocation."
                ));
            }
            input
                .with_device_extent(MbAxis::NAME, phys)
                .map_err(|e| e.0)?
        }
        None => input,
    };
    // The ACTIVATION's own pitch, when its placement declares one larger than the swept rows.
    let input = match pitches.map(|(a, _)| a) {
        Some(a_pitch) if a_pitch > plan.extent(MbAxis::NAME) => input
            .with_device_extent(MbAxis::NAME, a_pitch)
            .map_err(|e| e.0)?,
        _ => input,
    };
    let output = batched_output_arg(o_name, regime.output_walk(), plan)?.with_offset(o_off);
    // ⛔ AND THE OUTPUT DECLARES ITS OWN — this is the half that was missing. Checking the output
    // against a pitch it never declared makes the two agree on paper while the emitted walk still
    // strides by the shared value.
    let output = match pitches.map(|(_, o)| o) {
        Some(o_pitch) if o_pitch > plan.extent(MbAxis::NAME) => output
            .with_device_extent(MbAxis::NAME, o_pitch)
            .map_err(|e| e.0)?,
        _ => output,
    };
    Ok((input, output))
}

/// The batched INPUT descriptor write. Its walk slot is `Walk3<_, _, InAxis>` — only a
/// reduction-sticked walk can describe the operand this constructor stamps `Role::Input`.
fn batched_input_arg<A: WalkAxis, B: WalkAxis>(
    name: &str,
    walk: Walk3<A, B, InAxis>,
    plan: &WorkPlan,
) -> Result<TensorArg<3>, String> {
    TensorArg::<3>::new(
        true,
        name.to_string(),
        Role::Input,
        [Scale::Active, Scale::Active, Scale::Active],
        plan.iter_syms(walk.order()),
        walk.order(),
        walk.stick(),
        Allocation::Hbm,
    )
    .map_err(|e| e.0)
}

/// The batched OUTPUT descriptor write. Its walk slot is `Walk3<_, _, OutAxis>` — only an
/// output-sticked walk can describe the operand this constructor stamps `Role::Output`.
fn batched_output_arg<A: WalkAxis, B: WalkAxis>(
    name: &str,
    walk: Walk3<A, B, OutAxis>,
    plan: &WorkPlan,
) -> Result<TensorArg<3>, String> {
    TensorArg::<3>::new(
        false,
        name.to_string(),
        Role::Output,
        [Scale::Active, Scale::Active, Scale::Active],
        plan.iter_syms(walk.order()),
        walk.order(),
        walk.stick(),
        Allocation::Hbm,
    )
    .map_err(|e| e.0)
}

/// Build the typed [`OpSpec`] for a TRUE PER-BATCH (3-D-kernel) batchmatmul —
/// `A[batch,m,k] · W[batch,k,n] → O[batch,m,n]`, the multi-head attention bmm.
///
/// THE FIX for the confirmed shared-K attention bug: [`matmul_opspec`]'s KERNEL is a BARE 2-D
/// `["in","out"]` (broadcast across the batch — the dxp `SHARED_WEIGHT_UNIT_BMM`, correct ONLY
/// for batch=1 / a shared weight). For attention `batch=nqh>1` that makes EVERY query head
/// attend with head-0's K/V. torch-spyre's `lower_bmm` (`_inductor/lowering.py:417`) handles
/// this as the `x_ndim==3 && y_ndim==3` case: `tmp2 = y[i0,r0,i2]` — the KERNEL is
/// indexed PER-BATCH (`y[batch,K,N]`), and `_static_bmm_custom_meta` confirms a
/// batch>1 bmm gets NO shared-weight marker. So ALL THREE operands carry the batch
/// dim, BATCH-OUTERMOST (mirroring the reference's `[i0,…]`-first index):
///   INPUT  `[y,mb,in]`  stick `in`  — `y`=batch(nqh), `mb`=M(=mq), `in`=K
///   KERNEL `[y,in,out]` stick `out` — the 3-D per-batch weight (`y[batch,K,N]`)
///   OUTPUT `[y,mb,out]` stick `out`
/// With `y` in the KERNEL's layout, `per_core_addr`/`build_coordinates` emit a
/// per-`y` coordinate ⇒ dxp advances the kernel base by one head's `in·out` block
/// per batch (the per-head K/V). The caller MUST lay the kernel tensor out
/// head-contiguous batch-outermost (`[nqh,in,out]`) — the K-cache restickify now
/// emits `[mb,out,y]` for exactly this.
pub fn matmul_opspec_batched(
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a_name: &str,
    w_name: &str,
    o_name: &str,
) -> Result<OpSpec, String> {
    // The untyped wrapper boundary. Its own contract (the fn doc above) fixes the meanings: `m` is
    // the query rows and `batch` carries the HEADS of the per-batch bmm.
    matmul_opspec_batched_off(
        MatM::of_query_rows(crate::sdsc_abstract::QueryRowCount::of_mq(m)),
        MatN::of_out_features(n),
        MatK::of_in_features(k),
        MatY::of_heads(batch),
        a_name,
        w_name,
        o_name,
        0,
        0,
        0,
        None,
    )
}

/// [`matmul_opspec_batched`] with per-operand ELEMENT start-offsets and an optional KERNEL
/// physical-extent override — the form the batched DECODE attention needs.
///
/// Two things separate a batched attention bmm from the plain batched form:
///
/// 1. **Offsets.** Each block of the online-softmax loop reads a 64-slot window of the resident
///    K/V cache, so the kernel starts at `b · <block stride>`, not 0.
/// 2. **Swept extent ≠ storage stride.** `matmul_opspec_batched` derives the 3-D kernel's
///    per-batch (`y`) stride from the ITERATION extents (`in · out`). That is only the true
///    stride when the op sweeps the tensor's whole physical extent. The decode ladder sweeps
///    `active_cap ≤ cap` of a `[nkvh, hd, cap]` Kᵀ cache, and the V cache is `nqh`-sized while
///    only every `gqa`-th head slot is live — in both cases the real `y` stride is LARGER than
///    `in · out`. `kernel_device_extent = Some((dim, phys))` declares that dim's TRUE physical
///    extent via [`TensorArg::with_device_extent`] (torch-spyre's `arg.device_size`), so stride
///    derivation uses `phys` and the sub-rung ladder addresses correctly. Without it a batched
///    score is right only at the top rung — the "no ladder-safe drop-in exists" obstacle.
///
/// `None` reproduces [`matmul_opspec_batched`] exactly (all offsets 0, strides from iteration
/// extents), so this is a strict superset and existing behaviour is byte-identical.
#[allow(clippy::too_many_arguments)]
pub fn matmul_opspec_batched_off(
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    a_name: &str,
    w_name: &str,
    o_name: &str,
    a_off: u32,
    w_off: u32,
    o_off: u32,
    kernel_device_extent: Option<(&'static str, u32)>,
) -> Result<OpSpec, String> {
    // The floor of the typed run for the batched form: raw extents from here down.
    let (m, n, k, batch) = (m.get(), n.get(), k.get(), batch.get());
    let n_ext = StickExtent::<Fp16>::new(n)?;
    let k_ext = StickExtent::<Fp16>::new(k)?;
    // Attention bmm operands are fp16 (SEN169) — 64-lane sticks.
    let dims = matmul_dims::<Fp16>(m, &n_ext, &k_ext, batch);
    // LOWER to TileIR: same pattern as `matmul_opspec_split` — the batched-attention bmm's TileOp.
    let tile_op = TileOp {
        kind: crate::ir::island::tile_op::TileOpKind::Matmul,
        dims: dims.clone(),
        df: Df::Fp16,
    };
    let tiled = tile_op
        .tile(MaxCores::<MAX_CORES>, matmul_split_map, OutAxis::NAME)
        .map_err(|e| e.0)?;
    let (plan, time_tile) = (tiled.plan, tiled.time_tile);
    // INPUT [y,mb,in] stick=in (batch-outermost; reduction = OMIT `in` from OUTPUT).
    let input_walk = Walk3::input_head_outermost();
    let input = TensorArg::<3>::new(
        true,
        a_name.to_string(),
        Role::Input,
        [Scale::Active, Scale::Active, Scale::Active],
        plan.iter_syms(input_walk.order()),
        input_walk.order(),
        input_walk.stick(),
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(a_off);
    // KERNEL [y,in,out] stick=out — the 3-D PER-BATCH weight (`y` carries the head),
    // NOT the 2-D shared kernel. This is the entire difference from `matmul_opspec`.
    let kernel_walk = Walk3::kernel_per_head();
    let kernel = TensorArg::<3>::new(
        true,
        w_name.to_string(),
        Role::Kernel,
        [Scale::Active, Scale::Active, Scale::Active],
        plan.iter_syms(kernel_walk.order()),
        kernel_walk.order(),
        kernel_walk.stick(),
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(w_off);
    // The `y`-stride override (see the fn doc): declare the kernel dim whose PHYSICAL extent
    // exceeds this op's swept extent, so stride derivation uses storage, not iteration.
    let kernel = match kernel_device_extent {
        Some((dim, phys)) => {
            // `with_device_extent` means "this op sweeps a WINDOW of a larger allocation". A phys
            // SMALLER than the swept extent is not a window, it is a mis-declaration: the walk would
            // describe less memory than the op reads. Refuse at `cargo build` rather than emit a
            // descriptor whose declared size and address span disagree (an on-card Compute CB error).
            let swept = plan.extent(dim);
            if phys < swept {
                return Err(format!(
                    "matmul_opspec_batched_off '{w_name}': kernel device_extent {dim}={phys} is \
                     SMALLER than the swept extent {swept}. device_extent declares a window into a \
                     LARGER physical allocation (phys >= iteration); a smaller value cannot be \
                     expressed this way."
                ));
            }
            kernel.with_device_extent(dim, phys).map_err(|e| e.0)?
        }
        None => kernel,
    };
    let output_walk = Walk3::output_head_outermost();
    let output = TensorArg::<3>::new(
        false,
        o_name.to_string(),
        Role::Output,
        [Scale::Active, Scale::Active, Scale::Active],
        plan.iter_syms(output_walk.order()),
        output_walk.order(),
        output_walk.stick(),
        Allocation::Hbm,
    )
    .map_err(|e| e.0)?
    .with_offset(o_off);
    let tiled_symbols = time_tile.map(|t| vec![t.dim()]).unwrap_or_default();
    let args = vec![
        AnyTensorArg::R3(input),
        AnyTensorArg::R3(kernel),
        AnyTensorArg::R3(output),
    ];
    Ok(OpSpec {
        op: OpFunc::matmul(batch),
        is_reduction: true,
        iter: plan,
        args,
        op_info: OpInfo::None,
        tiled_symbols,
        time_tile,
    })
}
