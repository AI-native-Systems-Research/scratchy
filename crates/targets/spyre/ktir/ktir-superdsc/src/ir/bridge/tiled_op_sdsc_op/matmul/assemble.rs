//! Matmul family: the `assemble_matmul*` wrappers (opspec builder + `emit_sdsc_tiled`). See
//! `super`'s module doc.

use super::opspec::{matmul_opspec, matmul_opspec_batched, matmul_opspec_off, matmul_opspec_split};
use super::walk::SharedKernelBmmForm;
use crate::emit::EmittedOp;
use crate::placement::BundleLayout;
use crate::sdsc_abstract::{MatK, MatM, MatN, MatY, OperandPlacement, Stk};
use crate::superdsc_opspec::{DataFormat, Fp16, SdscFoldSet};

/// Assemble a complete SuperDSC for ONE matmul `A[m,k]·W[k,n]→O[m,n]` (batch
/// `b`), via the typed [`matmul_opspec`] builder + `emit_sdsc`. PANICS on a
/// sub-stick N/K (the witness `Err`) — the SubtileIR walk's guards catch this at
/// build time before `assemble_matmul` is reached, so a panic here is a true
/// internal-consistency bug, not reachable from model input.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul(
    op_name: &str,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    o: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let mut sym_id_base: i64 = 0;
    assemble_matmul_seeded(op_name, m, n, k, batch, a, w, o, &mut sym_id_base, layout)
}

/// [`assemble_matmul`] with an explicit running symbol-id counter so multiple
/// tiled ops in one bundle get globally-unique negative ids (design risk #1).
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_seeded(
    op_name: &str,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    o: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    try_assemble_matmul_seeded(op_name, m, n, k, batch, a, w, o, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"))
}

/// [`assemble_matmul_seeded`] AS A `Result` — the form a producer-facing entry point needs, exactly as
/// [`crate::emit::try_assemble_transpose`] is that form for the transpose door.
///
/// ⛔ THE PANIC IS RIGHT FOR THE ASSEMBLERS AND WRONG FOR THE DOOR, and this matmul is now the second
/// op to show why. Inside a `#[forward]` expansion a panic IS the build error, which is what
/// `assemble_matmul`'s own doc means by "a panic here is a true internal-consistency bug, not
/// reachable from model input" — true of the SUB-STICK witness it names. It is NOT true of the
/// PLACEMENT-FOOTPRINT refusal `resolve_seg_base` raises, which is reachable from the caller's own
/// buffer SIZING and was reached twice this session by a KTIR producer whose parameters are a Triton
/// kernel's descriptors. Reported as a panic it exits 101 with no stage label, so a driver that
/// tabulates its own `REFUSED <stage> <message>` lines shows a footprint mismatch as a BLANK ROW
/// rather than as a failure — a reporting hole precisely where the report has to be trusted.
///
/// The panicking form keeps its signature and its panic for every existing caller, and now delegates
/// here so there is ONE emission path rather than two that could drift.
#[allow(clippy::too_many_arguments)]
pub fn try_assemble_matmul_seeded(
    op_name: &str,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    o: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp, String> {
    // ⛔ THE OCCUPANCY PAD IS ONLY REAL IF SOMETHING RESERVED IT — see
    // [`out_width_the_weight_holds`]. A no-op for every caller whose layout reserves the padded
    // weight; it drops the pad for a producer whose weight is the CALLER'S OWN tensor.
    let n = out_width_the_weight_holds(layout, w.name(), m, n, k);
    // TYPED OPERANDS (compile-time addressing safety): a matmul's activation is RowBlocked, its weight
    // is a Kernel, its output is RowBlocked — a `cargo build` type error otherwise (you cannot hand a
    // `Flat`/`RowScalar` tensor here). The handles carry only the emitter NAME into `matmul_opspec`
    // (which owns df/shape), so the emit is byte-identical — the types are a pure addressing guard.
    let op = matmul_opspec(m, n, k, batch, a.name(), w.name(), o.name())?;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout).map_err(|e| e.0)
}

/// ⛔⛔⛔ THE `out` WIDTH THE LAYOUT ACTUALLY RESERVED FOR THIS MATMUL'S WEIGHT — because an
/// OCCUPANCY PAD IS ONLY REAL IF SOMETHING MADE IT REAL.
///
/// `DeviceWidth::for_output` pads a FLOP-heavy matmul's `out` stick count so the gemm fills cores
/// (`bump_sticks_to_splittable`), and the emitted kernel then MAC's over `[in, out_padded]`. Those
/// invented columns are legal only because a SECOND party makes them exist: the worker zero-pads the
/// staged weight by this same rule (`spyre_load`'s `n_dev`) and the bundle layout reserves the padded
/// footprint for it, so the placement is as wide as the access.
///
/// A KTIR producer whose parameters are the CALLER'S OWN TENSORS has no staging pass. A Triton
/// kernel's `desc_wg` is a `[D_FF, D_MODEL]` buffer the host hands over as-is, and the layout sizes
/// its placement from exactly that view. The pad is then width nobody made, and the kernel reads past
/// the weight — MEASURED as `resolve_seg_base`'s own refusal, at both blockings of the SwiGLU MLP:
///
/// ```text
///   m=64  n=256   k=128   → n_dev 512    t1: 0B + 131072B    exceeds footprint 65536B    (seg1)
///   m=64  n=12800 k=4096  → n_dev 14336  t1: 0B + 117440512B exceeds footprint 104857600B (seg1)
/// ```
///
/// (Not a constant factor — 2× and 1.12× — because the two shapes take DIFFERENT arms of the bump:
/// 4 sticks is rounded to the ≥8-core floor, 200 sticks takes the full-occupancy arm to 224.)
///
/// ⛔ AND IT ONLY EVER DROPS A PAD IT CAN PROVE IS THIS RULE'S OWN. `held` is the whole-stick width
/// the reserved footprint holds; it is used ONLY when re-applying the padding rule to `held`
/// reproduces the `n` we were handed. Any other shortfall is a REAL over-run — a wrong shape, a
/// mis-sized placement — and `n` is returned UNCHANGED so the footprint guard still refuses it BY
/// NAME instead of being quietly satisfied by a narrower emission. The util floor is re-checked at
/// `held` for the same reason: the caller's guard #11 measured the PADDED width, so a `held` that
/// would strand the gemm is left to that refusal rather than emitted below the floor.
///
/// A no-op wherever the layout reserves the padded weight (`held == n`), so the full-model path —
/// including the granite lm_head, the one weight whose stick count needs the pad — is byte-identical.
fn out_width_the_weight_holds(
    layout: Option<&BundleLayout>,
    w_name: &str,
    m: u32,
    n: u32,
    k: u32,
) -> u32 {
    use crate::work::{CoreSplit, DeviceWidth, FP16_ELEMS_PER_STICK};
    let Some(l) = layout else { return n };
    let Some(crate::place::PlaceId::Act(tid)) = l.id_of(w_name) else {
        return n;
    };
    let Some(p) = l.placements.get(&tid) else {
        return n;
    };
    // The weight buffer is the `out`-major `[n, k]` one this door's `KtirFunc::matmul` views (the
    // transposed-weight `indexing_maps`), and a placement is sized by `synth_footprint_bytes`: the
    // INNER axis stick-rounded, the outer multiplied. So one `out` row costs a stick-rounded `k`.
    let row_bytes = k.next_multiple_of(FP16_ELEMS_PER_STICK) as u64 * 2; // fp16 kernel
    if row_bytes == 0 {
        return n;
    }
    let held = ((p.size / row_bytes) as u32 / FP16_ELEMS_PER_STICK) * FP16_ELEMS_PER_STICK;
    let floor_ok = CoreSplit::plan(m, held).ncores() >= 8;
    if held < n && held > 0 && floor_ok && DeviceWidth::for_output(m, held, k).get() == n {
        held
    } else {
        n
    }
}

/// [`assemble_matmul_seeded`] with an INJECTABLE work-division `splitter` — the Kani-verified tower's
/// entry point. It supplies a `CoreSplit`-derived splitter (the PROVEN 32-core partition) and reuses this
/// ABI emit (`matmul_opspec_split` → `emit_sdsc_tiled`). Returns `Result` (no panic) so the tower keeps its
/// `Result`-based error flow; a bad shape is an `Err`, surfaced by the caller.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_split(
    op_name: &str,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a_name: &str,
    w_name: &str,
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    splitter: impl Fn(
        &[crate::superdsc_opspec::ItDim],
        u32,
    ) -> std::collections::BTreeMap<&'static str, u32>,
) -> Result<EmittedOp, crate::superdsc_error::SuperDscError> {
    let op = matmul_opspec_split::<Fp16, _>(
        m,
        n,
        k,
        batch,
        a_name,
        w_name,
        o_name,
        0,
        0,
        0,
        <Fp16 as DataFormat>::DF,
        None,
        // The tower's shapes are not attention decode rows; they keep the proven walk pair.
        SharedKernelBmmForm::batch_inner_proven(super::walk::MatmulWrapperSite::witness()),
        // No head axis to state: the Kani tower's seam drives dense shapes, never a GQA group, so
        // there are no per-head placements and the stride check has nothing to compare.
        None,
        splitter,
    )
    .map_err(crate::superdsc_error::SuperDscError)?;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
}

/// [`assemble_matmul_seeded`] for the TRUE PER-BATCH (3-D-kernel) batchmatmul —
/// the multi-head attention score/value bmms. See [`matmul_opspec_batched`]: the
/// KERNEL carries the batch dim (`y`) so each head reads its OWN K/V, fixing the
/// confirmed shared-K bug (vs `assemble_matmul_seeded`, whose 2-D kernel is shared).
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_batched_seeded(
    op_name: &str,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    a_name: &str,
    w_name: &str,
    o_name: &str,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let op = matmul_opspec_batched(m, n, k, batch, a_name, w_name, o_name)
        .unwrap_or_else(|e| panic!("assemble_matmul_batched {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_matmul_batched {op_name}: {e}"))
}

/// [`assemble_matmul_seeded`] with per-operand ELEMENT start-offsets — the per-GQA-group
/// attention bmm (the batchmatmul KERNEL is shared across the batch, so one bmm per
/// kv-head group with `batch=gqa` gives each group its OWN K/V via the offsets).
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_off<O: crate::sdsc_abstract::KindTag>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    // The bundle's shared-kernel bmm form — the attention emitter names it once at its own
    // boundary (`SharedKernelBmmForm::of_attn_rows`) and threads it here.
    form: SharedKernelBmmForm,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_off: crate::addr::DevOff,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: crate::addr::DevOff,
    o: &Stk<O>,
    o_off: crate::addr::DevOff,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let a_off = a_off.into_raw_elems();
    let w_off = w_off.into_raw_elems();
    let o_off = o_off.into_raw_elems();

    // TYPED OPERANDS: the activation is RowBlocked, the weight/cache is a Kernel — a `cargo build` error
    // otherwise. The OUTPUT kind is generic (`O`): a projection/score/value output is RowBlocked, but a
    // cache-WRITE (cachewr, matmul-by-identity) outputs the K/V cache as a `Stk<KernelTag>` — so addressing
    // the cache as a non-Kernel does not type-check (the documented attention bug class). Handles carry
    // only `.name()` into `matmul_opspec_off`, so the emit is byte-identical.
    let (a_name, w_name, o_name) = (a.name(), w.name(), o.name());
    let op = matmul_opspec_off::<Fp16>(
        m, n, k, batch, form, a_name, w_name, o_name, a_off, w_off, o_off,
    )
    .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"));
    assemble_from_opspec(op_name, op, sym_id_base, layout)
}

/// [`assemble_matmul_off`] for an op that sweeps only PART of the activation's rows: `m` is what it
/// COMPUTES; the activation's base offset and its head pitch (how many rows the tensor is actually
/// packed with per stick plane) arrive TOGETHER as one [`OperandPlacement`], and stride derivation
/// uses the pitch.
///
/// The row-batched attention fold is the caller. A pass computes one request's row per head (`m = 1`)
/// out of an activation with `mq` rows, and the y-batched form is stick-major — a head is one stick
/// PLANE away, and a plane is `m * stick` elements. Deriving that from `m` puts every head but the
/// first `mq`x too close; the placement's pitch is the tensor's own packing, correct while still
/// computing one row. There is no standalone pitch slot: offset and pitch are both answers of the
/// row law that placed the operand, so the score leg's packing cannot be handed to the value leg.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_off_phys_m<O: crate::sdsc_abstract::KindTag>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: crate::addr::DevOff,
    o: &Stk<O>,
    o_off: crate::addr::DevOff,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let (a_off, w_off, o_off) = (
        a_place.off().into_raw_elems(),
        w_off.into_raw_elems(),
        o_off.into_raw_elems(),
    );
    let (a_name, w_name, o_name) = (a.name(), w.name(), o.name());
    let op = super::opspec::matmul_opspec_off_operands_phys::<Fp16>(
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
        <Fp16 as crate::superdsc_opspec::DataFormat>::DF,
        Some(a_place.head_pitch()),
    )
    .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"));
    assemble_from_opspec(op_name, op, sym_id_base, layout)
}

/// ⭐⭐⭐ [`assemble_matmul_off_phys_m`] WITH THE OUTPUT PLACED BY A LAW TOO — both operands arrive as
/// [`OperandPlacement`]s, so neither one's offset can be paired with another law's pitch.
///
/// The KV cache write is the caller: its two sides are placed by the SAME law at different depths
/// (`mq` rows per source plane, `PLANE_SLOTS` per cache plane), and its output offset is an answer of
/// that law rather than a number the site adds up. `o_place`'s pitch reaches the walk through
/// [`crate::sdsc_abstract::MatY`]'s batch strides, exactly as `a_place`'s does; this entry point is
/// what stops the OFFSET half being taken from somewhere else.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_placed<O: crate::sdsc_abstract::KindTag>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: crate::addr::DevOff,
    o: &Stk<O>,
    o_place: OperandPlacement,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    assemble_matmul_off_phys_m(
        op_name,
        m,
        n,
        k,
        batch,
        form,
        a,
        a_place,
        w,
        w_off,
        o,
        o_place.off(),
        sym_id_base,
        layout,
    )
}

/// Shared tail: work-division folds + emit. Identical for both entry points, so a change to how an
/// OpSpec becomes an EmittedOp cannot apply to one and not the other.
fn assemble_from_opspec(
    op_name: &str,
    op: crate::superdsc_opspec::OpSpec,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"))
}

/// [`assemble_from_opspec`] with a FUSED POINTWISE EPILOGUE chained onto this matmul's own output —
/// the general foundation for every matmul-epilogue fusion in scratchy-superdsc (see
/// `OpSpec::attach_fused_epilogue`), not a
/// one-off for any single caller. `epi` is a typed operand handle (its OWN tag is irrelevant here —
/// only its device NAME is read; `attach_fused_epilogue` clones THIS op's own output shape/layout
/// onto it, matching `bmm.ddl`'s bnA/bnB/bias/resadd convention exactly) at device offset `epi_off`.
///
/// PANICS if `op.time() != 1` (a tiled matmul) — the epilogue operand is not threaded through
/// `rewrite_op_for_time_tile`'s per-trip affine-stride machinery, so fusing onto a tiled matmul would
/// silently address only the first trip. A future tiled target needs that machinery extended first,
/// not this guard removed.
#[allow(clippy::too_many_arguments)]
fn assemble_from_opspec_with_epilogue<E: crate::sdsc_abstract::KindTag>(
    op_name: &str,
    mut op: crate::superdsc_opspec::OpSpec,
    epi: &Stk<E>,
    epi_off: crate::addr::DevOff,
    epi_op_func: crate::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, crate::superdsc_opspec::Scale)],
    broadcast_batch: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    assert_eq!(
        op.time(),
        1,
        "assemble_matmul {op_name}: a fused epilogue needs an untiled (time=1) matmul — got time={}",
        op.time()
    );
    op.attach_fused_epilogue(crate::superdsc_opspec::EpilogueSpecs::One(
        crate::superdsc_opspec::EpilogueSpec {
            operand_name: epi.name().to_string(),
            offset_elems: epi_off.into_raw_elems(),
            op_func: epi_op_func,
            broadcast_dims,
            broadcast_batch,
        },
    ));
    assemble_from_opspec(op_name, op, sym_id_base, layout)
}

/// [`assemble_from_opspec_with_epilogue`] where the epilogue is OPTIONAL — the SAME op, with or
/// without a fused stage, chosen by whether the caller has an operand to chain.
///
/// The online-softmax value leg is the caller. A FOLD block adds the running accumulator it is
/// folding onto; the SEED block has nothing to add, because it ESTABLISHES that accumulator. Those
/// are the same matmul over the same operands at the same offsets — only the addend differs — so the
/// choice is one `Option` at one call site. Spelling it as two full argument lists instead is
/// precisely how a block that should fold gets emitted as one that seeds (which DISCARDS every page
/// folded before it, silently and fluently), because the two lists drift independently.
#[allow(clippy::too_many_arguments)]
fn assemble_from_opspec_maybe_epilogue<E: crate::sdsc_abstract::KindTag>(
    op_name: &str,
    op: crate::superdsc_opspec::OpSpec,
    epi: Option<(&Stk<E>, crate::addr::DevOff)>,
    epi_op_func: crate::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, crate::superdsc_opspec::Scale)],
    broadcast_batch: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    match epi {
        Some((e, off)) => assemble_from_opspec_with_epilogue(
            op_name,
            op,
            e,
            off,
            epi_op_func,
            broadcast_dims,
            broadcast_batch,
            sym_id_base,
            layout,
        ),
        None => assemble_from_opspec(op_name, op, sym_id_base, layout),
    }
}

/// [`assemble_matmul_off`] with an OPTIONAL fused pointwise epilogue — see
/// [`assemble_from_opspec_maybe_epilogue`] for why the epilogue is an `Option` and not two callers.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_off_maybe_epilogue<
    O: crate::sdsc_abstract::KindTag,
    E: crate::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_off: crate::addr::DevOff,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: crate::addr::DevOff,
    o: &Stk<O>,
    o_off: crate::addr::DevOff,
    epi: Option<(&Stk<E>, crate::addr::DevOff)>,
    epi_op_func: crate::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, crate::superdsc_opspec::Scale)],
    broadcast_batch: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let (a_off_raw, w_off_raw, o_off_raw) = (
        a_off.into_raw_elems(),
        w_off.into_raw_elems(),
        o_off.into_raw_elems(),
    );
    let (a_name, w_name, o_name) = (a.name(), w.name(), o.name());
    let op = matmul_opspec_off::<Fp16>(
        m, n, k, batch, form, a_name, w_name, o_name, a_off_raw, w_off_raw, o_off_raw,
    )
    .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"));
    assemble_from_opspec_maybe_epilogue(
        op_name,
        op,
        epi,
        epi_op_func,
        broadcast_dims,
        broadcast_batch,
        sym_id_base,
        layout,
    )
}

/// [`assemble_matmul_off_phys_m`] with an OPTIONAL fused pointwise epilogue — see
/// [`assemble_from_opspec_maybe_epilogue`].
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_off_phys_m_maybe_epilogue<
    O: crate::sdsc_abstract::KindTag,
    E: crate::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: crate::addr::DevOff,
    o: &Stk<O>,
    o_off: crate::addr::DevOff,
    epi: Option<(&Stk<E>, crate::addr::DevOff)>,
    epi_op_func: crate::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, crate::superdsc_opspec::Scale)],
    broadcast_batch: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let (a_off_raw, w_off_raw, o_off_raw) = (
        a_place.off().into_raw_elems(),
        w_off.into_raw_elems(),
        o_off.into_raw_elems(),
    );
    let (a_name, w_name, o_name) = (a.name(), w.name(), o.name());
    let op = super::opspec::matmul_opspec_off_operands_phys::<Fp16>(
        m,
        n,
        k,
        batch,
        form,
        a_name,
        w_name,
        o_name,
        a_off_raw,
        w_off_raw,
        o_off_raw,
        <Fp16 as crate::superdsc_opspec::DataFormat>::DF,
        Some(a_place.head_pitch()),
    )
    .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"));
    assemble_from_opspec_maybe_epilogue(
        op_name,
        op,
        epi,
        epi_op_func,
        broadcast_dims,
        broadcast_batch,
        sym_id_base,
        layout,
    )
}

/// [`assemble_matmul_off`] with a fused pointwise epilogue — see [`assemble_from_opspec_with_epilogue`].
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_off_with_epilogue<
    O: crate::sdsc_abstract::KindTag,
    E: crate::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_off: crate::addr::DevOff,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: crate::addr::DevOff,
    o: &Stk<O>,
    o_off: crate::addr::DevOff,
    epi: &Stk<E>,
    epi_off: crate::addr::DevOff,
    epi_op_func: crate::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, crate::superdsc_opspec::Scale)],
    broadcast_batch: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let (a_off_raw, w_off_raw, o_off_raw) = (
        a_off.into_raw_elems(),
        w_off.into_raw_elems(),
        o_off.into_raw_elems(),
    );
    let (a_name, w_name, o_name) = (a.name(), w.name(), o.name());
    let op = matmul_opspec_off::<Fp16>(
        m, n, k, batch, form, a_name, w_name, o_name, a_off_raw, w_off_raw, o_off_raw,
    )
    .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"));
    assemble_from_opspec_with_epilogue(
        op_name,
        op,
        epi,
        epi_off,
        epi_op_func,
        broadcast_dims,
        broadcast_batch,
        sym_id_base,
        layout,
    )
}

/// [`assemble_matmul_off_phys_m`] with a fused pointwise epilogue — see
/// [`assemble_from_opspec_with_epilogue`].
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_off_phys_m_with_epilogue<
    O: crate::sdsc_abstract::KindTag,
    E: crate::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: crate::addr::DevOff,
    o: &Stk<O>,
    o_off: crate::addr::DevOff,
    epi: &Stk<E>,
    epi_off: crate::addr::DevOff,
    epi_op_func: crate::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, crate::superdsc_opspec::Scale)],
    broadcast_batch: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let (a_off_raw, w_off_raw, o_off_raw) = (
        a_place.off().into_raw_elems(),
        w_off.into_raw_elems(),
        o_off.into_raw_elems(),
    );
    let (a_name, w_name, o_name) = (a.name(), w.name(), o.name());
    let op = super::opspec::matmul_opspec_off_operands_phys::<Fp16>(
        m,
        n,
        k,
        batch,
        form,
        a_name,
        w_name,
        o_name,
        a_off_raw,
        w_off_raw,
        o_off_raw,
        <Fp16 as crate::superdsc_opspec::DataFormat>::DF,
        Some(a_place.head_pitch()),
    )
    .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"));
    assemble_from_opspec_with_epilogue(
        op_name,
        op,
        epi,
        epi_off,
        epi_op_func,
        broadcast_dims,
        broadcast_batch,
        sym_id_base,
        layout,
    )
}

/// ⭐⭐⭐⭐⭐ [`assemble_matmul_off_phys_m_with_epilogue`] WITH ITS KERNEL READ THROUGH AN INDEX — the
/// hardware gather, on the leg that reads KV.
///
/// This is what indirect addressing is FOR in attention: the score leg's Kᵗ operand stops being reached
/// through a base this launch baked and becomes `addr = idx * skip_addr + base`, so every row reads its
/// OWN keys from one launch instead of one launch per request. `SessionKv::fold_requests` names the cost
/// that removes — *"ONE LAUNCHED OP HAS ONE PAGE BASE, so B requests cannot share a pass: the fold runs
/// `pages x requests` times"*.
///
/// `gather` is `(index operand name, the kernel axis one index entry advances)`. Which OPERAND is the
/// kernel is not this function's to say — `matmul_opspec_off_operands_phys_gathered` attaches the pair
/// inside the builder that ordered `[a, w, o]`, so the position is answered where it is decided, and
/// the axis is a closed [`crate::superdsc_opspec::KernelAxis`] rather than a dim name that could miss.
/// Together those two make the declaration either right or unwritable: an index tensor with no
/// declaration — which dxp reads as an ordinary arithmetic input — is not reachable from here.
///
/// ⛔ THE ONE THING THAT MAKES OR BREAKS IT IS `skip_addr`, AND WE DO NOT SUPPLY IT. dxp derives it from
/// the value tensor's own per-dim capacities, so it equals ONE ENTRY'S SIZE, and the index steps whatever
/// that is. Declare an entry smaller than the pool's real page stride and every index step lands a
/// fraction of a page in — fluent text built from another request's keys, from a CLEAN BAKE, with every
/// counter green. `zz_skip_addr_computed_from_our_own_emission.rs` recomputes dxp's own derivation off
/// the emitted JSON for exactly this reason. ⛔ CHECK IT BEFORE BAKING; a card run cannot tell you.
///
/// `None` emits byte-for-byte what the ungathered assembler does — the indirect fields are absent from
/// the descriptor when undeclared, which the gather-free test asserts rather than assumes.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_off_phys_m_with_epilogue_gathered<
    O: crate::sdsc_abstract::KindTag,
    E: crate::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: crate::addr::DevOff,
    o: &Stk<O>,
    o_off: crate::addr::DevOff,
    epi: &Stk<E>,
    epi_off: crate::addr::DevOff,
    epi_op_func: crate::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, crate::superdsc_opspec::Scale)],
    broadcast_batch: bool,
    gather: Option<crate::superdsc_opspec::GatherIndex>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> crate::emit::EmittedOp {
    // Same three offsets, from the same places, as the ungathered sibling — the activation's comes from
    // its PLACEMENT, so this entry point cannot pair a law's pitch with a hand-added offset.
    let (a_off_raw, w_off_raw, o_off_raw) = (
        a_place.off().into_raw_elems(),
        w_off.into_raw_elems(),
        o_off.into_raw_elems(),
    );
    let (a_name, w_name, o_name) = (a.name(), w.name(), o.name());
    // ONE failure channel: the gather is attached inside the builder, so a shortfall arrives as the
    // same `Err` the ungathered path already reports, at the same `cargo build`.
    let op = super::opspec::matmul_opspec_off_operands_phys_gathered::<Fp16>(
        m,
        n,
        k,
        batch,
        form,
        a_name,
        w_name,
        o_name,
        a_off_raw,
        w_off_raw,
        o_off_raw,
        <Fp16 as crate::superdsc_opspec::DataFormat>::DF,
        Some(a_place.head_pitch()),
        gather,
    )
    .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"));
    assemble_from_opspec_with_epilogue(
        op_name,
        op,
        epi,
        epi_off,
        epi_op_func,
        broadcast_dims,
        broadcast_batch,
        sym_id_base,
        layout,
    )
}

/// [`assemble_matmul_off`] for the TRUE PER-BATCH (3-D-kernel) batchmatmul: ONE op covering
/// `batch` kv-heads instead of `batch · gqa` per-head ops. See [`matmul_opspec_batched_off`] for
/// the offset / `kernel_device_extent` contract.
///
/// Same typed-operand discipline as [`assemble_matmul_off`] — activation is `RowBlocked`, the
/// K/V cache is a `Kernel`, so addressing the cache as a non-Kernel does not type-check.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_batched_off<O: crate::sdsc_abstract::KindTag>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    a: &Stk<crate::sdsc_abstract::RowBlockedTag>,
    a_off: u32,
    w: &Stk<crate::sdsc_abstract::KernelTag>,
    w_off: u32,
    kernel_device_extent: Option<(&'static str, u32)>,
    o: &Stk<O>,
    o_off: u32,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let (a_name, w_name, o_name) = (a.name(), w.name(), o.name());
    let op = super::opspec::matmul_opspec_batched_off(
        m,
        n,
        k,
        batch,
        a_name,
        w_name,
        o_name,
        a_off,
        w_off,
        o_off,
        kernel_device_extent,
    )
    .unwrap_or_else(|e| panic!("assemble_matmul_batched_off {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::emit::emit_sdsc_tiled(op_name, &op, &folds, sym_id_base, layout)
        .unwrap_or_else(|e| panic!("assemble_matmul_batched_off {op_name}: {e}"))
}

/// THE CONTROL SET FOR [`out_width_the_weight_holds`], and the reason it is not vacuous.
///
/// ⛔⛔⛔ THE GUARD UNDER TEST EXISTS BECAUSE AN INVENTED WIDTH IS A SILENT WRONG ANSWER, so a
/// test that only pins "it narrows" would be satisfied by a function that narrows ALWAYS — which is
/// the opposite defect and just as silent. Three cases, and each moves exactly ONE thing:
///
/// | case | weight footprint | holds | verdict | what it pins |
/// |---|---|---|---|---|
/// | (a) | 65536 B | 256 cols | caps to **256** | the pad was fictional, so it is dropped |
/// | (b) | 131072 B | 512 cols | keeps **512** | the pad was REAL, so nothing moves |
/// | (c) | 16384 B | 64 cols | keeps **512** | the shortfall is NOT this rule's pad — left to the footprint guard |
///
/// (a)→(b) moves only the footprint and must flip the verdict, so the cap cannot be unconditional.
/// (a)→(c) moves only the footprint again and must flip it BACK, so the cap cannot be "narrow to
/// whatever is there": [`DeviceWidth::for_output`] at `(64, 64, 128)` is 64 — its MACs are under the
/// `2^20` bump floor, so no pad is applied — which does not reproduce the 512 we were handed. That is
/// a real over-run, and it must still reach `resolve_seg_base` and be refused BY NAME rather than
/// quietly satisfied by a narrower descriptor computing fewer columns than the program states. Drop
/// the reproduction check and (c) goes red while (a) and (b) stay green.
///
/// The shape is the MEASURED one: `m=64 n=256 k=128` is the small SwiGLU MLP blocking, whose
/// `for_output` bumps 4 sticks to the 8-core floor (`n_dev 512`) and whose weight the caller supplies
/// at its logical `[256, 128]` — the emission that asked for `0B + 131072B` of a 65536 B buffer.
///
/// ⚠️ WHAT (b) ACTUALLY DISCRIMINATES, MEASURED BY MUTATION RATHER THAN ASSUMED. (b) does NOT
/// catch a cap that narrows unconditionally: at (b) the held width EQUALS `n`, so returning `held`
/// and returning `n` are the same answer, and a mutant with `held < n` and the reproduction check
/// both deleted leaves (b) green (it is (c) that goes red there). What (b) catches is a `held`
/// computed WRONG — proven by mutating this function's hardcoded two-bytes-per-element to four,
/// which turns (b) red. That is not a hypothetical mutation: it is precisely the latent fp8 hazard
/// this function's own doc names, where a packed one-byte weight makes the same arithmetic compute
/// a held width twice the truth. So (b) is the guard on the FOOTPRINT ARITHMETIC, and (c) is the
/// guard on the narrowing CONDITION.
#[cfg(test)]
mod out_width_caps {
    use super::*;
    use crate::place::{PlaceId, act_name};
    use crate::placement::{SegRole, TensorPlacement};

    /// The measured small-MLP contraction: `[64, 128] · [128, 256]`, whose device `out` width is the
    /// 8-stick occupancy floor rather than the logical 4 sticks.
    const M: u32 = 64;
    const K: u32 = 128;
    const N_DEV: u32 = 512;

    /// A layout in which `t1` — the weight — has exactly `size` bytes reserved. Only the two things
    /// the guard reads are populated: the name→id map behind `id_of`, and the placement's `size`.
    fn layout_holding(size: u64) -> BundleLayout {
        let mut l = BundleLayout::default();
        l.ids.borrow_mut().insert(act_name(1), PlaceId::Act(1));
        l.placements.insert(
            1,
            TensorPlacement {
                tid: 1,
                role: SegRole::Weight,
                segment: 1,
                bank: 0,
                offset: 0,
                size,
            },
        );
        l
    }

    /// The width the guard returns for a weight placement of `size` bytes.
    fn width_for(size: u64) -> u32 {
        let l = layout_holding(size);
        out_width_the_weight_holds(Some(&l), &act_name(1), M, N_DEV, K)
    }

    /// ⛔ CASE (a) — THE MEASURED OVER-RUN. The weight is the caller's own `[256, 128]` tensor, so the
    /// layout reserved `256 · 128 · 2 = 65536` B and nobody made the pad real. The emitted kernel must
    /// MAC over the 256 columns that exist, not the 512 the occupancy rule invented.
    #[test]
    fn an_under_reserved_weight_caps_the_out_width_at_what_it_holds() {
        assert_eq!(
            width_for(256 * 128 * 2),
            256,
            "a weight whose placement holds 256 columns cannot be read at 512 — that is the \
             measured `0B + 131072B exceeds footprint 65536B` refusal"
        );
    }

    /// ⭐ CASE (b) — THE CONTROL THAT MAKES (a) NON-VACUOUS. The full-model path DOES reserve the
    /// padded weight (the worker zero-pads the staged bytes by this same rule), so the cap must be a
    /// no-op there. If this ever fails, the fix has started narrowing the shipping emission — which
    /// the 8b fp16 emitted-bundle fingerprint would also catch, far later and far more expensively.
    #[test]
    fn a_fully_reserved_weight_keeps_the_padded_width() {
        assert_eq!(
            width_for(u64::from(N_DEV) * 128 * 2),
            N_DEV,
            "when the layout reserves the padded weight the pad is REAL and nothing may move"
        );
    }

    /// ⛔ CASE (c) — A SHORTFALL THAT IS NOT THIS RULE'S PAD IS NOT QUIETLY SATISFIED. One stick of
    /// weight under a 512-column access is a mis-sized placement or a wrong shape, not an occupancy
    /// pad: re-padding 64 gives 64 (its MACs are below the `2^20` bump floor), which does not
    /// reproduce 512. So `n` comes back UNCHANGED and the footprint guard still gets to refuse it by
    /// name. A version of this function without the reproduction proof returns 64 here, and this test
    /// is the only thing that notices.
    #[test]
    fn a_shortfall_that_is_not_this_rules_pad_is_left_to_the_footprint_guard() {
        assert_eq!(
            width_for(64 * 128 * 2),
            N_DEV,
            "64 does not re-pad to 512, so this is a real over-run and must NOT be narrowed away"
        );
    }
}

// ⛔⛔⛔ WHAT WAS HERE, AND WHY IT IS GONE: `assemble_matmul_requests_off_maybe_epilogue` — the
// per-batch-3-D-KERNEL bmm carrying a `RequestAxis`, built for the collapsed fold's two legs. Its form
// is REFUTED ON CARD. A `[y,in,out]` kernel gives every core its own weight start, so with `mb` pinned
// to 1 and a one-stick `out` the `y` split IS `numCoresUsed_`, and the launch faulted at
// `job_bin_ptr + numCoresUsed_*128` — one flit past the program's per-core patch table — at rungs 2, 4
// and 8 alike, syndrome `0xc00` / locator `0x2` / cases `[PrepZeroFlitCnt,PrepSwVer]` bit-identical with
// only the index moving. Rung 4 closed it: `{mb:1, y:4}` is the on-hardware-proven solo-decode split, so
// the split VALUE is exonerated and the kernel RANK is what the address is made of.
//
// The collapsed fold now emits one op per (request, kv head) through
// `assemble_matmul_off_phys_m_with_epilogue` — the SHIPPED shared-2-D-kernel form, `y` on the GQA group —
// and reaches request `r`'s block by a baked offset (`GatherScratch::kernel_row_off`). Nothing in
// attention declares a 3-D kernel any more, which is asserted on the emitted descriptors by
// `zz_the_declared_y_of_every_gathered_op`.
