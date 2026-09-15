//! Matmul family: the `assemble_matmul*` wrappers (opspec builder + `emit_sdsc_tiled`). See
//! `super`'s module doc.

use super::opspec::{matmul_opspec, matmul_opspec_batched, matmul_opspec_off, matmul_opspec_split};
use super::walk::SharedKernelBmmForm;
use crate::lower_subtile_tape_to_superdsc::{BundleLayout, EmittedOp};
use scratchy_subtile::sdsc_abstract::{MatK, MatM, MatN, MatY, OperandPlacement, Stk};
use scratchy_subtile::superdsc_opspec::{DataFormat, Fp16, SdscFoldSet};

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
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    o: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
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
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    o: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    // TYPED OPERANDS (compile-time addressing safety): a matmul's activation is RowBlocked, its weight
    // is a Kernel, its output is RowBlocked — a `cargo build` type error otherwise (you cannot hand a
    // `Flat`/`RowScalar` tensor here). The handles carry only the emitter NAME into `matmul_opspec`
    // (which owns df/shape), so the emit is byte-identical — the types are a pure addressing guard.
    let op = matmul_opspec(m, n, k, batch, a.name(), w.name(), o.name())
        .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"));
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_matmul {op_name}: {e}"))
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
        &[scratchy_subtile::superdsc_opspec::ItDim],
        u32,
    ) -> std::collections::BTreeMap<&'static str, u32>,
) -> Result<EmittedOp, scratchy_subtile::superdsc_error::SuperDscError> {
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
    .map_err(scratchy_subtile::superdsc_error::SuperDscError)?;
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
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
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_matmul_batched {op_name}: {e}"))
}

/// [`assemble_matmul_seeded`] with per-operand ELEMENT start-offsets — the per-GQA-group
/// attention bmm (the batchmatmul KERNEL is shared across the batch, so one bmm per
/// kv-head group with `batch=gqa` gives each group its OWN K/V via the offsets).
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_off<O: scratchy_subtile::sdsc_abstract::KindTag>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    // The bundle's shared-kernel bmm form — the attention emitter names it once at its own
    // boundary (`SharedKernelBmmForm::of_attn_rows`) and threads it here.
    form: SharedKernelBmmForm,
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    a_off: scratchy_subtile::addr::DevOff,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    w_off: scratchy_subtile::addr::DevOff,
    o: &Stk<O>,
    o_off: scratchy_subtile::addr::DevOff,
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
pub fn assemble_matmul_off_phys_m<O: scratchy_subtile::sdsc_abstract::KindTag>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    w_off: scratchy_subtile::addr::DevOff,
    o: &Stk<O>,
    o_off: scratchy_subtile::addr::DevOff,
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
        <Fp16 as scratchy_subtile::superdsc_opspec::DataFormat>::DF,
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
/// [`scratchy_subtile::sdsc_abstract::MatY`]'s batch strides, exactly as `a_place`'s does; this entry point is
/// what stops the OFFSET half being taken from somewhere else.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_placed<O: scratchy_subtile::sdsc_abstract::KindTag>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    w_off: scratchy_subtile::addr::DevOff,
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
    op: scratchy_subtile::superdsc_opspec::OpSpec,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> EmittedOp {
    let folds = SdscFoldSet::new(op.iter.cores_used());
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
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
fn assemble_from_opspec_with_epilogue<E: scratchy_subtile::sdsc_abstract::KindTag>(
    op_name: &str,
    mut op: scratchy_subtile::superdsc_opspec::OpSpec,
    epi: &Stk<E>,
    epi_off: scratchy_subtile::addr::DevOff,
    epi_op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, scratchy_subtile::superdsc_opspec::Scale)],
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
    op.attach_fused_epilogue(scratchy_subtile::superdsc_opspec::EpilogueSpecs::One(
        scratchy_subtile::superdsc_opspec::EpilogueSpec {
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
fn assemble_from_opspec_maybe_epilogue<E: scratchy_subtile::sdsc_abstract::KindTag>(
    op_name: &str,
    op: scratchy_subtile::superdsc_opspec::OpSpec,
    epi: Option<(&Stk<E>, scratchy_subtile::addr::DevOff)>,
    epi_op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, scratchy_subtile::superdsc_opspec::Scale)],
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
    O: scratchy_subtile::sdsc_abstract::KindTag,
    E: scratchy_subtile::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    a_off: scratchy_subtile::addr::DevOff,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    w_off: scratchy_subtile::addr::DevOff,
    o: &Stk<O>,
    o_off: scratchy_subtile::addr::DevOff,
    epi: Option<(&Stk<E>, scratchy_subtile::addr::DevOff)>,
    epi_op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, scratchy_subtile::superdsc_opspec::Scale)],
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
    O: scratchy_subtile::sdsc_abstract::KindTag,
    E: scratchy_subtile::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    w_off: scratchy_subtile::addr::DevOff,
    o: &Stk<O>,
    o_off: scratchy_subtile::addr::DevOff,
    epi: Option<(&Stk<E>, scratchy_subtile::addr::DevOff)>,
    epi_op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, scratchy_subtile::superdsc_opspec::Scale)],
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
        <Fp16 as scratchy_subtile::superdsc_opspec::DataFormat>::DF,
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
    O: scratchy_subtile::sdsc_abstract::KindTag,
    E: scratchy_subtile::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    a_off: scratchy_subtile::addr::DevOff,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    w_off: scratchy_subtile::addr::DevOff,
    o: &Stk<O>,
    o_off: scratchy_subtile::addr::DevOff,
    epi: &Stk<E>,
    epi_off: scratchy_subtile::addr::DevOff,
    epi_op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, scratchy_subtile::superdsc_opspec::Scale)],
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
    O: scratchy_subtile::sdsc_abstract::KindTag,
    E: scratchy_subtile::sdsc_abstract::KindTag,
>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    form: SharedKernelBmmForm,
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    a_place: OperandPlacement,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
    w_off: scratchy_subtile::addr::DevOff,
    o: &Stk<O>,
    o_off: scratchy_subtile::addr::DevOff,
    epi: &Stk<E>,
    epi_off: scratchy_subtile::addr::DevOff,
    epi_op_func: scratchy_subtile::superdsc_opspec::EpilogueOpFunc,
    broadcast_dims: &[(&str, scratchy_subtile::superdsc_opspec::Scale)],
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
        <Fp16 as scratchy_subtile::superdsc_opspec::DataFormat>::DF,
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

/// [`assemble_matmul_off`] for the TRUE PER-BATCH (3-D-kernel) batchmatmul: ONE op covering
/// `batch` kv-heads instead of `batch · gqa` per-head ops. See [`matmul_opspec_batched_off`] for
/// the offset / `kernel_device_extent` contract.
///
/// Same typed-operand discipline as [`assemble_matmul_off`] — activation is `RowBlocked`, the
/// K/V cache is a `Kernel`, so addressing the cache as a non-Kernel does not type-check.
#[allow(clippy::too_many_arguments)]
pub fn assemble_matmul_batched_off<O: scratchy_subtile::sdsc_abstract::KindTag>(
    op_name: &str,
    m: MatM,
    n: MatN,
    k: MatK,
    batch: MatY,
    a: &Stk<scratchy_subtile::sdsc_abstract::RowBlockedTag>,
    a_off: u32,
    w: &Stk<scratchy_subtile::sdsc_abstract::KernelTag>,
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
    crate::lower_subtile_tape_to_superdsc::emit_sdsc_tiled(
        op_name,
        &op,
        &folds,
        sym_id_base,
        layout,
    )
    .unwrap_or_else(|e| panic!("assemble_matmul_batched_off {op_name}: {e}"))
}
