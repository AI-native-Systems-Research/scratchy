//! ⭐⭐⭐⭐⭐ A BATCHED GATHERED CONTRACTION — the gather's ENTRY AXIS IS THE DOT'S BATCH AXIS.
//!
//! The rung-4 materializer ([`super::whole_function`]'s `gathered_matmul_materializes`) serves the
//! one gathered-contraction frame the vendor's own fixtures and the embedding spell: gathered rows
//! materialized `[entries, head]` and read `[k, n]` — the ENTRY AXIS AS K, a lookup table whose
//! rows are the contraction's K extent. A `linalg.batch_matmul` states a different frame:
//!
//! ```python
//! w = tl.reshape(desc.gather(ids, 0), [BATCH, K, N])
//! c = tl.dot(a, w)          # [BATCH, M, K] x [BATCH, K, N]
//! ```
//!
//! Here the entry axis is the **BATCH**: each gathered entry is its own `[K, N]` tile, `page`
//! batch rows per entry. No 2-D `[k, n]` reading of the gathered bytes expresses that — A's batch
//! axis folds into M (every A row owns an output row; the door already reads A flat
//! `[batch·m, k]`), but B's batch axis is a per-batch tile, structurally outside `[k, n]`. That
//! asymmetry is why this is a door arm and not a framing choice: the kernel cannot restate around
//! it without changing the math.
//!
//! # THE ONE FRAME BOTH SIDES SPEAK: C ORDER
//!
//! * **THE COPY** ([`copy_opspecs`]) is this crate's own gather-copy door
//!   ([`gather_copy_opspec`]) in its one-stick sub-row form: each gathered entry is
//!   `entry_elems / 64` sub-rows of ONE stick, so `skip_addr` is one whole table row and the
//!   destination is CONTIGUOUS — sub-row `s` of entry `e` lands at `(e·page + s)·64`. That is the
//!   gathered rows in C order, element `(b, k, n)` at `b·K·N + k·N + n` with `b = entry·page +
//!   pos`: the row-major `[BATCH, K, N]` tensor the reshape names.
//! * **THE DOT** ([`dot_opspecs`]) is the per-batch bmm ([`matmul_opspec_batched_off`]) —
//!   `INPUT [y, mb, in]`, `KERNEL [y, in, out]`, `OUTPUT [y, mb, out]`. A rank-3 view in that
//!   emitter is ROW-MAJOR ([`StickLayout::for_view_df`] classifies it `Flat`), so the KERNEL view
//!   at the buffer's own physical extents `[BATCH, K, N]` reads `(b, k, n)` at `b·K·N + k·N + n`
//!   — the copy's bytes, the same fact stated by the stride derivation. The physical `in`/`out`
//!   extents are declared ([`AnyTensorArg::set_device_extent`]) so the `y` stride is `K·N` and the
//!   `in` stride `N` — the buffer's, not the tile's.
//!
//! # ⛔ THE ACTIVATION AND OUTPUT ARE STICK-MAJOR, WHICH IS WHY THE DOT IS CUT PER STICK
//!
//! The whole-function door mints every rank-3 value FLAT-FRAMED and stick-major —
//! `[batch·m, C]` row-blocked, `(r, c)` at `(c/64)·(rows·64) + r·64 + c%64`. A row-major rank-3
//! view of that buffer agrees with it only inside ONE stick column, so the dot is emitted as one
//! op per (contraction stick `ks`, output stick `ns`): each op's A and OUT a single-stick column
//! block of the flat buffer (its base `ks·rows·64` / `ns·rows·64`) and its KERNEL the `[64, 64]`
//! sub-tile at `ks·64·N + ns·64` of the row-major `[BATCH, K, N]` scratch. Contraction sticks
//! after the first ACCUMULATE through the fused `stridedadd` epilogue reading the output's own
//! column block — `out_ns = Σ_ks A_ks · W_(ks, ns)`. The cut is a correctness law at this frame,
//! not a tuning choice: a multi-stick `in` or `out` in one rank-3 op would walk the stick-major
//! A/OUT row-major and read another row's sticks.
//!
//! # ⛔ THE WIRE ORDER — every operand of the dot is serialized INNERMOST-FIRST
//!
//! dxp reads `layoutDimOrder_` idx 0 as INNERMOST (`perfdsc/perfLdsInfo.h:29`,
//! `dsc/dataOpDsc.h:347`) and honours a physical extent wider than the sweep only as a
//! `backGapCore_` gap (`sizesWithGaps_`, dsc2.cpp:2899) — the walks this crate states are
//! outermost-first, and the card-verified parity of every OTHER op rests on dxp's internal
//! reconstruction, which a per-batch 3-D kernel with declared strides bypasses. Every operand
//! here carries [`TensorArg::with_dxp_innermost_first`] (the wire fields reverse, the address
//! model does not) and the KERNEL's physical extents ride as back gaps.
//!
//! # THE HONESTY ROW
//!
//! Nothing here names attention. The vocabulary is the pool's and the dot's own: `entries`,
//! `page`, `k`, `n`. A per-token MoE gather-dot — activations `[tokens, m, k]` against gathered
//! expert weights `[tokens, k, n]`, one expert per token — is the same frame and is pinned by the
//! tests below at a geometry no attention kernel states.

use crate::ir::bridge::tiled_op_sdsc_op::matmul_opspec_batched_off;
use crate::sdsc_abstract::{CopyDims, MatK, MatM, MatN, MatY, QueryRowCount};
use crate::superdsc_opspec::{
    DestEntry, EpilogueOpFunc, EpilogueSpec, EpilogueSpecs, GatherIndex, GatherRun, OpSpec,
    PageExtent,
};
use crate::work::FP16_ELEMS_PER_STICK;

/// One fp16 stick, in elements.
const STK: u32 = FP16_ELEMS_PER_STICK;

/// ⭐ THE BATCHED GATHERED DOT'S GEOMETRY, as the program states it — every number read off the
/// op's own operand types and the gather's own tile, checked against each other at the one door
/// ([`BatchedGathered::of`]), so no builder below can be handed a frame the bytes do not have.
///
/// The vocabulary is POOL AND DOT GEOMETRY, nothing else: `batch` the reshape's leading extent
/// (and A's), `m` the query rows per batch, `k` the contraction, `n` the output width, `entries`
/// the gathered table rows, `page` the batch rows per entry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BatchedGathered {
    /// The reshape's batch rows — A's `dims[0]`, B's `dims[0]`.
    pub batch: u32,
    /// The query rows per batch — A's `dims[1]`.
    pub m: u32,
    /// The contraction — A's `dims[2]`, B's `dims[1]`.
    pub k: u32,
    /// The output width — B's `dims[2]`.
    pub n: u32,
    /// The gathered table rows — the indirect tile's own row count.
    pub entries: u32,
    /// Batch rows per gathered entry — `batch / entries`.
    pub page: u32,
}

impl BatchedGathered {
    /// The door. `a_dims`/`b_dims` are the dot's own operand types (B through its reshape),
    /// `entries` the gather's row count, `entry_elems` the gathered load's own per-row width.
    ///
    /// Refuses, naming both numbers: a non-rank-3 operand, a batch or contraction disagreement, a
    /// batch the entries do not divide, an entry whose `page·K·N` is not the load's own row width,
    /// and any extent that is not whole sticks (the cut addresses stick columns).
    pub fn of(
        a_dims: &[i64],
        b_dims: &[i64],
        entries: u32,
        entry_elems: u32,
    ) -> Result<BatchedGathered, String> {
        let [ab, am, ak] = a_dims[..] else {
            return Err(format!(
                "the batched gathered dot's A must be rank-3 [batch, m, k]; it states {a_dims:?}"
            ));
        };
        let [bb, bk, bn] = b_dims[..] else {
            return Err(format!(
                "the batched gathered dot's B must be rank-3 [batch, k, n]; it states {b_dims:?}"
            ));
        };
        if ab != bb {
            return Err(format!(
                "the batched gathered dot's A batch {ab} and B batch {bb} disagree"
            ));
        }
        if ak != bk {
            return Err(format!(
                "the batched gathered dot's A contraction {ak} and B contraction {bk} disagree"
            ));
        }
        let to_u32 = |v: i64, what: &str| {
            u32::try_from(v)
                .map_err(|_| format!("the batched gathered dot's {what} {v} is not a u32"))
        };
        let (batch, m, k, n) = (
            to_u32(ab, "batch")?,
            to_u32(am, "m")?,
            to_u32(ak, "k")?,
            to_u32(bn, "n")?,
        );
        if entries == 0 || !batch.is_multiple_of(entries) {
            return Err(format!(
                "the batched gathered dot's batch {batch} is not a whole number of the gather's \
                 {entries} entries — a batch row would straddle two gathered table rows"
            ));
        }
        let page = batch / entries;
        let per_entry = (page as u64) * (k as u64) * (n as u64);
        if per_entry != entry_elems as u64 {
            return Err(format!(
                "the batched gathered dot reads {page} batch row(s) of [{k}, {n}] per gathered \
                 entry = {per_entry} elements, but the gathered load states {entry_elems} per row \
                 — the reshape would frame bytes the gather did not fetch"
            ));
        }
        for (what, v) in [("k", k), ("n", n)] {
            if v == 0 || !v.is_multiple_of(STK) {
                return Err(format!(
                    "the batched gathered dot's {what} = {v} is not a whole number of \
                     {STK}-element sticks — the per-stick cut has no column to address"
                ));
            }
        }
        Ok(BatchedGathered {
            batch,
            m,
            k,
            n,
            entries,
            page,
        })
    }

    /// One gathered table row's elements — `page·k·n`.
    pub fn entry_elems(&self) -> u32 {
        self.page * self.k * self.n
    }

    /// The materialized B's flat-frame footprint `[entries·entry_elems / 64, 64]` — the copy's
    /// sub-rows.
    pub fn copy_rows(&self) -> u32 {
        self.entries * self.entry_elems() / STK
    }

    /// How many ops the dot emits — one per (contraction stick, output stick).
    pub fn dot_ops(&self) -> u32 {
        (self.k / STK) * (self.n / STK)
    }
}

/// ⭐ THE COPY — the gathered rows into a CONTIGUOUS C-order destination, one leg per index
/// stick. Each leg is the gather-copy door's own one-stick sub-row form; the leg's index run
/// starts at the program's own anchor advanced one stick per leg, and its destination lands
/// `leg·cap` entries into the scratch — the two bases the door derives from one another.
pub fn copy_opspecs(
    bg: &BatchedGathered,
    table: &str,
    index: &str,
    dst: &str,
    first_entry: crate::superdsc_opspec::EntryBase,
) -> Result<Vec<OpSpec>, String> {
    let sub_rows_per_entry = bg.entry_elems() / STK;
    let cap = CopyDims::ENTRIES_PER_OP;
    let legs = bg.entries.div_ceil(cap);
    let mut ops = Vec::with_capacity(legs as usize);
    for leg in 0..legs {
        let leg_entries = (bg.entries - leg * cap).min(cap);
        // The leg's index anchor: the program's own, advanced one whole index stick per leg.
        // `of_entries` IS the stick-alignment check — a leg straddling a stick cannot be named.
        let index_base =
            crate::superdsc_opspec::EntryBase::of_entries(first_entry.entries() + leg * cap)?;
        ops.push(crate::ir::bridge::tiled_op_sdsc_op::gather_copy_opspec(
            table,
            dst,
            GatherIndex::of_scratch_rows(
                index.to_string(),
                PageExtent::of_positions(sub_rows_per_entry),
                index_base,
            ),
            crate::sdsc_abstract::GatherCopySpec {
                rows: leg_entries * sub_rows_per_entry,
                cols: STK,
                df: crate::superdsc_opspec::Df::Fp16,
                run: GatherRun::split(DestEntry::of_entries(leg * cap), 0),
                dest: crate::sdsc_abstract::GatherDest::Scratch {
                    rows: bg.copy_rows(),
                    cols: STK,
                },
            },
        )?);
    }
    Ok(ops)
}

/// The stick-major flat frame's column-block base: column stick `s` of an `R`-row tensor starts
/// at `s·R·64` (the row-blocked packing's `(c/64)·(R·64)` term at `c = s·64`).
fn col_block(rows: u32, stick: u32) -> u32 {
    stick * rows * STK
}

/// ⭐ THE DOT — one per-batch bmm per (contraction stick, output stick), in emission order
/// (`ns` outer, `ks` inner, so each output block's accumulation chain is contiguous). See the
/// module doc for the frame law each operand's offset states.
///
/// `a`/`w`/`o` are the operand NAMES; `a_rows`/`o_rows` the flat-frame row counts of the
/// stick-major A and OUT buffers (`batch·m` for both, checked here).
pub fn dot_opspecs(
    bg: &BatchedGathered,
    a: &str,
    a_rows: u32,
    w: &str,
    o: &str,
    o_rows: u32,
) -> Result<Vec<OpSpec>, String> {
    let rows = bg.batch * bg.m;
    if a_rows != rows || o_rows != rows {
        return Err(format!(
            "the batched gathered dot's A ({a_rows} rows) and OUT ({o_rows} rows) must both be \
             the flat frame's batch·m = {rows} rows — the per-stick column blocks are addressed \
             at that row count"
        ));
    }
    let mut ops = Vec::with_capacity(bg.dot_ops() as usize);
    for ns in 0..bg.n / STK {
        for ks in 0..bg.k / STK {
            let mut op = matmul_opspec_batched_off(
                MatM::of_query_rows(QueryRowCount::of_mq(bg.m)),
                MatN::of_out_features(STK),
                MatK::of_in_features(STK),
                MatY::of_heads(bg.batch),
                a,
                w,
                o,
                col_block(rows, ks),
                ks * STK * bg.n + ns * STK,
                col_block(rows, ns),
                None,
            )?;
            // ⭐ THE KERNEL'S PHYSICAL `in`/`out` EXTENTS — the row-major `[batch, k, n]` buffer
            // the copy wrote — so its `y` stride is `k·n` and its `in` stride `n`, not the tile's.
            let kernel = op
                .args
                .get_mut(1)
                .ok_or_else(|| "the batched bmm has no KERNEL operand".to_string())?;
            kernel.set_device_extent("in", bg.k);
            kernel.set_device_extent("out", bg.n);
            // ⭐⭐⭐ AND EVERY OPERAND IS SERIALIZED IN dxp's OWN (INNERMOST-FIRST) ORDER, its
            // physical extents as back gaps — see the module doc and
            // [`TensorArg::with_dxp_innermost_first`]. The address model (`per_core_addr`, the
            // byte gates below) reads `layout` unchanged, in walk order.
            for arg in op.args.iter_mut() {
                arg.set_dxp_innermost_first();
            }
            if op.time() != 1 {
                return Err(format!(
                    "the batched gathered dot's [{}, {}, {STK}] x [{STK}, {STK}] stick op needs {} \
                     time trips — the fused accumulate cannot ride a time-tiled matmul",
                    bg.batch,
                    bg.m,
                    op.time()
                ));
            }
            if ks > 0 {
                // ⭐ THE ACCUMULATE: `out_ns += A_ks · W_(ks, ns)`, the prior partial read from
                // the output's own column block (same name, same offset) — the fused
                // `stridedadd` epilogue `bmm.ddl` itself exposes.
                op.attach_fused_epilogue(EpilogueSpecs {
                    one: EpilogueSpec {
                        operand_name: o.to_string(),
                        offset_elems: col_block(rows, ns),
                        op_func: EpilogueOpFunc::StridedAdd,
                        broadcast_dims: &[],
                        broadcast_batch: false,
                    },
                });
            }
            ops.push(op);
        }
    }
    Ok(ops)
}

/// ⭐⭐⭐ THE DOT'S EMISSION — the door's whole-region checks, then one bmm per (contraction
/// stick, output stick) through the shared fold machinery ([`emit_sdsc_tiled`]), each op
/// LX-consulted like every op the door emits.
///
/// `per_op` is `[A, B, OUT]` with B the C-order copy the materializer wrote. A and OUT must be
/// WHOLE buffers at corner 0 — the per-stick column blocks address `[batch·m, k]` / `[batch·m, n]`
/// stick-major at their base — and B must be the whole C-order scratch (its region's own rows are
/// the geometry's `copy_rows`), or the door's region and the geometry disagree about which buffer
/// the dot reads.
pub fn emit_dot(
    name: &str,
    bg: &BatchedGathered,
    per_op: &[crate::emit::lower_ktir_to_superdsc::Region],
    sym_id_base: &mut i64,
    layout: Option<&crate::placement::BundleLayout>,
) -> Result<Vec<crate::emit::EmittedOp>, crate::emit::lower_ktir_to_superdsc::Error> {
    use crate::emit::lower_ktir_to_superdsc::{Error, err};
    let [a, w, o] = per_op[..] else {
        return err(format!(
            "{name}: the batched gathered dot expects [A, B, OUT], found {} region(s)",
            per_op.len()
        ));
    };
    if a.is_out || w.is_out || !o.is_out {
        return err(format!(
            "{name}: the batched gathered dot's regions are not [input, input, output]"
        ));
    }
    let whole = |r: &crate::emit::lower_ktir_to_superdsc::Region| {
        r.r_start == 0 && r.c_start == 0 && r.r_len == r.v_rows && r.c_len == r.v_cols
    };
    for (what, r, cols) in [("A", a, bg.k), ("OUT", o, bg.n)] {
        if !whole(&r) || r.c_len != cols {
            return err(format!(
                "{name}: the batched gathered dot's {what} t{} is the window [{}, {}] at ({}, {}) \
                 of a [{}, {}] buffer — the per-stick column blocks address a WHOLE [batch·m, \
                 {cols}] stick-major buffer at its base",
                r.tid, r.r_len, r.c_len, r.r_start, r.c_start, r.v_rows, r.v_cols
            ));
        }
    }
    if !whole(&w) || w.r_len != bg.copy_rows() || w.c_len != STK {
        return err(format!(
            "{name}: the batched gathered dot's B t{} is the window [{}, {}] at ({}, {}) of a \
             [{}, {}] buffer, not the whole [{}, {STK}] C-order scratch the copy wrote — the \
             door's region and the stated geometry disagree about the buffer the dot reads",
            w.tid,
            w.r_len,
            w.c_len,
            w.r_start,
            w.c_start,
            w.v_rows,
            w.v_cols,
            bg.copy_rows()
        ));
    }
    let ops =
        dot_opspecs(bg, &a.name(), a.v_rows, &w.name(), &o.name(), o.v_rows).map_err(|msg| {
            Error {
                message: format!("{name}: {msg}"),
            }
        })?;
    let ks_count = bg.k / STK;
    let mut out = Vec::with_capacity(ops.len());
    for (i, mut op) in ops.into_iter().enumerate() {
        let (ns, ks) = (i as u32 / ks_count, i as u32 % ks_count);
        crate::emit::apply_lx_residency(&mut op, layout);
        let folds = crate::superdsc_opspec::SdscFoldSet::new(op.iter.cores_used());
        out.push(
            crate::emit::emit_sdsc_tiled(
                &format!("matmul_o{}_n{ns}k{ks}", o.tid),
                &op,
                &folds,
                sym_id_base,
                layout,
            )
            .map_err(Error::from)?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod byte_gate {
    //! ⛔⛔⛔ THE BYTE GATE — every address below is folded from the EMITTED `OpSpec`s' own views
    //! through the emitter's own `view_stick_layout` + `DeviceExtents::of_view` + the device-walk
    //! derivation (`sl.off_view`), the same machinery `per_core_addr` folds a core's start
    //! through. Nothing here restates the copy's or the dot's addressing: a change to either
    //! builder that moves a byte moves the folded address, and the assertions compare the two
    //! folds against each other and against the stick-major law the whole-function door's minted
    //! A/OUT buffers are written in.
    use super::*;
    use crate::superdsc_opspec::{AnyTensorArg, Role, Scale, WorkPlan};

    /// A paged-attention geometry: one pool block (4096 elements) per entry, 32 entries, the
    /// g-replicated batch over them.
    fn pooled() -> BatchedGathered {
        BatchedGathered::of(&[32, 512, 64], &[32, 64, 64], 32, 4096).unwrap()
    }
    /// The g-replicated decode geometry: 32 entries, batch 32 over them, m the group.
    fn grouped() -> BatchedGathered {
        BatchedGathered::of(&[32, 4, 64], &[32, 64, 64], 32, 4096).unwrap()
    }
    /// ⭐ THE HONESTY ROW — a per-token MoE gather-dot: 64 tokens each routed to one expert,
    /// activations `[64, 8, 256]` against gathered expert weights `[64, 256, 128]`. No attention
    /// kernel states this geometry; the door and the byte gate hold at it regardless.
    fn moe_per_token() -> BatchedGathered {
        BatchedGathered::of(&[64, 8, 256], &[64, 256, 128], 64, 256 * 128).unwrap()
    }
    /// A MULTI-STICK contraction — the per-stick cut is exercised at k = 2 sticks.
    fn wide_k() -> BatchedGathered {
        BatchedGathered::of(&[32, 4, 128], &[32, 128, 64], 32, 8192).unwrap()
    }

    /// The element offset a view of `op` reads at logical index `idx` (one entry per layout dim),
    /// folded exactly as the emitter's own address model folds it — through the view's own
    /// arrangement and declared extents — plus the view's own `offset_elems`.
    fn fold(op: &OpSpec, arg: &AnyTensorArg, idx: &[usize]) -> u64 {
        let v = arg.view();
        let plan: &WorkPlan = &op.iter;
        let sl = crate::emit::view_stick_layout(&v, plan);
        let ext = crate::sdsc_abstract::DeviceExtents::of_view(
            &v.layout
                .iter()
                .map(|&d| plan.extent(d) as usize)
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
        );
        v.offset_elems as u64 + sl.off_view(&ext, idx) as u64
    }

    /// The logical index of `arg` at named coordinates — positional over its own layout.
    fn at(arg: &AnyTensorArg, coords: &[(&str, usize)]) -> Vec<usize> {
        arg.view()
            .layout
            .iter()
            .map(|d| {
                coords
                    .iter()
                    .find(|(n, _)| n == d)
                    .map(|&(_, x)| x)
                    .unwrap_or_else(|| panic!("no coordinate for layout dim '{d}'"))
            })
            .collect()
    }

    /// The stick-major flat-frame law of an `R x C` buffer — the law every whole-function
    /// intermediate is minted in.
    fn stick_major(rows: u32, cols: u32, r: u32, c: u32) -> u64 {
        crate::sdsc_abstract::StickLayout::row_blocked(rows as usize, cols as usize)
            .dev_off(r as usize, c as usize) as u64
    }

    /// Where the EMITTED copy legs wrote table row `e`'s within-entry element `j` (the gathered
    /// load's own C-order column): the copy's OUTPUT view folded at sub-row `s` of the entry and
    /// lane `j%64`, the entry located by the leg's own run base.
    fn copy_write(copies: &[OpSpec], bg: &BatchedGathered, e: u32, j: u32) -> u64 {
        let cap = CopyDims::ENTRIES_PER_OP;
        let leg = &copies[(e / cap) as usize];
        let ia = leg
            .indirect
            .as_ref()
            .expect("a copy leg declares its gather");
        let per_entry = ia.page.get();
        assert_eq!(
            per_entry * STK,
            bg.entry_elems(),
            "skip_addr is one whole table row"
        );
        let out = leg.args.last().expect("the copy has an output");
        assert!(matches!(out.view().role, Role::Output));
        let sub_row = (e % cap) * per_entry + j / STK;
        fold(
            leg,
            out,
            &at(
                out,
                &[
                    ("mb", sub_row as usize),
                    ("out", (j % STK) as usize),
                    ("y", 0),
                ],
            ),
        )
    }

    /// ⭐⭐⭐⭐⭐ THE HEADLINE: at every geometry, every (b, k, n) the EMITTED dot's KERNEL views
    /// read is the address the EMITTED copy wrote that element at — the reshape's own identity
    /// `b = entry·page + pos`, element `(pos·K + k)·N + n` of entry `entry`'s gathered row.
    #[test]
    fn the_dot_reads_every_kernel_element_where_the_copy_wrote_it() {
        for bg in [pooled(), grouped(), moe_per_token(), wide_k()] {
            let copies = copy_opspecs(
                &bg,
                "tbl",
                "ids",
                "w",
                crate::superdsc_opspec::EntryBase::ZERO,
            )
            .unwrap();
            let rows = bg.batch * bg.m;
            let dots = dot_opspecs(&bg, "a", rows, "w", "o", rows).unwrap();
            let mut checked = 0u64;
            for (i, op) in dots.iter().enumerate() {
                let (ns, ks) = (i as u32 / (bg.k / STK), i as u32 % (bg.k / STK));
                let w = &op.args[1];
                assert!(matches!(w.view().role, Role::Kernel));
                for b in 0..bg.batch {
                    let (entry, pos) = (b / bg.page, b % bg.page);
                    for kk in 0..STK {
                        for nn in 0..STK {
                            let (gk, gn) = (ks * STK + kk, ns * STK + nn);
                            let read = fold(
                                op,
                                w,
                                &at(
                                    w,
                                    &[("y", b as usize), ("in", kk as usize), ("out", nn as usize)],
                                ),
                            );
                            let j = (pos * bg.k + gk) * bg.n + gn;
                            let wrote = copy_write(&copies, &bg, entry, j);
                            assert_eq!(
                                read, wrote,
                                "{bg:?}: op {i} reads W(b={b}, k={gk}, n={gn}) at {read}, the copy \
                                 wrote it at {wrote}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
            assert_eq!(
                checked,
                (bg.batch * bg.k * bg.n) as u64,
                "every element read once"
            );
        }
    }

    /// ⭐⭐⭐ A AND OUT LAND IN THE STICK-MAJOR FLAT FRAME their minted buffers are written in:
    /// element `(b, m, c)` of a `[batch, m, C]` value is flat row `b·m + m`, column `c`.
    #[test]
    fn the_dot_reads_a_and_writes_out_in_the_stick_major_flat_frame() {
        for bg in [pooled(), grouped(), moe_per_token(), wide_k()] {
            let rows = bg.batch * bg.m;
            let dots = dot_opspecs(&bg, "a", rows, "w", "o", rows).unwrap();
            let m_probe: Vec<u32> = [0, 1, bg.m / 2, bg.m - 1].into_iter().collect();
            let b_probe: Vec<u32> = [0, 1, bg.page, bg.batch / 2, bg.batch - 1]
                .into_iter()
                .collect();
            for (i, op) in dots.iter().enumerate() {
                let (ns, ks) = (i as u32 / (bg.k / STK), i as u32 % (bg.k / STK));
                let a = &op.args[0];
                let o = op.args.last().unwrap();
                assert!(matches!(a.view().role, Role::Input));
                assert!(matches!(o.view().role, Role::Output));
                for &b in &b_probe {
                    for &m in &m_probe {
                        for lane in [0u32, 1, 31, 63] {
                            let r = b * bg.m + m;
                            let ra = fold(
                                op,
                                a,
                                &at(
                                    a,
                                    &[("y", b as usize), ("mb", m as usize), ("in", lane as usize)],
                                ),
                            );
                            assert_eq!(
                                ra,
                                stick_major(rows, bg.k, r, ks * STK + lane),
                                "{bg:?} op {i} A"
                            );
                            let ro = fold(
                                op,
                                o,
                                &at(
                                    o,
                                    &[
                                        ("y", b as usize),
                                        ("mb", m as usize),
                                        ("out", lane as usize),
                                    ],
                                ),
                            );
                            assert_eq!(
                                ro,
                                stick_major(rows, bg.n, r, ns * STK + lane),
                                "{bg:?} op {i} OUT"
                            );
                        }
                    }
                }
                // The accumulate chain: the first contraction stick seeds, every later one adds
                // the prior partial read from the SAME output block.
                match (ks, &op.op_info) {
                    (0, crate::superdsc_opspec::OpInfo::None) => {}
                    (_, crate::superdsc_opspec::OpInfo::FusedEpilogue { arg_idx, .. })
                        if ks > 0 =>
                    {
                        let e = &op.args[*arg_idx];
                        assert_eq!(e.view().name, "o");
                        assert_eq!(e.view().offset_elems, o.view().offset_elems);
                        assert!(e.view().scale.iter().all(|s| matches!(s, Scale::Active)));
                    }
                    (ks, info) => panic!("{bg:?} op {i} (ks={ks}) has op_info {info:?}"),
                }
            }
        }
    }

    /// ⛔ THE CONTROL THAT MAKES THE HEADLINE A GATE: at a MULTI-STICK geometry (the MoE honesty
    /// row), the same stick op WITHOUT the kernel's declared physical `in`/`out` extents derives
    /// its strides from the `[64, 64]` tile, reads the wrong bytes, and the fold sees it — so a
    /// green headline is evidence about the addresses, not the fold. (At a one-stick `k = n = 64`
    /// geometry the tile strides ARE the buffer's, so the control must not run there.)
    #[test]
    fn the_kernel_without_its_physical_extents_reads_the_wrong_bytes() {
        let bg = moe_per_token();
        let copies = copy_opspecs(
            &bg,
            "tbl",
            "ids",
            "w",
            crate::superdsc_opspec::EntryBase::ZERO,
        )
        .unwrap();
        // W(b=1, k=1, n=0): page = 1, so batch row 1 is entry 1, within-entry element
        // (pos·K + k)·N + n.
        let (b, kk, nn) = (1u32, 1u32, 0u32);
        let wrote = copy_write(
            &copies,
            &bg,
            b / bg.page,
            (b % bg.page * bg.k + kk) * bg.n + nn,
        );
        // The emitted (ks=0, ns=0) dot op — WITH its physical extents — reads it there.
        let dots = dot_opspecs(&bg, "a", bg.batch * bg.m, "w", "o", bg.batch * bg.m).unwrap();
        let op = &dots[0];
        let w = &op.args[1];
        let read = fold(
            op,
            w,
            &at(
                w,
                &[("y", b as usize), ("in", kk as usize), ("out", nn as usize)],
            ),
        );
        assert_eq!(read, wrote, "the emitted op agrees with the copy");
        // The control: the same stick op built WITHOUT the declared extents.
        let plain = matmul_opspec_batched_off(
            MatM::of_query_rows(QueryRowCount::of_mq(bg.m)),
            MatN::of_out_features(STK),
            MatK::of_in_features(STK),
            MatY::of_heads(bg.batch),
            "a",
            "w",
            "o",
            0,
            0,
            0,
            None,
        )
        .unwrap();
        let pw = &plain.args[1];
        let tile_read = fold(
            &plain,
            pw,
            &at(
                pw,
                &[("y", b as usize), ("in", kk as usize), ("out", nn as usize)],
            ),
        );
        assert_ne!(
            tile_read, wrote,
            "the tile-derived strides must NOT agree with the copy"
        );
    }

    /// ⭐ THE OP COUNT IS PINNED — `(K/64)·(N/64)` per dot — so a change cannot silently multiply
    /// the cut. One-stick geometries emit ONE op; `wide_k`'s two-stick contraction emits two.
    #[test]
    fn the_op_count_per_dot_is_pinned() {
        for (bg, want) in [
            (pooled(), 1usize),
            (grouped(), 1),
            (moe_per_token(), (256 / STK) as usize * (128 / STK) as usize),
            (wide_k(), 2),
        ] {
            let rows = bg.batch * bg.m;
            let dots = dot_opspecs(&bg, "a", rows, "w", "o", rows).unwrap();
            assert_eq!(dots.len(), want, "{bg:?}");
            assert_eq!(bg.dot_ops() as usize, want);
            let copies = copy_opspecs(
                &bg,
                "tbl",
                "ids",
                "w",
                crate::superdsc_opspec::EntryBase::ZERO,
            )
            .unwrap();
            assert_eq!(
                copies.len(),
                bg.entries.div_ceil(CopyDims::ENTRIES_PER_OP) as usize,
                "{bg:?}: one leg per index stick"
            );
        }
    }

    /// ⭐⭐⭐ THE WIRE ORDER — a dot operand's serialized dim fields are REVERSED (dxp reads
    /// `layoutDimOrder_` idx 0 as innermost), its `maxDimSizes_` is the reconstruct spelling, and
    /// its declared physical extents ride as back gaps — while the address model above reads the
    /// SAME view in walk order. The control is the same op WITHOUT the flag: stated order, pinned
    /// walk, no gaps — every view the crate emitted before this arm.
    #[test]
    fn the_dot_serializes_innermost_first_with_back_gaps() {
        let bg = pooled();
        let rows = bg.batch * bg.m;
        let dots = dot_opspecs(&bg, "a", rows, "w", "o", rows).unwrap();
        for op in &dots {
            for arg in &op.args {
                let v = arg.view();
                assert!(v.dxp_innermost_first, "every dot operand carries the flag");
            }
            // The KERNEL's back gaps are its two physical extents minus the tile's sweeps.
            let kernel = &op.args[1];
            let kv = kernel.view();
            let in_gap = kv.device_extent[kv.layout.iter().position(|&d| d == "in").unwrap()]
                .unwrap()
                - op.iter.extent("in");
            let out_gap = kv.device_extent[kv.layout.iter().position(|&d| d == "out").unwrap()]
                .unwrap()
                - op.iter.extent("out");
            assert_eq!(in_gap, bg.k - STK, "the in gap is phys − swept");
            assert_eq!(out_gap, bg.n - STK, "the out gap is phys − swept");
        }
        // The control: the same builder's args without the flag keep the stated order and pin
        // their walk — the byte-identical form every pre-existing view takes.
        let mut plain = matmul_opspec_batched_off(
            MatM::of_query_rows(QueryRowCount::of_mq(bg.m)),
            MatN::of_out_features(STK),
            MatK::of_in_features(STK),
            MatY::of_heads(bg.batch),
            "a",
            "w",
            "o",
            0,
            0,
            0,
            None,
        )
        .unwrap();
        for arg in plain.args.iter_mut() {
            let v = arg.view();
            assert!(!v.dxp_innermost_first, "the flag defaults to stated order");
        }
    }

    /// The door's refusals, each one variable away from a geometry that builds.
    #[test]
    fn the_door_refuses_a_frame_the_bytes_do_not_have() {
        assert!(BatchedGathered::of(&[32, 512, 64], &[32, 64, 64], 32, 4096).is_ok());
        // A rank-2 B is the rung-4 2-D path's own frame, not this one.
        assert!(BatchedGathered::of(&[32, 512, 64], &[64, 64], 32, 4096).is_err());
        // The batch the entries do not divide.
        assert!(BatchedGathered::of(&[30, 512, 64], &[30, 64, 64], 32, 4096).is_err());
        // The entry-size law: one batch row of [64, 64] is 4096 elements, not 8192.
        assert!(BatchedGathered::of(&[32, 512, 64], &[32, 64, 64], 32, 8192).is_err());
        // A contraction disagreement.
        assert!(BatchedGathered::of(&[32, 512, 128], &[32, 64, 64], 32, 4096).is_err());
        // A sub-stick output width.
        assert!(BatchedGathered::of(&[32, 512, 64], &[32, 64, 32], 32, 2048).is_err());
        // A batch disagreement between the operands.
        assert!(BatchedGathered::of(&[32, 512, 64], &[64, 64, 64], 32, 4096).is_err());
    }

    /// ⛔ THE FLAT-FRAME ROW LAW — A and OUT at any other row count than `batch·m` refuse: the
    /// per-stick column blocks are addressed at that row count, and a mismatched buffer would
    /// stride its stick groups at the wrong pitch from a clean bake.
    #[test]
    fn the_dot_refuses_mismatched_flat_frame_rows() {
        let bg = pooled();
        assert!(dot_opspecs(&bg, "a", bg.batch * bg.m, "w", "o", bg.batch * bg.m).is_ok());
        let Err(m) = dot_opspecs(&bg, "a", bg.batch * bg.m + 64, "w", "o", bg.batch * bg.m) else {
            panic!("an A wider than batch·m must refuse");
        };
        assert!(m.contains("flat frame"), "{m}");
    }
}
