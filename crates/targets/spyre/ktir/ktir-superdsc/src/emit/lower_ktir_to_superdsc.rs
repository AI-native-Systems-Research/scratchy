// SPDX-License-Identifier: Apache-2.0
//! KTIR → SuperDSC — **`ibm/main`'s `lower_*_node` bodies, with exactly one thing changed: their
//! input door.**
//!
//! ⛔⛔⛔ NOTHING HERE IS A GENERIC OP WALK, AND THAT IS THE WHOLE POINT. A walk that mapped each
//! `arith.*`/`math.*`/`linalg.*` op to an `OpFunc` lived here and was deleted: KTIR names ops at a
//! grain this device does not have descriptors for, so the walk kept asking the dxp scheduler for
//! shapes it refuses — a `linalg.transpose` of a per-step `[1, 64]` key became a restickify whose
//! stick extent is 1, which is not a whole multiple of the 64-element SEN169_FP16 stick. main's
//! attention never transposes a `[1, 64]` key: its Kᵀ is a whole-page re-transpose. The mapping
//! authority is main's bodies, not a per-op table.
//!
//! ⛔ AND NOTHING HERE SERIALIZES ANYTHING. These bodies build SuperDSC as VALUES through the SAME
//! proven builders the SubtileIR path used — `pw1`/`pw2`/`assemble_matmul`/`assemble_rmsnorm`/
//! `assemble_attn` — and everything after that is main's mechanism, unchanged: `render_dxp_input`
//! renders, `launch_index` partitions, and the `superdsc_bake` queue stages, seals and compiles.
//!
//! ⭐ THE DOOR. Each body took a `&SubtileNode` and read its output tensor, its input regions and its
//! op attributes off it. It now takes the node's KTIR program, where all of that is already carried:
//!
//!   * `KtirNode::args[i]` is the SubtileIR tensor the `i`-th parameter points at — that field's own
//!     doc says "the pairing is CARRIED, not re-derived" — and `crate::place::act_name(tid)` is the
//!     operand-name spelling every proven builder and `BundleLayout` agree on, i.e. exactly what
//!     `ds_name` gave the SubtileIR path;
//!   * a parameter's `ktdp.construct_memory_view` states the buffer's `[rows, cols]` and its
//!     `ktdp.construct_access_tile` states the window taken of it — together, the `TensorRegion`;
//!   * a model constant (a ScalarMul scale, an rmsnorm eps, the attention multiplier) arrives as a
//!     parameter at its RESERVED tid, so its registry index is recovered by inverting
//!     `scalarmul_scale_tid` rather than by a value lookup;
//!   * the node KIND is the program's own name, written by the producer (`matmul_s2`, `rmsnorm_s1`,
//!     `attn_s7`, …). Reading that name is not recognition.
//!
//! ⭐ [`super::ktir_matmul_fp8`] IS THIS PATTERN ALREADY DONE — main's `lower_matmul_node` arity-3
//! branch, unchanged, reached from facts the KTIR states — and the fp8 arm below calls it.

use super::{
    EmittedOp, In, PointwiseFacts, assemble_convert, assemble_pointwise_broadcast_off,
    assemble_restickify_kt_2d, bmm_site, check_pointwise_cols, emit_sdsc_tiled, fl,
    op_func_from_str, pointwise_broadcast_opspec_df, pointwise_chunk_out_offset, pw1, pw2, rb,
    rb_df, rbo,
};
use crate::ir::bridge::tiled_op_sdsc_op::{
    AttnFacts, assemble_attn, assemble_matmul_off, assemble_matmul_seeded, assemble_matmul_windowed,
};
use crate::ir::bridge::tiled_op_sdsc_op::{
    RmsnormFacts, RmsnormUnitFacts, TanhsoftcapFacts, assemble_pointwise_broadcast_off_from_tile,
    assemble_pointwise_seeded_from_tile, assemble_rmsnorm, assemble_rmsnorm_unit,
    assemble_tanhsoftcap,
};
use crate::ir::island::tile_op::{TileOp, TileOpKind};
use crate::ktir_node::{Elementwise, KtirNode};
use crate::place::{PlaceId, SynthRole};
use crate::placement::{BundleLayout, syn};
use crate::reserved_tids::{
    ATTN_CAUSAL_TID, ATTN_MASK_TID, ATTN_ZERO_TID, KctResident, LAST_HIDDEN_TID,
    scalarmul_scale_tid,
};
use crate::score_scale::{NewK, ScoreScaleLaw};
use crate::sdsc_abstract::{BlockCols, KernelTag, Lanes, RowCount, Stk};
use crate::sdsc_abstract::{MatK, MatM, MatN};
use crate::superdsc_opspec::{DataFormat, Df, Fp16, ItDim, SdscFoldSet};
use crate::work::{CoreSplit, DeviceWidth, EwDivision, FP16_ELEMS_PER_STICK, UTIL_FLOOR_CORES};
use ktir_core::affine::AffineExpr;
use ktir_core::attrkey::AttrKey;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::opkind::OpKind;

#[derive(Debug)]
pub struct Error {
    pub message: String,
}

pub fn err<T>(message: String) -> Result<T, Error> {
    Err(Error { message })
}

impl From<crate::superdsc_error::SuperDscError> for Error {
    fn from(e: crate::superdsc_error::SuperDscError) -> Self {
        Error { message: e.0 }
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE DOOR — a `TensorRegion`, recovered from the program that states it.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The SubtileIR `TensorRegion` one KTIR parameter carries: the tensor it points at, the whole
/// buffer's `[rows, cols]`, and the window this program takes of it.
///
/// ⛔ EVERY FIELD IS READ OUT OF THE PROGRAM, none is threaded in beside it. `tid` comes from
/// [`KtirNode::args`], the buffer extent from the parameter's `ktdp.construct_memory_view` `Shape`,
/// and the window from the `ktdp.construct_access_tile` built over that view — its `Shape` is the
/// extent and its two index operands are the corner, which is where a column-chunked node's
/// `region.cols.start` lives.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    /// The SubtileIR tensor id — `crate::place::act_name(tid)` is its operand name.
    pub tid: u32,
    /// The buffer's own declared `[rows, cols]` extent — the parameter's
    /// `ktdp.construct_memory_view` `Shape`.
    ///
    /// ⭐ NOT ALWAYS THE TENSOR'S SHAPE, AND THAT IS THE POINT. The rope program views `x` as
    /// `[rows·heads, hd]`, so this states the HEAD DIM directly.
    ///
    /// ⛔ `v_rows` USED TO BE DROPPED, AND THAT COST THE FIRST GENERATED TOKEN. The prefill lm-head
    /// tail's extraction addresses `hidden[mq, hidden]` as the tall-narrow `[(hidden/64)·mq, 64]` it
    /// physically is, and the row it copies is `j·mq + row` — so `mq` is a term of the ADDRESS, not
    /// decoration. Recovering it as `r_start + 1` from the SSOT identity `selector_lastrow_col(m) =
    /// m - 1` would be an inference; the view states it, so it is read.
    pub v_rows: u32,
    pub v_cols: u32,
    /// The window the program takes of the buffer (the access tile's corner + `Shape`). Equal to the
    /// whole view when the program reads all of it.
    ///
    /// ⛔⛔⛔ `r_start` USED TO BE DISCARDED INTO `_r_start`, AND THAT WAS THE DEFECT. The mq>1
    /// prefill lm-head tail states its row in the KTIR — the matmul puts
    /// `selector_lastrow_col(mq)` on its activation access tile as an `arith.constant` — and this
    /// walk threw it away, so `matmul` built `rb(&a.name(), m, k)` at the BUFFER BASE and the
    /// vocab-wide tail read the wrong `hidden` bytes. MEASURED, granite-3.1-2b fp8, prompt `hi`
    /// (14 tokens on the m=15 rung): first generated token `yun` where main gives `Hello`, then a
    /// perfectly fluent continuation — and still wrong on `hi there`, which fills the rung EXACTLY
    /// (`m_used=15 prefill_m=15`), so it was never the zero-padded row it looked like.
    ///
    /// Every emitter here addresses the buffer BASE (`rb(name, rows, cols)` takes no offset — which
    /// is precisely why main MATERIALIZES the row, see [`lmlast`]). So a nonzero `r_start` reaching
    /// one is a fact the descriptor cannot carry, and the emitters below REFUSE it by name rather
    /// than silently addressing row 0.
    pub r_start: u32,
    pub c_start: u32,
    pub r_len: u32,
    pub c_len: u32,
    /// The row span EVERY access tile over this view covers, as `(first row, one past the last)` —
    /// `(0, r_len)` for a program that takes one window, `(0, m)` for one that ROW-BLOCKS its region
    /// into `ceil(m / blk)` windows. See [`node_rows`], which is the only reader: a body whose
    /// descriptor spans the whole node needs to know the blocks tile the node, and `r_len` above is
    /// block 0's height alone.
    pub r_cover: (u32, u32),
    /// The COLUMN span every access tile over this view covers — `r_cover`'s twin. The reshape door
    /// is the reader: a re-laying copy is stated as one `[1, 64]` tile per stick (the `lmlast`
    /// mechanism, which the splice's kernel mirrors), so its FIRST tile is a stick and `c_len`
    /// understates the buffer by the stick count — the COVERS are the form-independent statement of
    /// "the program takes the whole view on this axis", and a chunked window (`(8192, 12800)` of a
    /// 12800-wide view) answers differently than a whole one.
    pub c_cover: (u32, u32),
    /// This parameter is the program's OUTPUT — a `ktdp.store` writes through its view.
    ///
    /// ⛔ NOT "THE LAST PARAMETER". Parameters are minted in FIRST-USE order, and a model constant
    /// first used after the output's view (a matmul's zero seed, read by `splat_zero` only once the
    /// contraction is built) lands after it. The store is what says which buffer is written.
    pub is_out: bool,
    /// This parameter's view declares `Fp8E4m3` elements — one byte each.
    ///
    /// ⭐ THE fp8-NESS IS A VIEW FACT, not an arity guess: the fp8 matmul builds its weight
    /// through `view_fp8`, which writes `Dtype = Fp8E4m3` on the `ktdp.construct_memory_view`.
    pub is_fp8: bool,
    /// ⭐⭐⭐⭐⭐ WALL 13 — THE STICK BASIS IS THE DATA'S OWN `Df`. A physical stick is 128 bytes:
    /// 64 `Fp16` lanes, 32 `Fp32` lanes, 128 `Fp8` lanes (`DataFormat::ELEMS_PER_STICK`). The
    /// nonstick arm's sub-stick question — "are the survivors a whole stick?" — is meaningless
    /// until the basis is named, and the only honest basis is the one the region's own data
    /// states: the online-softmax `m`/`l` merges carry 32 survivors, HALF an fp16 stick but
    /// EXACTLY one whole fp32 stick (`summeanmaxexx2_fp32.ddl` binds the full nonstick family at
    /// IEEE_FP32), and the f32-ness is the kernel's own statement (`m` is `tl.float32`, the f16
    /// cast is after the merge). Until the walk threads each region's true element type into
    /// this field (wall 13's island commit), every region mints `Df::Fp16` — the classification
    /// of every f16 region is byte-identical to the pre-wall-13 basis hardcode.
    pub df: Df,
}

impl Region {
    /// The operand name — the spelling `ds_name` produced for the SubtileIR path.
    pub fn name(&self) -> String {
        crate::place::act_name(self.tid)
    }
}

/// ⭐⭐⭐⭐⭐ WHAT A GATHERED READ READS — the value side of an indirect access tile, classified
/// ONCE at the only door that mints a [`Gather`].
///
/// ⛔⛔⛔ THE DEFECT THIS TYPE EXISTS TO MAKE UNWRITABLE: the reserved Kᵀ plane's name and its
/// K-cache SOURCE tensor were both bare `u32`s, so the score gather's value side could be
/// resolved through the source tensor's own placement and bake over the NATURAL-K plane —
/// same size, wrong plane, invisible to every extent law. Classified here, a binding in the
/// `kct_resident` region is a [`GatherSource::KtPlane`] and resolves through the reserved
/// tid's own placement (the pool's Kᵀ plane law). There is NO arm that yields the K-cache
/// source: [`KctResident::source_k`] is a SCAN fact (the wiring completeness marker), never a
/// placement fact, and this enum does not carry it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherSource {
    /// A real graph tensor, resolved by its own placement.
    Tensor(u32),
    /// The reserved Kᵀ plane — the score gather's ONLY legal K side.
    KtPlane(crate::reserved_tids::KctResident),
}

impl GatherSource {
    /// The tid this source NAMES — the operand name and the placement key. For a Kᵀ plane this
    /// is the reserved tid, so the descriptor gathers from the placement the pool law placed,
    /// never from the source tensor's.
    pub fn tid(&self) -> u32 {
        match self {
            GatherSource::Tensor(tid) => *tid,
            GatherSource::KtPlane(kct) => kct.wire_tid(),
        }
    }

    /// ⭐ THE ONE CLASSIFICATION DOOR — a bound value tid becomes a source HERE or nowhere. A
    /// tid in the `kct_resident` region is a Kᵀ plane binding; every other tid is a tensor.
    /// `pub` so the splice's mint and the placement's sizing can classify the same way, and
    /// `const`-shaped so nothing downstream "re-derives" it.
    pub fn of_bound(tid: u32) -> Self {
        match crate::reserved_tids::KctResident::of_wire(tid) {
            Some(kct) => GatherSource::KtPlane(kct),
            None => GatherSource::Tensor(tid),
        }
    }
}

/// ⭐⭐⭐ A GATHER THE PROGRAM STATES — the index parameter, the gathered parameter, and how many
/// entries are taken.
///
/// `ktdp.construct_indirect_access_tile` is the ONE op that says "the row index is data": its first
/// operand is the DATA view and its second the INDEX view, and each view's own first operand is the
/// function parameter it reinterprets. So all three facts are read off the program; none is threaded in.
#[derive(Clone, Copy, Debug)]
pub struct Gather {
    /// The parameter holding the index vector — `crate::place::act_name(index_tid)` is its operand name.
    pub index_tid: u32,
    /// ⭐ WHAT the gather reads, CLASSIFIED — never a bare tid. The score gather's K side is a
    /// [`GatherSource::KtPlane`] and resolves through the reserved placement; a bare source tid
    /// here is the wrong-plane defect this field's type exists to refuse.
    pub source: GatherSource,
    /// Entries the gather takes: the gathered tile's SLOW extent, one index per gathered row.
    pub entries: u32,
    /// ⭐ WHERE THIS GATHER'S RUN OF ENTRIES STARTS IN THE INDEX BUFFER — the indirect dim's own
    /// subscript offset, which an unrolled sweep states per trip. The tile's `dim_subs[0]` is
    /// `Add(Sym(0), Dim(0))`: enumeration row `Dim(0)` reads index word `Sym(0) + row`, and
    /// `Sym(0)` is `intermediate_vars[0]`'s constant. A one-block program states 0; an unrolled
    /// KV sweep's trip `b` states `b · BLOCK_N`, and dropping it gathers every trip's rows through
    /// block 0's ids — the right shape, the right dtype, the wrong rows, from a clean bake. The
    /// value is in ENTRIES (one `SenUint32` word per entry), which is what
    /// [`crate::superdsc_opspec::EntryBase`] reports and what the index operand's
    /// `with_offset` takes. An [`EntryBase`], so it is stick-aligned by construction (issue 201
    /// item 7) — `gathers_of` builds it through `of_entries` and an unaligned anchor refuses at
    /// the walk, before any descriptor exists.
    pub first_entry: crate::superdsc_opspec::EntryBase,
    /// ⭐ THE TILE'S OWN RESULT — the `ktdp.construct_indirect_access_tile` value this gather was
    /// read from. Carried so every joiner (the whole-function walk, the matmul materializer) reads
    /// THIS gather's tile by identity instead of re-filtering `KtdpConstructIndirectAccessTile` and
    /// matching by enumeration index — three walks that must agree by position are three walks that
    /// can silently disagree (issue 201 item 3).
    pub tile: Ssa,
    /// ⭐ THE GATHERED LOAD — the `ktdp.load` over [`Gather::tile`] whose result carries the
    /// gathered rows. This is the identity an op's input must BE for this gather to be the one it
    /// reads: a TID comparison cannot prove it (a plain load of the same table resolves to the
    /// same tid), the load's own SSA can.
    pub load: Ssa,
}

/// [`Gather`] for this program, or `None` when it states no indirect access.
///
/// ⛔ MORE THAN ONE IS REFUSED BY NAME. `computeOp_.indirectAccessIndexLabeledDs` is a list, but no
/// shipped vendor input has more than one entry and nothing states how a second index pairs with its
/// operand — and the pairing this crate emits IS the operand adjacency ([`Role::KernelIdx`]), which two
/// indices cannot both have. So a two-gather program is a refusal, not a guess.
///
/// ⛔ AND THE INDEX MAY NOT BE A VIEW OF THE TENSOR BEING GATHERED. An indirect tile whose indices come
/// out of the buffer it indexes is a self-reference the alloc pair cannot express (one alloc node cannot
/// be both `index_tensor` and `value_tensor`), and it is also certainly a producer bug.
pub fn gather_of(k: &KtirNode) -> Result<Option<Gather>, Error> {
    let mut all = gathers_of(k)?;
    match all.len() {
        0 => Ok(None),
        1 => Ok(Some(all.remove(0))),
        n => err(format!(
            "{}: {n} `ktdp.construct_indirect_access_tile` op(s). One descriptor carries ONE index \
             operand, paired with the tensor it gathers by POSITION (the index sits immediately \
             after it, which is what `DSC2ToDataflowIR.cpp:51` requires), and two indices cannot \
             both be adjacent to their own operand. Split the program into one node per gather.",
            k.func.name,
        )),
    }
}

/// ⭐⭐⭐ EVERY `Gather` THE PROGRAM STATES, ONE PER INDIRECT TILE — the multi-gather reading.
///
/// The vendor's contract carries ONE index operand per descriptor (`DSC2ToDataflowIR.cpp:51` pairs
/// an index with the tensor it gathers BY POSITION), which is why [`gather_of`] refuses two tiles
/// in one program: one descriptor cannot carry two indices. But the Triton front end's trip unroll
/// (`unroll_constant_trip_loops`) SPLICES a time-tiled KV sweep into straight-line trips, each trip
/// carrying its own `ktdp.construct_indirect_access_tile` over the SAME index and table parameters
/// — the paged multi-block shape, where every trip gathers a different block's rows through the
/// same two buffers. That is not two descriptors on one op; it is N ops, each with its own index
/// operand, which the contract admits and the one-gather reading refuses only because it counts
/// tiles instead of ops.
///
/// ⭐⭐ AND THE PAIRS MAY DIFFER — the batched paged-decode shape. That kernel gathers TWO tables
/// through ONE ids buffer: K^T whole pages and V whole pages, each feeding its own contraction,
/// so its tiles span two (index, value) pairs in one function. This is safe for the same reason
/// the unrolled sweep is: the one-index law is a PER-DESCRIPTOR fact, and the whole-function walk
/// joins EACH op to its own tile by the identity of the gathered LOAD (`op_gather`), appending
/// that tile's index region to THAT op's operand list — two ops gathering different tables are
/// two descriptors, each carrying its own single index operand. The shapes that would violate the
/// per-descriptor law are refused downstream, by name: an op reading a gathered value reaches the
/// walk's ScalarMul-or-materialized rule, the materializer demands the gather be the matmul's B
/// with exactly one consumer, and a tile no descriptor declares trips the uncarried check. None
/// of those guards looks at a SECOND tile, so none is weakened by admitting one.
///
/// What the multi-pair reading DOES add is a fact the placement owns: the set of index buffers.
/// A program may now state SEVERAL distinct index tids, and the layout sizes every one of them
/// `SenUint32` (a 4 B/elem index is per-INDEX-BUFFER, not per-program), dtype-checks each, and
/// gives each its flush-at-end fetch slack.
///
/// A gather the walk below does not carry is still the silent-direct-read hazard `gather_of`'s
/// comment names, so callers refuse an uncarried one exactly as before.
pub fn gathers_of(k: &KtirNode) -> Result<Vec<Gather>, Error> {
    let f = &k.func;
    let consts = index_constants(&k.func);
    let tiles: Vec<&Operation<'_>> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpConstructIndirectAccessTile)
        .collect();
    let mut out: Vec<Gather> = Vec::with_capacity(tiles.len());
    for tile in &tiles {
        // ⭐ A TILE A `ktdp.store` WRITES THROUGH (and no load reads) IS A SCATTER — [`scatters_of`]
        // reads it, with the same tile facts.
        if tile_is_scattered(f, tile) {
            continue;
        }
        let (value_tid, index_tid, entries, first_entry) = indirect_tile_facts(k, &consts, tile)?;
        // ⭐ NO CROSS-TILE PAIR CHECK — deliberately. The one-index law is per DESCRIPTOR, and the
        // walk below joins each op to its own tile by load identity, so two tiles over different
        // pairs are two descriptors each carrying one index operand.
        // ⭐ THIS TILE'S GATHERED LOAD — the `ktdp.load` over the tile, found HERE so the value
        // travels with the gather. Exactly one load per tile: two loads over one tile would be two
        // gathers on one descriptor shape.
        let Some(tv) = tile.result else {
            return err(format!(
                "{}: an indirect access tile states no result, so no `ktdp.load` can read it and \
                 the gather it describes never happens",
                f.name,
            ));
        };
        let loads: Vec<&Operation<'_>> = f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::KtdpLoad && o.operands.first() == Some(&tv))
            .collect();
        let [load] = loads[..] else {
            return err(format!(
                "{}: {} `ktdp.load` op(s) over one indirect access tile — one tile is ONE gather \
                 (one index operand per descriptor, DSC2ToDataflowIR.cpp:51), and its rows reach \
                 consumers through one value",
                f.name,
                loads.len(),
            ));
        };
        let Some(load) = load.result else {
            return err(format!(
                "{}: the indirect access tile's `ktdp.load` states no result, so the gathered rows \
                 have no identity to join by",
                f.name,
            ));
        };
        out.push(Gather {
            index_tid,
            // ⭐ THE ONE CLASSIFICATION — the value side is a source, classified here at the only
            // mint, so a `kct_resident` binding can never arrive downstream as a bare source tid.
            source: GatherSource::of_bound(value_tid),
            entries,
            first_entry,
            tile: tv,
            load,
        });
    }
    Ok(out)
}

/// Whether a `ktdp.store` writes THROUGH this indirect tile and no `ktdp.load` reads it — the
/// scatter form ([`scatters_of`]).
fn tile_is_scattered(f: &IRFunction<'static>, tile: &Operation<'_>) -> bool {
    let Some(tv) = tile.result else { return false };
    let reads = f
        .operations
        .iter()
        .any(|o| o.op_type == OpKind::KtdpLoad && o.operands.first() == Some(&tv));
    let writes = f
        .operations
        .iter()
        .any(|o| o.op_type == OpKind::KtdpStore && o.operands.get(1) == Some(&tv));
    writes && !reads
}

/// ⭐⭐⭐ ONE INDEXED WRITE THE PROGRAM STATES — `tt.descriptor_scatter`, lowered by the front end to
/// a `ktdp.store` of `value` through a `ktdp.construct_indirect_access_tile`. The store twin of
/// [`Gather`]: the same tile facts ([`indirect_tile_facts`]), and the stored VALUE instead of the
/// gathered load. It lowers to the vendor's `test_scatter_1core` shape: an `identity` copy whose
/// OUTPUT is the indexed table.
#[derive(Clone, Copy, Debug)]
pub struct Scatter {
    /// The parameter holding the index vector.
    pub index_tid: u32,
    /// The parameter written THROUGH the index (the table).
    pub value_tid: u32,
    /// Rows written: the tile's slow extent, one index entry per row.
    pub entries: u32,
    /// Where this scatter's run of entries starts in the index buffer.
    pub first_entry: crate::superdsc_opspec::EntryBase,
    /// The tile's own result.
    pub tile: Ssa,
    /// The value the `ktdp.store` writes — the rows that land in the table.
    pub value: Ssa,
}

/// Every [`Scatter`] the program states, one per indirect tile a `ktdp.store` writes through.
///
/// ⛔ EXACTLY ONE STORE PER TILE: a tile is one index operand on one descriptor
/// (`DSC2ToDataflowIR.cpp:51`), so two stores through one tile would be two writes claiming one
/// index pairing.
pub fn scatters_of(k: &KtirNode) -> Result<Vec<Scatter>, Error> {
    let f = &k.func;
    let consts = index_constants(&k.func);
    let mut out = Vec::new();
    for tile in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpConstructIndirectAccessTile)
    {
        if !tile_is_scattered(f, tile) {
            continue;
        }
        let Some(tv) = tile.result else { continue };
        let stores: Vec<&Operation<'_>> = f
            .operations
            .iter()
            .filter(|o| o.op_type == OpKind::KtdpStore && o.operands.get(1) == Some(&tv))
            .collect();
        let [store] = stores[..] else {
            return err(format!(
                "{}: {} `ktdp.store` op(s) through one indirect access tile — one tile is ONE \
                 indexed write (one index operand per descriptor, DSC2ToDataflowIR.cpp:51)",
                f.name,
                stores.len(),
            ));
        };
        let Some(&value) = store.operands.first() else {
            return err(format!(
                "{}: a `ktdp.store` through an indirect tile states no value",
                f.name
            ));
        };
        let (value_tid, index_tid, entries, first_entry) = indirect_tile_facts(k, &consts, tile)?;
        out.push(Scatter {
            index_tid,
            value_tid,
            entries,
            first_entry,
            tile: tv,
            value,
        });
    }
    Ok(out)
}

/// The four facts an indirect tile states — the table it indexes, the index buffer, the entries it
/// takes and where its run starts in the index — read ONE way for both a gather and a scatter, so
/// the two cannot disagree about which parameter is the table.
fn indirect_tile_facts(
    k: &KtirNode,
    consts: &std::collections::HashMap<Ssa, i64>,
    tile: &Operation<'_>,
) -> Result<(u32, u32, u32, crate::superdsc_opspec::EntryBase), Error> {
    let f = &k.func;
    {
        // The SAME param walk `gather_of` runs, per tile — a closure over this tile's operands so
        // the two readings cannot disagree about which parameter a tile's view names.
        let param_tid = |slot: usize, what: &str| -> Result<u32, Error> {
            let view_v = tile.operands.get(slot).copied().ok_or_else(|| Error {
                message: format!(
                    "{}: the indirect access tile states no {what} operand — a gather needs both the \
                     tensor it reads and the index vector that chooses the rows",
                    f.name
                ),
            })?;
            let view = f
                .operations
                .iter()
                .find(|o| o.result == Some(view_v) && o.op_type == OpKind::KtdpConstructMemoryView)
                .ok_or_else(|| Error {
                    message: format!(
                        "{}: the indirect access tile's {what} operand is not a \
                         `ktdp.construct_memory_view`, so no parameter can be named for it",
                        f.name
                    ),
                })?;
            let ptr = view.operands.first().copied();
            let i = f
                .arguments
                .iter()
                .position(|(a, _)| Some(*a) == ptr)
                .ok_or_else(|| Error {
                    message: format!(
                        "{}: the indirect access tile's {what} view does not reinterpret a PARAMETER, \
                         so the buffer it names has no binding and no placement",
                        f.name
                    ),
                })?;
            k.bindings.get(i).map(|b| b.get()).ok_or_else(|| Error {
                message: format!("{}: parameter {i} has no bound buffer", f.name),
            })
        };
        let value_tid = param_tid(0, "gathered")?;
        let index_tid = param_tid(1, "index")?;
        if value_tid == index_tid {
            return err(format!(
                "{}: the gather's index view and the tensor it gathers are the SAME parameter \
                 (t{value_tid}) — one HBM allocation cannot be both this gather's `index_tensor` and \
                 its `value_tensor`, which is the bidirectional `relatedIndirectAccessAlloc_` pair dbo \
                 follows in both directions.",
                f.name
            ));
        }
        let (entries, _) = shape_2d(tile).ok_or_else(|| Error {
            message: format!(
                "{}: the indirect access tile states no 2-D `shape`, so the number of gathered \
                 entries is unknown",
                f.name
            ),
        })?;
        // ⭐⭐⭐ THE TRIP'S ANCHOR — where this tile's run of entries starts in the index buffer.
        // `dim_kinds[0]` is `"indirect"` and its `dim_subs[0]` is `Add(Sym(0), Dim(0))`: enumeration
        // row `Dim(0)` reads index word `Sym(0) + row`, with `Sym(0)` named by `intermediate_vars[0]`.
        // A one-block program states 0 (folded away or cast, both of which `index_constants`
        // resolves); an unrolled sweep's trip `b` states `b · BLOCK_N`, and this walk refuses a
        // tile whose anchor is NOT a constant by name rather than gathering from index word 0 —
        // the silent wrong-row shape `EntryBase` exists to make unrepresentable at the emitter.
        let first_entry = {
            let kinds = tile.attr(AttrKey::DimKinds);
            let vars = tile.attr(AttrKey::IntermediateVars);
            let subs = tile.attr(AttrKey::DimSubs);
            let (
                Some(Attr::StrList(kinds)),
                Some(Attr::Ssas(vars)),
                Some(Attr::AffineMapList(subs)),
            ) = (kinds, vars, subs)
            else {
                return err(format!(
                    "{}: an indirect access tile states no `dim_kinds`/`intermediate_vars`/`dim_subs` \
                     triple, so where its run of entries starts in the index buffer is unknown — the \
                     gather would read index word 0 by default",
                    f.name
                ));
            };
            let Some(d) = kinds.iter().position(|k| *k == "indirect") else {
                return err(format!(
                    "{}: an indirect access tile states no `indirect` dim, so no subscript names the \
                     index buffer it reads",
                    f.name
                ));
            };
            let Some(sub) = subs.get(d) else {
                return err(format!(
                    "{}: the indirect access tile's dim {d} carries no `dim_subs` map",
                    f.name
                ));
            };
            // ⛔ THE SUBSCRIPT MUST BE `Sym(0) + Dim(0)` AND NOTHING ELSE — the one form
            // `gather_subscripts` builds and the one this reading can fold into an entry base. A
            // tile with a scaled or second-symbol subscript names a different addressing program,
            // and guessing its constant is an address; refusing names the map.
            let anchor = match sub.exprs.first() {
                Some(AffineExpr::Add(a, b)) => match (a, b) {
                    (AffineExpr::Sym(i), AffineExpr::Dim(j)) if *i < vars.len() && *j == 0 => {
                        consts.get(&vars[*i]).copied()
                    }
                    (AffineExpr::Dim(j), AffineExpr::Sym(i)) if *i < vars.len() && *j == 0 => {
                        consts.get(&vars[*i]).copied()
                    }
                    _ => None,
                },
                Some(AffineExpr::Dim(j)) if *j == 0 => Some(0),
                _ => None,
            };
            let Some(anchor) = anchor else {
                return err(format!(
                    "{}: the indirect access tile's subscript for dim {d} is not the form this \
                     reader folds (`Sym(0) + Dim(0)` with `Sym(0)` a constant, or a bare `Dim(0)`), so \
                     where this gather's entries start in the index buffer is unknown — the gather \
                     would read index word 0 by default. State the trip's anchor as a constant.",
                    f.name
                ));
            };
            if anchor < 0 || anchor > u32::MAX as i64 {
                return err(format!(
                    "{}: the indirect access tile's index anchor is {anchor}, which is not a u32 \
                     entry count",
                    f.name
                ));
            }
            // The constructor IS the stick-alignment check (issue 201 item 1): an anchor that is
            // not a whole `SenUint32` stick into the index buffer refuses HERE, at the walk,
            // before any descriptor exists — on both the uncut and the cut arm.
            crate::superdsc_opspec::EntryBase::of_entries(anchor as u32).map_err(|msg| Error {
                message: format!("{}: {msg}", f.name),
            })?
        };
        Ok((value_tid, index_tid, entries, first_entry))
    }
}

/// ⛔⛔⛔ THE GUARD FOR THE FACT THIS FILE USED TO DROP: an operand this emitter addresses at the
/// BUFFER BASE may not carry a row corner.
///
/// `rb(name, rows, cols)` — every operand spelling below — names a tensor, not an offset into one, so
/// the only row a base-addressed descriptor can read is row 0. A KTIR program CAN state a row
/// (the matmul puts `selector_lastrow_col(mq)` on its activation access tile), and for the
/// two years' worth of programs here the corner is 0 — except the prefill lm-head tail, whose corner
/// this walk discarded, producing a wrong first generated token with a fluent continuation behind it.
///
/// So the row corner is now a fact, and a fact the descriptor cannot carry is a BUILD ERROR naming
/// the op — not row 0 by default. The one emitter that CAN place a row is [`lmlast`], which does it
/// main's way: `hidden/64` single-stick copies into a `[1, hidden]` synthetic, addressed on the
/// tall-narrow reinterpretation where a row offset IS representable.
fn base_addressed(name: &str, r: &Region, role: &str) -> Result<(), Error> {
    if r.r_start != 0 {
        return err(format!(
            "{name}: {role} t{} states row corner {} of its `[{}, {}]` view, and this operand is \
             addressed at the BUFFER BASE (`rb` names a tensor, not an offset into one) — so the \
             descriptor would read row 0 and the program would compute a different row's answer. \
             A row offset is representable only on the tall-narrow `[(cols/64)·rows, 64]` \
             reinterpretation; materialize the row first, the way `lmlast` does.",
            r.tid, r.r_start, r.v_rows, r.v_cols,
        ));
    }
    Ok(())
}

/// ⛔⛔⛔ THE NODE'S ROW COUNT, WHICH IS NOT ITS FIRST WINDOW'S HEIGHT.
///
/// main's pointwise bodies (`lower_elementwise_node`, `lower_silumul_node`, `lower_rmsnorm_node`,
/// `lower_scalarmul_node`) were each called ONCE per node with `node.output.region.rows.len` and emit
/// descriptors spanning all of it. The KTIR side states that same number on
/// [`KtirNode::out_shape`] — and it has to, because a producer arm may ROW-BLOCK its region so the
/// emulator's per-core LX holds the live tile set (silu_mul always does; elementwise and
/// scalarmul do above `EW_LX_ELEMS`), which makes the program `ceil(m / blk)` windows rather than one.
/// Reading `Region::r_len` there yields the BLOCK HEIGHT, and a descriptor cut to the block height
/// leaves every later block's rows unwritten on the card — MEASURED as `mb_=16` beside `mb_=31` in
/// granite-3.1-2b's prefill layer, which is the whole ktir→superdsc prefill divergence above rung 16.
///
/// So the row count comes from the node, and the blocks are checked to TILE the node: they must cover
/// `0..out_shape.0` exactly, since the emitted descriptor computes all of it. A program whose windows
/// do not is a build error naming it, never a descriptor that spans rows the program never states.
/// ⛔ IT WAS `KtirNode::out_shape.0`, CHECKED AGAINST THIS. The check forced the two equal, so the
/// declared row count was never independent information — it was a copy of what the store windows
/// already say, travelling `SubtileIR → record → SuperDSC` past the IR. The windows are the source
/// now, and the guard stays inside the program: they must start at row 0 and reach exactly the output
/// view's own row extent, so a descriptor still cannot span rows the program never states.
fn node_rows(name: &str, out: &Region) -> Result<u32, Error> {
    if out.r_cover != (0, out.v_rows) {
        return err(format!(
            "{name}: the program's store windows over its output t{} cover rows {}..{} while that \
             output's own view states {} row(s) — a descriptor spanning the view would compute rows \
             the program never stores. Its windows must tile `0..{}`.",
            out.tid, out.r_cover.0, out.r_cover.1, out.v_rows, out.v_rows,
        ));
    }
    Ok(out.v_rows)
}

/// ⭐⭐⭐ THE ROW BASIS AN ELEMENTWISE OP SWEEPS — the matmul windowing arm's law, stated once for the
/// pointwise arms.
///
/// [`node_rows`] folds a ROW-BLOCKED program: its store windows tile `0..v_rows` and one descriptor
/// spans all of them, which is the only shape a whole-view sweep can serve. A GRID program states the
/// opposite — ONE window, at a corner this program instance owns — and its inputs are INTERMEDIATES
/// at the window's own height (an unrolled flash block's `p·V` is `[64, 128]` against a `[1024, 128]`
/// output view). Sweeping the view there would read 960 rows nobody wrote and write 960 rows nobody
/// stores; sweeping the WINDOW is exactly what `matmul_oriented` already does (`m = out.r_len`).
///
/// The corner itself is still [`base_addressed`]'s business — `split_out` refuses a nonzero row
/// corner on every operand, so a windowed op here is always at row 0 of its view (the grid's block 0
/// or an unrolled trip whose corner the fold pass carried away). A program whose window sits at a
/// nonzero row is the 16-tile grid-corner case, and it is refused by that guard rather than here.
fn pointwise_rows(name: &str, out: &Region) -> Result<u32, Error> {
    if out.r_cover == (0, out.v_rows) {
        node_rows(name, out)
    } else {
        // The window basis is only honest if the op's rows ARE the window's: the caller's extent
        // checks compare every operand against this number, and `r_len` is the window one tile
        // states. A cover that is neither the whole view nor a single window (two disjoint
        // windows, say) is a program shape this basis does not describe, and is refused by name
        // rather than folded.
        if out.r_cover.1 - out.r_cover.0 != out.r_len {
            return err(format!(
                "{name}: the output t{}'s store windows cover rows {}..{} in blocks of {} row(s) — \
                 neither a row-blocked program one descriptor can fold (windows tiling the view) nor \
                 a grid program's single window. The pointwise arms sweep one rectangle; this \
                 program states several disjoint ones.",
                out.tid, out.r_cover.0, out.r_cover.1, out.r_len,
            ));
        }
        Ok(out.r_len)
    }
}

/// This program's parameters as [`Region`]s, in parameter order.
///
/// ⭐ THE PAIRING IS A ZIP, NOT A SEARCH. A func's `arguments` are `%0..%{n-1}` in first-use order
/// and `KtirNode::args[i]` is the tensor the `i`-th points at, so the join is by position; what this
/// walk adds is each parameter's declared extent, found by following the ONE
/// `ktdp.construct_memory_view` whose address operand is that parameter.
pub fn regions(k: &KtirNode) -> Result<Vec<Region>, Error> {
    if k.func.arguments.len() != k.bindings.len() {
        return err(format!(
            "{}: {} parameters against {} bound buffers — a launch binds one address per parameter, \
             so the two must be the same length and in the same order",
            k.func.name,
            k.func.arguments.len(),
            k.bindings.len()
        ));
    }
    let consts = index_constants(&k.func);
    let stored = store_views(&k.func);
    let mut out = Vec::with_capacity(k.bindings.len());
    for (i, ((ptr, _), tid)) in k.func.arguments.iter().zip(k.bindings.iter()).enumerate() {
        let view = k
            .func
            .operations
            .iter()
            .find(|o| {
                o.op_type == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(ptr)
            })
            .ok_or_else(|| Error {
                message: format!(
                    "{}: parameter {i} (t{tid}) states no `ktdp.construct_memory_view`, so nothing \
                     says how wide the buffer it addresses is",
                    k.func.name
                ),
            })?;
        let (rows, cols) = shape_2d(view).ok_or_else(|| Error {
            message: format!(
                "{}: parameter {i} (t{tid})'s view states no 2-D `shape`",
                k.func.name
            ),
        })?;
        // The FIRST access tile over this view — the window the program takes. A program that tiles
        // one buffer many ways (attention's per-head reads) is not addressed through this field;
        // those bodies read their geometry from the node kind's own facts instead.
        let (r_start, c_start, r_len, c_len) = k
            .func
            .operations
            .iter()
            .find(|o| {
                o.op_type == OpKind::KtdpConstructAccessTile
                    && o.operands.first() == view.result.as_ref()
            })
            .and_then(|t| {
                let (tr, tc) = shape_2d(t)?;
                let rs = t.operands.get(1).and_then(|s| consts.get(s).copied())?;
                let cs = t.operands.get(2).and_then(|s| consts.get(s).copied())?;
                Some((rs as u32, cs as u32, tr, tc))
            })
            .unwrap_or((0, 0, rows, cols));
        // ⛔⛔⛔ A WINDOW STATED INSIDE AN `scf.for` IS NOT A WINDOW THIS WALK CAN SEE, AND THE
        // FALLBACK ABOVE IS SILENT ABOUT IT.
        //
        // Every reader here — the `find` above, `param_tiles`, `r_cover` below — walks
        // `f.operations`, which is the TOP LEVEL only. `IRFunction::ops_deep`'s own doc says what
        // that costs: "a time-tiled op puts its whole computation inside an `scf.for` body, so
        // anything asking 'what does this program do' and reading `operations` alone sees a loop and
        // nothing else — and reports the tiled case as empty rather than as tiled". Here the empty
        // case is not reported at all: `unwrap_or((0, 0, rows, cols))` says "the window is the whole
        // buffer", which is the ONE answer a K-blocked program never means. A `[128, 256]` weight
        // read `[128, 64]` per trip is then described at `[128, 256]`, and the descriptor computes a
        // tile four times the width of the one the program states — well-formed, and wrong.
        //
        // ⭐ INERT FOR A STRAIGHT-LINE PRODUCER, WHICH IS EVERY PROGRAM THIS CRATE HAS SEEN. Neither
        // this crate nor the producer mentions `scf.for` anywhere (`grep -rn ScfFor` is empty in
        // both), so the guard fires only for a program shape that has never reached here — a Triton
        // front end's K-blocked MLP, whose windows all live in the loop body. It changes nothing for a
        // program whose tiles are at the top level, and nothing for one with no tiles at all.
        // ⛔ AND IT IS `deep > top`, NOT `top == 0`. A parameter loaded whole ABOVE a loop and
        // re-blocked INSIDE it has a top-level tile, so a "no window at the top level" test passes it
        // — and then `r_cover` silently omits every in-loop window from the span it hands
        // [`node_rows`], which is the same unaccounted-window defect with a first block in front of
        // it. Counting both sides means the guard asks the real question: is there a window this walk
        // cannot see? Equal counts is the straight-line case and stays free.
        let top = param_tiles(&k.func, ptr).count();
        let deep = param_tiles_deep(&k.func, ptr);
        if deep > top {
            return err(format!(
                "{}: parameter {i} (t{tid}) states {} of its {deep} `ktdp.construct_access_tile` \
                 window(s) INSIDE a region (an `scf.for` body), and every reader here walks the \
                 function's TOP LEVEL — so {} and every body below would describe a tile the program \
                 never takes. A time-tiled program has to be split into one node per trip, or its \
                 windows hoisted, before a descriptor can span it.",
                k.func.name,
                deep - top,
                if top == 0 {
                    format!("the window would fall back to the whole `[{rows}, {cols}]` view")
                } else {
                    format!("the {top} visible window(s) would stand for all {deep} of them")
                },
            ));
        }
        // ⛔⛔⛔ EVERY access tile's ROW SPAN, because "the FIRST" is not "the program's". A producer
        // arm that ROW-BLOCKS its region for the emulator's LX (silu_mul; main's
        // `lower_elementwise_node_rows`, `lower_scalarmul_node`'s `by_row`) states one access tile per
        // block, and the field above keeps only block 0 — so a ported body reading its row count off
        // it emitted main's descriptors over the BLOCK HEIGHT and every later block's rows were never
        // written. MEASURED, granite-3.1-2b prefill at mq=31: `silu_o461`/`mulsilu_o461` carried
        // `mb_=16` where every other op in the layer carried `mb_=31`, so rows 16..30 of all 40
        // layers' MLP output kept whatever the buffer held. This states the span so
        // [`node_rows`] can prove the blocks tile the node before a body spans them with one
        // descriptor.
        //
        // ⛔ AND IT IS EVERY VIEW OF THE PARAMETER, NOT THE ONE FOUND ABOVE. The producer mints
        // a FRESH `ktdp.construct_memory_view` per call and `store_region` calls it once per block, so
        // a row-blocked program states `ceil(m / blk)` views of the same pointer — the `find` above
        // reaches block 0's alone. Collecting only its tiles reported `rows 0..16` for a 31-row node
        // (and would have refused the build instead of fixing it).
        let param_views: Vec<Ssa> = k
            .func
            .operations
            .iter()
            .filter(|o| {
                o.op_type == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(ptr)
            })
            .filter_map(|o| o.result)
            .collect();
        let r_cover = k
            .func
            .operations
            .iter()
            .filter(|o| {
                o.op_type == OpKind::KtdpConstructAccessTile
                    && o.operands.first().is_some_and(|v| param_views.contains(v))
            })
            .filter_map(|t| {
                let (tr, _) = shape_2d(t)?;
                let rs = t.operands.get(1).and_then(|s| consts.get(s).copied())? as u32;
                Some((rs, rs + tr))
            })
            .fold(None::<(u32, u32)>, |acc, (s, e)| match acc {
                None => Some((s, e)),
                Some((s0, e0)) => Some((s0.min(s), e0.max(e))),
            })
            .unwrap_or((0, rows));
        // `r_cover`'s twin over columns — see the field's doc for who reads it.
        let c_cover = k
            .func
            .operations
            .iter()
            .filter(|o| {
                o.op_type == OpKind::KtdpConstructAccessTile
                    && o.operands.first().is_some_and(|v| param_views.contains(v))
            })
            .filter_map(|t| {
                let (_, tc) = shape_2d(t)?;
                let cs = t.operands.get(2).and_then(|s| consts.get(s).copied())? as u32;
                Some((cs, cs + tc))
            })
            .fold(None::<(u32, u32)>, |acc, (s, e)| match acc {
                None => Some((s, e)),
                Some((s0, e0)) => Some((s0.min(s), e0.max(e))),
            })
            .unwrap_or((0, cols));
        let is_out = view.result.is_some_and(|v| stored.contains(&v));
        // ⭐⭐⭐ WALL 13 — THE STICK BASIS IS THE DATA'S OWN `df`, READ OFF THE VIEW ITSELF. The
        // view's `Dtype` attribute is the one statement of the buffer's element type ([`param_df`]
        // reads the same op for the placement mint's sizing half), so the region carries it as its
        // `df` and every door the region reaches classifies at the DATA'S OWN stick: an f32
        // parameter's 32 survivor lanes are one WHOLE fp32 stick, not half an fp16 one.
        //
        // ⛔ A DTYPE WITH NO STICK GEOMETRY (bool, i64) KEEPS THE Fp16 BASIS, STATED HONESTLY. The
        // reduce/pointwise DDL families bind no operand at those formats, so no classification
        // this field drives can be reached by one — the basis is inert for them, and inventing a
        // `Df` variant the vendor does not bind would be a knob, not a fact. [`param_df`]'s own
        // `None` contract is the same law for a caller that sizes buffers.
        let dtype = view.attributes.iter().find_map(|(key, v)| match (key, v) {
            (AttrKey::Dtype, Attr::Dtype(d)) => Some(*d),
            _ => None,
        });
        let is_fp8 = dtype.is_some_and(|d| d == ktir_core::dtypes::DType::Fp8E4m3);
        let df = dtype
            .and_then(super::whole_function::df_of_dtype)
            .unwrap_or(Df::Fp16);
        out.push(Region {
            tid: tid.get(),
            v_rows: rows,
            v_cols: cols,
            r_start,
            c_start,
            r_len,
            c_len,
            r_cover,
            c_cover,
            is_out,
            is_fp8,
            // The view's OWN element type (wall 13): every f16 parameter keeps the fp16 basis
            // byte-identical to the pre-wall-13 hardcode; an f32 parameter states IEEE_FP32.
            df,
        });
    }
    Ok(out)
}

/// The element format of binding `i`'s own view — the `Df` its `ktdp.construct_memory_view` states,
/// for a caller that must size a buffer the way the program itself describes it.
///
/// ⭐ THE PLACEMENT MINT'S SIZING HALF. A host-staged per-step input ([`KtirNode::step_inputs`])
/// has no graph tensor, so no placement census ever stated its footprint; the ONE honest source for
/// the extent is the program's own view — the same op [`regions`] reads — and this hands a caller
/// the element format beside it. `None` says the view states a dtype with no stick geometry (bool,
/// i64), which no buffer this ladder sizes can be, and the caller refuses by name.
pub fn param_df(k: &KtirNode, i: usize) -> Option<crate::superdsc_opspec::Df> {
    let (ptr, _) = k.func.arguments.get(i)?;
    let view = k.func.operations.iter().find(|o| {
        o.op_type == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(ptr)
    })?;
    let dtype = view.attributes.iter().find_map(|(key, v)| match (key, v) {
        (AttrKey::Dtype, Attr::Dtype(d)) => Some(*d),
        _ => None,
    })?;
    super::whole_function::df_of_dtype(dtype)
}
fn store_views(f: &IRFunction<'static>) -> std::collections::HashSet<Ssa> {
    let tile_view: std::collections::HashMap<Ssa, Ssa> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpConstructAccessTile)
        .filter_map(|o| Some((o.result?, *o.operands.first()?)))
        .collect();
    f.operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpStore)
        .filter_map(|o| tile_view.get(o.operands.get(1)?).copied())
        .collect()
}

/// Every `arith.constant` index in the func, by the SSA it defines — an access tile's corner is one
/// of these, and a corner is what carries a column-chunked node's `region.cols.start`.
pub fn index_constants(f: &IRFunction<'static>) -> std::collections::HashMap<Ssa, i64> {
    let mut m = std::collections::HashMap::new();
    for op in f.operations.iter() {
        if op.op_type != OpKind::ArithConstant {
            continue;
        }
        let Some(r) = op.result else { continue };
        if let Some(v) = op.attributes.iter().find_map(|(kk, v)| match (kk, v) {
            (AttrKey::Value, Attr::Int(i)) => Some(*i),
            _ => None,
        }) {
            m.insert(r, v);
        }
    }
    // A Triton front end wraps its tile corners in `arith.index_cast` (an i64 constant narrowed
    // to the index width), so a cast of a constant is a constant too and is recorded under the
    // CAST's result. Chains resolve iteratively; a cast of anything non-constant stays absent.
    loop {
        let mut grew = false;
        for op in f.operations.iter() {
            if op.op_type != OpKind::ArithIndexCast {
                continue;
            }
            let Some(r) = op.result else { continue };
            let Some(src) = op.operands.first() else {
                continue;
            };
            if !m.contains_key(&r)
                && let Some(v) = m.get(src)
            {
                m.insert(r, *v);
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    m
}

/// A corner's affine form over the launch position: `stride · core_id + base`.
///
/// The door's half of the shared contract with triton-ktir's
/// `uniform_grid_corners::uniform_corners` — the recognizer that crate's fire gate consults
/// and whose pass mints the spelling this one reads. This crate is a serde-only leaf and
/// cannot depend on that one, so the family is stated twice; the bridge test in
/// `triton-ktir-superdsc` runs one module through both and asserts they agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uniform {
    pub stride: i64,
    pub base: i64,
}

/// Every SSA the affine-form propagation can state as `stride · core_id + base` over the
/// landmark — the door's twin of triton-ktir's `uniform_corners`, built the same way
/// `index_constants` builds its constants: one pass over the function's top-level ops in
/// program order (SSA definitions precede uses), seeded with the landmark itself.
///
/// SOUND by the same law as the twin: only ops closed under affine forms propagate — an
/// `index_cast` is transparent, an `addi` of a constant or another uniform composes, a
/// `muli` by a constant scales. Division, remainder, and a product of two position terms
/// are dropped, so nothing is claimed the twin cannot also claim. Only the landmark's own
/// `result` seeds the position: a multi-axis `ktdp.get_compute_tile_id` (scratchy's own
/// producer emits those) names its other axes in `result_names`, and an axis coordinate is
/// not the flat position — a corner riding one is not claimed rather than misclaimed.
pub fn uniform_corners(
    f: &IRFunction<'static>,
    consts: &std::collections::HashMap<Ssa, i64>,
) -> std::collections::HashMap<Ssa, Uniform> {
    let mut out: std::collections::HashMap<Ssa, Uniform> = std::collections::HashMap::new();
    let Some(landmark) = f
        .operations
        .iter()
        .find(|o| o.op_type == OpKind::KtdpGetComputeTileId)
        .and_then(|o| o.result)
    else {
        return out;
    };
    out.insert(landmark, Uniform { stride: 1, base: 0 });
    for op in f.operations.iter() {
        let Some(r) = op.result else { continue };
        // ⛔ THE UNPACK IS PER-ARM, NOT UP FRONT: an `arith.index_cast` is UNARY (operand 1 is
        // absent), so a two-operand tuple match would skip every cast — and the canonical
        // spelling is `index_cast(muli(index_cast(landmark), S))`, three casts deep. The casts
        // must propagate or NOTHING does.
        let form: Option<Uniform> = match op.op_type {
            OpKind::ArithIndexCast => op.operands.first().and_then(|a| out.get(a).copied()),
            OpKind::ArithAddi | OpKind::ArithMuli => {
                let (Some(a), Some(b)) =
                    (op.operands.first().copied(), op.operands.get(1).copied())
                else {
                    continue;
                };
                let (fa, fb, ka, kb) = (
                    out.get(&a).copied(),
                    out.get(&b).copied(),
                    consts.get(&a).copied(),
                    consts.get(&b).copied(),
                );
                if op.op_type == OpKind::ArithAddi {
                    match (fa, fb, ka, kb) {
                        (Some(x), Some(y), _, _) => {
                            compose(x.stride.checked_add(y.stride), x.base.checked_add(y.base))
                        }
                        (Some(x), None, _, Some(c)) => {
                            compose(Some(x.stride), x.base.checked_add(c))
                        }
                        (None, Some(y), Some(c), _) => {
                            compose(Some(y.stride), y.base.checked_add(c))
                        }
                        _ => None,
                    }
                } else {
                    // A product of two position terms is quadratic, not uniform: only mul by a
                    // CONSTANT propagates — the same deliberate narrowing the twin states.
                    match (fa, fb, ka, kb) {
                        (Some(x), None, _, Some(c)) => {
                            compose(x.stride.checked_mul(c), x.base.checked_mul(c))
                        }
                        (None, Some(y), Some(c), _) => {
                            compose(y.stride.checked_mul(c), y.base.checked_mul(c))
                        }
                        _ => None,
                    }
                }
            }
            _ => None,
        };
        if let Some(form) = form {
            out.insert(r, form);
        }
    }
    out
}

/// `Some` for a form inside the i32 domain, `None` for one that left it (or whose
/// arithmetic overflowed) — the same guard the twin applies, so a form one side refuses
/// the other refuses too.
fn compose(stride: Option<i64>, base: Option<i64>) -> Option<Uniform> {
    let stride = stride.filter(|s| (i32::MIN as i64..=i32::MAX as i64).contains(s))?;
    let base = base.filter(|b| (i32::MIN as i64..=i32::MAX as i64).contains(b))?;
    Some(Uniform { stride, base })
}

/// An op's `[.., rows, cols]` from its own `shape` attribute or result type — the same reading
/// `logical_2d` did, kept because a 1-D shape is a ROW (our per-channel vectors are `[hidden]`).
pub(crate) fn shape_2d(op: &Operation<'_>) -> Option<(u32, u32)> {
    let dims = op
        .attributes
        .iter()
        .find(|(k, _)| *k == AttrKey::Shape)
        .and_then(|(_, v)| match v {
            Attr::IntList(l) => Some(l.to_vec()),
            _ => None,
        })
        .or_else(|| op.result_type.and_then(|t| t.dims().map(|d| d.to_vec())))?;
    Some(match dims.as_slice() {
        [] => (1, 1),
        [a] => (1, *a as u32),
        // ⭐⭐⭐ RANK > 2 IS READ IN THE FLAT FRAME: every axis but the last folded into rows, the
        // last the columns. Row-major, a rank-N tile and its `[prod(leading..=second-last), last]`
        // flattening are the SAME BYTES at the same offsets — the batched paged-decode kernel's
        // `[B, G, PS]` score tile IS the `[B·G, PS]` tile its pointwise consumers address — so this
        // is a reading, not a relayout. For rank 2 the leading fold is the empty product (1) and
        // the pair is exactly what it always was, so every existing caller is byte-identical; the
        // OLD `[.., a, b]` arm took the LAST TWO AXES ONLY (`[256, 4, 256]` read as `(4, 256)`),
        // which drops the batch multiplier and was never reachable anyway — no rank-3 op survived
        // the ladder to reach it until the batched dot and the reshape bridge landed.
        [rest @ .., a, b] => (
            rest.iter()
                .copied()
                .chain(std::iter::once(*a))
                .product::<i64>() as u32,
            *b as u32,
        ),
    })
}

/// The registry slot a reserved-tid parameter names — the inverse of [`scalarmul_scale_tid`],
/// computed by search so no arithmetic about the reserved block is restated here.
fn scale_idx_of(layout: Option<&BundleLayout>, tid: u32) -> Option<usize> {
    let n = layout.map(|l| l.scalarmul_scales.len()).unwrap_or(0);
    (0..n).find(|&i| scalarmul_scale_tid(i) == tid)
}

/// The `TileOp` a node's kind declares, rebuilt from the extents the KTIR states.
///
/// ⛔ IT IS `node_to_tile_ops`' OWN DECLARATION, NOT A NEW ONE. That function builds `mb` from the
/// output's rows, `out` from its cols (`is_stick`), a size-1 `y`, and `n_operands` from the operand
/// count — pure shape, nothing that needs the node. `in`(K) is added for a matmul, and a ScalarMul's
/// `out` is the DEVICE width. Those are the three arms, and they are the three arms here — plus the
/// `y` PARAMETER the row-split broadcast mode adds: the one arm that states a real `y > 1` is the
/// leading-axis broadcast consumer (`elementwise_broadcast`'s Y arm, which derives `y` and
/// `mb = rows/y` from the checked [`crate::sdsc_abstract::RowSplitFrame`]); every other caller
/// passes 1 and keeps the size-1 dim this doc describes.
/// ⭐ WALL 13 — the tile op states its OWN df: every dim and the struct carry the output's element
/// format, which the bridge threads into every `TensorArg` (`.with_df`) and the LX-residency math
/// (`pointwise_lx_resident_generic`). Every existing caller passes the region's own `df`, so an
/// f16 op is byte-identical to the pre-wall-13 hardcode; an f32 op states IEEE_FP32 end to end.
pub(crate) fn pointwise_tile_op(rows: u32, cols: u32, y: u32, n_operands: u32, df: Df) -> TileOp {
    TileOp {
        kind: TileOpKind::PointwiseOrReduce { n_operands },
        dims: vec![
            ItDim {
                name: "mb",
                size: rows,
                is_reduction: false,
                is_stick: false,
                df,
            },
            ItDim {
                name: "out",
                size: cols.max(1),
                is_reduction: false,
                is_stick: true,
                df,
            },
            ItDim {
                name: "y",
                size: y.max(1),
                is_reduction: false,
                is_stick: false,
                df,
            },
        ],
        df,
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  THE PORTED BODIES. Each one is main's, verbatim below its door.
//
//  ⭐ EACH IS `pub`, AND THAT IS THE PUBLIC API. The name-keyed dispatch that used to call them is
//  `scratchy-target-spyre`'s `lower_ktir_to_superdsc::lower`, because two of them ([`attn_at`],
//  [`rope_at`]) are reached through a model-geometry const door whose match arms are generated from
//  the CONSUMER's model inventory. A third-party producer picks the body for its node kind and
//  crosses that door itself — the extraction's two recorded decisions ("the public API takes the
//  node kind explicitly", and "the caller crosses the geometry door"), made real.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// THE POINTWISE TABLE: an [`Elementwise`] kind's dxp op-func key and its arity.
///
/// ⛔ THE STRING IS A DESCRIPTOR NAME, NOT A LABEL. [`elementwise`] renders it as
/// `format!("{op_func}_o{tid}")`, and that name feeds the bundle FINGERPRINT — so each key here is
/// the EXACT `op_func_from_str` key, and the four that already shipped (`silu`, `gelu`, `multiply`,
/// `add`) keep their spelling to the byte. In particular `multiply` must NOT become `mul` and `gelu`
/// must NOT become `gelufwd`: those are `OpFunc::name()`'s dxp spellings, which the emitted
/// descriptor's `opFuncName` gets from the `OpFunc` — this key is the *input* to
/// `op_func_from_str`, a different string with a different job. The test module pins all four.
///
/// ⛔ ENUMERATED, NEVER `_`: a producer that gains a new elementwise kind is an E0004 here, which is
/// the whole reason the kind is not a `&str` any more.
pub fn elementwise_op_func(name: &str, kind: Elementwise) -> Result<(&'static str, usize), Error> {
    Ok(match kind {
        Elementwise::Silu => ("silu", 1),
        // A REAL DDL primitive (`OpFunc::Gelu`), not a decomposition: the SFP
        // constant table ships gelu's polynomial, so this is one pointwise op.
        Elementwise::Gelu => ("gelu", 1),
        Elementwise::Mul => ("multiply", 2),
        Elementwise::Add => ("add", 2),
        // ⛔ `Sub` WAS REFUSED HERE, AND THE STATED REASON WAS FALSE IN EVERY CLAUSE. It read "Sub
        // takes a [m, 1] broadcast operand that pw2 cannot express (see the two-operand arity
        // law)". There is no arity law on `pw2`; `In::col` expresses precisely the `[m, 1]` operand
        // and is live on card (`ktir_matmul_fp8`, `attn`, `rmsnorm`); and THIS FUNCTION NEVER CALLS
        // `pw2` — it calls `assemble_pointwise_seeded_from_tile`. A non-broadcasting subtract is
        // structurally identical to `add`/`multiply`, `OpFunc::Subtract` spells `"sub"`, and `"sub"`
        // is in the on-card-verified recognized set (`tests/superdsc_time_tile.rs`, asserted against
        // dxp's own `dscdefn.cpp opFuncsToString`). The reference evaluator permits `bc == ac ||
        // bc == 1` for Mul/Add/Sub ALIKE, so broadcasting was never Sub-specific either.
        //
        // The hazard the old refusal gestured at is real but general, and it is now guarded where it
        // belongs — [`pointwise_extents_agree`], for EVERY binary kind.
        Elementwise::Sub => ("sub", 2),
        // ── The rest of the `OpFunc` pointwise set. Each already had an `OpFunc` variant AND an
        // `op_func_from_str` arm before this table named it, and most are already emitted on card by
        // the attention / rmsnorm / fp8-quantiser bodies; what was missing was only the door.
        // The exp key is DERIVED from the exp-base law (`ExpBase`), never spelled — the device's
        // one exponential is e^x and the recurrence's spelling of it is one fact, not two.
        Elementwise::Exp => (crate::exp_base::ExpBase::E.op_func(), 1),
        Elementwise::Rsqrt => ("rsqrt", 1),
        Elementwise::Sqrt => ("sqrt", 1),
        Elementwise::Abs => ("abs", 1),
        Elementwise::Reciprocal => ("reciprocal", 1),
        Elementwise::Sigmoid => ("sigmoid", 1),
        Elementwise::Tanh => ("tanh", 1),
        Elementwise::Mish => ("mish", 1),
        Elementwise::RealDiv => ("realdiv", 2),
        Elementwise::Maximum => ("maximum", 2),
        Elementwise::Minimum => ("minimum", 2),
        // ⛔ THE DDL HAS NO PRIMITIVE FOR THESE TWO, AND SUBSTITUTING THE NEAREST ONE IS THE BUG.
        // `QuickGelu` is x·σ(1.702x) and `GeluErf` is the exact-erf gelu — neither is
        // `OpFunc::Gelu`'s tanh polynomial, so emitting "gelu" for them would run a DIFFERENT
        // function and report success. There is no `OpFunc` variant for either.
        //
        // They reach this emitter because the SHARED front end now expresses the whole
        // arch vocabulary — that is the point: the op exists in the IR, and the TARGET
        // says whether it has a kernel. Before, the lowering panicked for everyone.
        Elementwise::QuickGelu | Elementwise::GeluErf => {
            return err(format!(
                "Elementwise({kind:?}) {name} has no dxp DDL primitive: quick-gelu is x·σ(1.702x) \
                 and exact-erf gelu is the erf form, while OpFunc::Gelu is a TANH polynomial — \
                 emitting \"gelu\" for either would run a DIFFERENT function and report success. \
                 There is no OpFunc variant for either, so a target that needs them must decompose \
                 them in the producer."
            ));
        }
    })
}

/// ⛔⛔⛔ THE GUARD FOR THE FACT THIS EMITTER NEVER CHECKED: every operand is addressed at the
/// OUTPUT's extent, so an operand of a DIFFERENT extent is silently mis-addressed.
///
/// [`pointwise_opspec_from_tile`](crate::ir::bridge::tiled_op_sdsc_op::pointwise_opspec_from_tile)
/// computes `device_dims` ONCE — off the tile, which [`elementwise`] builds from the OUTPUT's
/// `[node_rows, c_len]` — and hands the SAME dims to every `TensorArg` it makes, inputs and output
/// alike. Nothing downstream re-reads an input's own view. So a `[1, n]` row-broadcast source or a
/// `[m, 1]` per-row scalar is addressed as a full `[rows, cols]` dense tile: the descriptor walks
/// past the end of the operand and every row after the first reads the wrong bytes. It compiles, it
/// bakes, and it reports success.
///
/// ⛔ IT IS A REFUSAL, NOT A WARNING, AND IT IS NOT SPECIAL-CASED. A broadcast operand IS
/// expressible on this device — but only through the [`EwOperand`](crate::emit::EwOperand) builders,
/// `pw2` with `In::col`, which mark the operand out-broadcast so the descriptor states the
/// broadcast. This seeded whole-tensor path has no way to say it, so the only honest answer is to
/// name the operand and stop.
///
/// ⭐ EVERY INPUT, NOT JUST A BINARY'S SECOND. The builder gives a unary's single input the output's
/// dims too, so the same silence applies; the invariant is "all operands share the output's extent",
/// and stating it per-arity would be the scattered logic this crate exists to avoid.
/// ⭐⭐⭐ TWO BASES, ONE LAW: every operand must hold the op's row sweep (`rows` — the basis
/// [`pointwise_rows`] reads off the OUTPUT), and the basis an operand holds it in is the
/// operand's own.
///
/// A WHOLE-VIEW operand holds the sweep by BEING it (`v_rows == rows`, at row 0) — the seeded
/// family's shape, and the only shape the old law here described (it compared `v_rows` against
/// the output's `v_rows`). A WINDOWED operand — a banded program's per-core window, or a minted
/// intermediate consumed by a banded op — holds the sweep as its WINDOW (`r_len == rows`, fitting
/// its view), and comparing its `v_rows` instead would refuse exactly the programs the band lift
/// admits: a banded output's whole view is every core's bands, not this op's sweep. The c_len
/// equality is unchanged — the op's width is one number every operand shares.
///
/// ⛔⛔⛔ THE BASIS IS THE OPERAND'S OWN FORM, NEVER THE PROGRAM-LEVEL COVER. This law was first
/// implemented gated on `r_cover == (0, v_rows)`: a flush cover selected the whole-view basis,
/// anything else the windowed one. That conflated two facts — the cover is the UNION of every
/// window every trip over the parameter states, so it is flush exactly when the program's OTHER
/// trips tile the buffer, and it says nothing about how THIS op addresses THIS operand. MEASURED
/// on the spliced batched paged-decode attention (wall 10): the runtime length mask's four chunk
/// trips tile `[0, 4096)` flush, each trip's add consuming its own `[1024, 64]` window at row
/// `1024·c` — `r_len == rows == 1024`, the windowed basis's own case — while the flush cover
/// routed it to the whole-view basis (`v_rows == 4096 != 1024`) and refused it. Every whole-view
/// operand keeps its verdict — the two bases coincide there (`r_len == v_rows`) — so the
/// row-blocked family (`v_rows == rows` at row 0 over per-block windows) and the seeded family
/// stay admitted exactly as before, and the broadcast shapes both bases refuse stay refused.
///
/// ⭐ A PURE PREDICATE, DELIBERATELY: the law is stated as a `bool` over the `Region`'s own
/// fields so the kani proof pins it WITHOUT the refusal path's `format!` in reach — the fmt
/// machinery's loop forest is the grind site (MEASURED: 36 minutes in CBMC at 98.7% CPU, no
/// verdict, killed), and a law this small should verify in seconds. The wiring — that the guard
/// refuses exactly the operands this predicate rejects, with the message selected by the
/// operand's form — is pinned at runtime by the guard's grid test and the wall-10 unit test.
fn pointwise_operand_holds_sweep(x: &Region, rows: u32) -> bool {
    let whole_view = x.r_start == 0 && x.v_rows == rows;
    let windowed = x.r_len == rows && x.r_start + x.r_len <= x.v_rows;
    whole_view || windowed
}

fn pointwise_extents_agree(
    name: &str,
    kind: Elementwise,
    ins: &[Region],
    out: &Region,
    rows: u32,
) -> Result<(), Error> {
    for (i, x) in ins.iter().enumerate() {
        if !pointwise_operand_holds_sweep(x, rows) || x.c_len != out.c_len {
            if x.r_start == 0 && x.r_len == x.v_rows {
                return err(format!(
                    "Elementwise({kind:?}) {name}: input {i} t{} is `[{}, {}]` (view rows × window \
                     cols) but the output t{} is `[{}, {}]`, and this emitter addresses EVERY operand \
                     at the OUTPUT's extent — `pointwise_opspec_from_tile` computes `device_dims` once \
                     from the output tile and hands the same dims to every TensorArg, so nothing \
                     downstream ever reads this input's own view. The descriptor would read a {}-row × \
                     {}-col tile out of a {}-row × {}-col operand, running off its end and taking the \
                     wrong bytes for every row after the first — silently, and reporting success. \
                     A row-broadcast `[1, n]` source or a per-row `[m, 1]` scalar is expressible only \
                     through the EwOperand builders (`pw2` + `In::col`), which mark the operand \
                     out-broadcast so the descriptor STATES the broadcast; this seeded whole-tensor \
                     path cannot express it, so it refuses rather than mis-address it.",
                    x.tid,
                    x.v_rows,
                    x.c_len,
                    out.tid,
                    out.v_rows,
                    out.c_len,
                    rows,
                    out.c_len,
                    x.v_rows,
                    x.c_len,
                ));
            }
            return err(format!(
                "Elementwise({kind:?}) {name}: input {i} t{}'s window is `[{}, {}]` at row {} of its \
                 `[{}, {}]` view, but the op sweeps `[{}, {}]` (the output t{}'s basis) — a windowed \
                 operand must hold the sweep as its OWN window (`r_len` the sweep, fitting the view), \
                 or the descriptor would read the wrong rows of this operand for the rows the output \
                 states. The column extent is shared (`c_len`), so a mismatch there is the same \
                 refusal; a row-broadcast `[1, n]` source or a per-row `[m, 1]` scalar is expressible \
                 only through the EwOperand builders (`pw2` + `In::col`), which state the broadcast.",
                x.tid, x.r_len, x.c_len, x.r_start, x.v_rows, x.v_cols, rows, out.c_len, out.tid,
            ));
        }
    }
    Ok(())
}

/// WHICH AXIS a pointwise operand is broadcast along.
///
/// The three are not interchangeable and the device says which through a different [`In`] builder, so
/// they are a variant rather than a `bool`: [`In::col`] marks the operand out-broadcast (a per-row
/// scalar sprayed along the stick axis) and [`In::mb`] marks it mb-broadcast (a row vector sprayed
/// down the rows). Emitting one for the other reads a `[m, 1]` operand as a `[1, n]` and takes the
/// wrong bytes for every row after the first. [`In::y_broadcast`] marks the ROW-SPLIT form — a
/// multi-row `[mb, out]` source sprayed down the op's LEADING dim — whose count is carried in the
/// variant because the op's `[mb, out, y]` facts are a function of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcastAxis {
    /// `[m, 1]` → `[m, n]`: a per-row scalar (a row reduction's result) sprayed along the columns.
    /// The softmax's `m[:, None]` and `l[:, None]`.
    Col,
    /// `[1, n]` → `[m, n]`: a row vector (an rmsnorm gain) sprayed down the rows. `n1[None, :]`.
    Mb,
    /// `[mb, out]` → `[y, mb, out]`: a MULTI-row source sprayed down a new LEADING dim — the
    /// row-split form, the batched paged-decode kernel's J-slot merge (`[64, 4] -> [4, 64, 4]`,
    /// `Dimensions=[0]` over a source the walk-back proved to be `y·mb` rows). The count is the
    /// broadcast's own stated leading extent; the consumer checks it against the source region's
    /// frame ([`crate::sdsc_abstract::RegionFrame::split_rows`]) before building the operand.
    Y(u32),
}

/// main's `lower_elementwise_node` (main 8362-8432). Its door: the [`Elementwise`] kind is stated by
/// the caller, the operand names are the parameters, and the row count is the node's ([`node_rows`] —
/// the producer row-blocks a region wider than the LX holds, so the first window is not the node).
///
/// `bcast[i]` is the axis input `i` is broadcast along, `None` for a dense operand. The caller proves
/// it from the program (see `whole_function::program_broadcast_chains`, which reads the
/// `expand_shape`'s degenerate dim and cross-checks the `linalg.broadcast`'s `Dimensions`, then
/// checks the axis against the resolved source's own extent). An empty slice means "no operand is
/// broadcast", which is every existing caller.
pub fn elementwise(
    name: &str,
    kind: Elementwise,
    r: &[Region],
    bcast: &[Option<BcastAxis>],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let (op_func, arity) = elementwise_op_func(name, kind)?;
    // ⭐ SPLIT WITHOUT THE BASE-ADDRESS GUARD: the dense arm below places row corners
    // per-operand (the windowed emission), so which operands are inputs is a fact about the
    // parameter list, not about how one arm addresses them. The arms that CANNOT carry a corner
    // are refused it by the guard just below; every other body keeps `split_out`'s guard.
    let (ins, out) = split_regions(name, r, layout, arity, &[])?;
    // ⛔⛔⛔ WALL 13 — THE MIXED-df PIN, ELEMENTWISE SEAT (the auditor's law, the walk-level
    // enforcement of the reduce.rs const law): every operand of ONE elementwise op states the
    // SAME df, the output's own. A TTIR op whose operands differ in df is a producer defect —
    // MLIR's own semantics insert the `arith.extf`/`arith.truncf` the promotion requires, so a
    // mixed op reaching here uncast means the producer skipped the cast the vocabulary mandates.
    // Refused BY NAME (never emitted, never re-typed): together with the convert door's own
    // pair check this makes the DD2 mixed-op refusal ("Unsupported result precision
    // conversion") UNREACHABLE by our emission — every df transition is a convert op.
    for (i, x) in ins.iter().enumerate() {
        if x.df != out.df {
            return err(format!(
                "Elementwise({kind:?}) {name}: input {i} t{} states df {:?} but the output t{} \
                 states {:?} — a MIXED-df op. The vocabulary requires the transition as an \
                 explicit `arith.extf`/`arith.truncf` (which this walk lowers as a convert with \
                 each side at its own df), so an uncast mixed op is a producer defect; emitting \
                 it as one descriptor would hand dxp a mixed [_, _] op the DDL refuses at bake",
                x.tid, x.df, out.tid, out.df
            ));
        }
    }
    // ⛔⛔⛔ WALL 13 — THE FP32 ALLOWLIST, fail-closed with TWO different laws named (never
    // conflated): some opFuncs the HARDWARE lacks at fp32 — the DDL's own comments say so
    // (`tanh`/`gelu`/`mish`/`fastexp`/`fastsigmoid`, "not possible in fp32 (FEST hardware
    // limitation)", unary_pipeline.ddl:20-21 and unary_parallel.ddl) — and the rest
    // (`sigmoid`/`rsqrt`/`sqrt`/`abs`/`silu`/`reciprocal`) ARE fp32-bound in the vendor's
    // unary_parallel.ddl (:22-:37) but OUTSIDE this island's tested set, so the island refuses
    // them as NARROW-BY-DEFAULT, not as hardware-absent. The merge chain this island exists for
    // needs exactly {add, sub, multiply, realdiv, maximum, minimum, exp} — the family
    // broadcast_ops.ddl binds at [%type_fp16, %type_fp32] plus unary_parallel.ddl:30's `exp`.
    if out.df == Df::Fp32 {
        let hardware_lacks = matches!(
            kind,
            Elementwise::Tanh
                | Elementwise::Gelu
                | Elementwise::Mish
                | Elementwise::QuickGelu
                | Elementwise::GeluErf
        );
        let in_allowlist = matches!(
            kind,
            Elementwise::Add
                | Elementwise::Sub
                | Elementwise::Mul
                | Elementwise::RealDiv
                | Elementwise::Maximum
                | Elementwise::Minimum
                | Elementwise::Exp
        );
        if hardware_lacks {
            return err(format!(
                "Elementwise({kind:?}) {name}: the output t{} states IEEE_FP32 and `{op_func}` is \
                 HARDWARE-ABSENT at fp32 — the vendor's own DDL comments say so (\"not possible \
                 in fp32 (FEST hardware limitation)\", unary_pipeline.ddl:20-21); the fp16 bind is \
                 the only one. A producer that needs this function at f32 must keep the value at \
                 f16 through it (cast, compute, cast back)",
                out.tid
            ));
        }
        if !in_allowlist {
            return err(format!(
                "Elementwise({kind:?}) {name}: the output t{} states IEEE_FP32 and `{op_func}` is \
                 not in the fp32 island's allowlist — the vendor DOES bind it at fp32 \
                 (unary_parallel.ddl:22-:37), but this island has tested only the merge chain's \
                 family ({{add, sub, multiply, realdiv, maximum, minimum, exp}}, \
                 broadcast_ops.ddl's [%type_fp16, %type_fp32] binds plus unary_parallel.ddl:30). \
                 Widen the allowlist only with a DDL citation and a pin, never by removal of this \
                 refusal",
                out.tid
            ));
        }
    }
    if !bcast.is_empty() && bcast.len() != ins.len() {
        return err(format!(
            "Elementwise({kind:?}) {name}: {} broadcast flags for {} inputs — the flags are \
             POSITIONAL (flag `i` selects input `i`'s operand mode), so a length mismatch would \
             address some other operand as the vector",
            bcast.len(),
            ins.len(),
        ));
    }
    // ⛔⛔⛔ THE BROADCAST AND PER-ROW ARMS PLACE ONLY THE CORNERS THEIR CORNERED OPERAND MODES
    // CAN STATE. A broadcast operand is addressed at its OWN extent (`In::col`: one lane per row;
    // `In::mb`: one row of values) and the per-row chain writes LANE 0 of a one-stick output at
    // the buffer base — neither form can state "columns `[c_start, c_start+c_len)` of a wider
    // stick-blocked tensor" (what a column-blocked chunk is) nor "rows `[r_start, r_start+r_len)`
    // of a taller one" (what a banded program's window is). TWO cornered forms are legal: the Mb
    // mode's own — an Mb-broadcast source at a whole-stick COLUMN corner of its flat frame, the
    // per-block slice shape (`In::mb_at` — the card-proven prefix-mask law, one stick-wide block
    // at a time) — and the Y mode's own — a row-split source at a per-trip ROW window of a
    // region-stated view, the prefill mask (`In::y_at`): the source is single-stick by the frame
    // law, so the window's row corner composes additively with the mb-row addressing. Each
    // carries its corner into the descriptor, and the broadcast extent check inside
    // [`elementwise_broadcast`] still refuses a window whose shape the mode cannot address — the
    // exemptions here are placement facts, not shape admissions. Everything else — the output,
    // any corner on a dense/Col operand, a Y corner off column 0, any corner in the per-row
    // chain — is a producer shape this door refuses BY NAME rather than mis-addressing (the same
    // contract `base_addressed` states for a row corner on every other body). The DENSE arm
    // below carries both corners; this guard is what keeps the others from silently dropping
    // them.
    let corner_unplaceable = |i: usize, x: &Region| {
        (x.c_start != 0 || x.r_start != 0)
            && !(matches!(bcast.get(i), Some(Some(BcastAxis::Mb))) && x.r_start == 0)
            && !(matches!(bcast.get(i), Some(Some(BcastAxis::Y(_)))) && x.c_start == 0)
    };
    if (bcast.iter().any(Option::is_some) || out.c_len == 1)
        && (out.c_start != 0
            || out.r_start != 0
            || ins
                .iter()
                .enumerate()
                .any(|(i, x)| corner_unplaceable(i, x)))
    {
        let (who, what, cs) = ins
            .iter()
            .enumerate()
            .find(|&(i, x)| corner_unplaceable(i, x))
            .map(|(i, x)| {
                if x.r_start != 0 {
                    (format!("input {i} t{}", x.tid), "row", x.r_start)
                } else {
                    (format!("input {i} t{}", x.tid), "column", x.c_start)
                }
            })
            .unwrap_or_else(|| {
                if out.r_start != 0 {
                    (format!("the output t{}", out.tid), "row", out.r_start)
                } else {
                    (format!("the output t{}", out.tid), "column", out.c_start)
                }
            });
        return err(format!(
            "Elementwise({kind:?}) {name}: {who} states {what} corner {cs} of its view, and the \
             arm this shape routes to (a broadcast operand or the per-row one-stick chain) \
             addresses its operands at the BUFFER BASE or at a broadcast mode — the one corner \
             those arms can place is a COLUMN window on an Mb-broadcast source (`In::mb_at`, the \
             per-block slice law) or a row window on a Y-broadcast source (`In::y_at`, the \
             per-trip mask law), which this operand is not. A column-blocked chunk or a banded \
             window is DENSE elementwise; the dense arm carries both corners, so give the \
             broadcast/per-row chains whole-tensor regions (or an Mb source a whole-stick column \
             window, or a Y source a row window at column 0).",
        ));
    }
    if bcast.iter().any(Option::is_some) {
        return elementwise_broadcast(name, kind, &ins, &out, bcast, sym_id_base, layout);
    }
    // ⭐⭐⭐ THE PER-ROW CHAIN TAKES THE ONE-STICK ARM — see [`elementwise_per_row`]. A `[rows, 1]`
    // output is the running-max/alpha/denominator chain of an online softmax, which the
    // hardware-proven attention body lowers at one stick wide (value in lane 0), not one column.
    if out.c_len == 1 {
        return elementwise_per_row(name, kind, &ins, &out, sym_id_base, layout);
    }
    // ⛔ THIS COMMENT USED TO SAY `assemble_pointwise` EMITS THE SFP POLYNOMIAL TABLE "via
    // `constant_info(op_func)`". THERE IS NO SUCH FUNCTION. `constant_info` is a local in `emit_sdsc`
    // derived from the TYPED `op.op_info`, and the table is gated on
    // `OpFunc::needs_sfp_const_table()`, whose body is `let _ = self; false` — UNCONDITIONALLY false.
    // So `OpInfo::SfpConstTable` is never selected and `sfp_constant_table()` is dead on this path.
    //
    // That is not an accident: the crate's own note on `needs_sfp_const_table` records that shipping a
    // 20-entry table SHADOWED the DDL's internal Newton bit-trick constants and made `rsqrt` compute
    // `1/x`, and the fix was to stop shipping it rather than to hand-build the iteration. The DSM fills
    // them. A comment claiming we emit the table describes the bug, not the fix.
    let cols = out.c_len;
    check_pointwise_cols(cols, out.df, "Elementwise", out.tid)?;
    // ⭐ THE SWEEP IS COMPUTED BEFORE THE EXTENTS LAW, because the law is now stated OVER it: a
    // windowed (banded) output's sweep is its window, not its whole view, and the operands are
    // held to that basis ([`pointwise_extents_agree`]'s two-basis law below).
    let rows = pointwise_rows(name, &out)?;
    pointwise_extents_agree(name, kind, &ins, &out, rows)?;
    let op_name = format!("{op_func}_o{}", out.tid);
    // ⛔⛔⛔ A COLUMN-BLOCKED CHUNK IS DENSE ELEMENTWISE WITH A CORNER, AND THE CORNER MUST RIDE
    // EVERY OPERAND — the 12b gemma-4 garbage. `subtile_ir.rs`'s column-tiling set
    // (`EwKind::Silu | Mul | Add`, `ScalarMul`, `SiluMul`) splits a wide op into nb-blocks that
    // SHARE one output tensor; `SiluMul`'s body (`silumul`) carries each chunk's `region.cols.start` through `col_of`, but this door emitted every chunk
    // at offset 0 — so chunk 1 RE-COMPUTED chunk 0's columns at base 0 and the columns past the
    // first block were NEVER WRITTEN. MEASURED on the 12b card run: the gated-act product
    // `t1121[8192..15360]` read back ALL ZERO at every one of 48 layers, the down-GEMM contracted
    // a half-zero activation (residual row-norm correlates 0.937 with the chunk-1 activation
    // norm), and |d_card|/|d_emu| = 0.687 against sqrt(8192/15360) = 0.73 predicted for a missing
    // chunk. The tiny fixture never split (one block), which is why the class was invisible at
    // small geometry. The fix is `silumul`'s own mechanism, verbatim: every read `In::sliced`
    // through `col_of(rows, full_cols, c_start)`, the write at the same corner, `full_cols` the
    // WHOLE tensor width (`c_start + c_len`) because the nest's row stride is a property of the
    // storage, not of this chunk's view.
    //
    // ⭐ A WHOLE-TENSOR PROGRAM KEEPS THE SEEDED EMISSION, BYTE-IDENTICAL. The `broadcast_off`
    // path and the seeded path produce different `TensorArg` forms at offset 0 (rank-2
    // stick-major vs the seeded rank-3), and every whole-tensor elementwise program this crate
    // has ever baked went through the seeded one — nine-plus fixtures verified on card. The
    // offset is the ONLY thing this change adds, so it is the only thing that may change the
    // bytes: any corner on any operand routes to the offset form, and a corner-free program does
    // not.
    //
    // ⛔ AND "CORNER-FREE" MEANS ROWS TOO. The seeded form addresses every operand at the buffer
    // base — a banded op whose every COLUMN corner happens to be 0 (a rope copy's first half)
    // would silently read the wrong band at the base if the row corner were not part of this
    // gate. The row term joins the column term here rather than in `base_addressed`, because the
    // arm below this one CAN place it.
    if out.c_start == 0
        && ins.iter().all(|x| x.c_start == 0)
        && out.r_start == 0
        && ins.iter().all(|x| x.r_start == 0)
    {
        let in_names: Vec<String> = ins.iter().map(|x| x.name()).collect();
        let in_refs: Vec<&str> = in_names.iter().map(|s| s.as_str()).collect();
        let tile_op = pointwise_tile_op(rows, cols, 1, arity as u32 + 1, out.df);
        let op = assemble_pointwise_seeded_from_tile(
            &op_name,
            &tile_op,
            op_func,
            &in_refs,
            &out.name(),
            sym_id_base,
            layout,
        );
        return Ok(vec![op]);
    }
    // ⛔⛔⛔ THE CORNERS ARE PER-OPERAND, AND THE PAIRING IS THE PROGRAM'S OWN — the rotate-half
    // finding. The old law here required every input's column corner to equal the output's,
    // because the tiling front end mints one block per operand set and nothing else was known to
    // reach this arm. But a rotate-half rope pairs the halves' windows over ONE view: `x1` reads
    // columns `[0, 64)` while `x2` reads `[64, 128)` of the SAME tensor, and each product's
    // operands sit at their OWN corners (`x1·cos` reads `x1@0` against `cos@0`, `x2·sin` reads
    // `x2@64` against a `sin` window of its own view) — a single shared corner cannot state any
    // of it. So the corner is per-operand now: every operand — input or output — is addressed at
    // the window ITS OWN region states, and the extents law above is what makes that safe (every
    // operand holds the same `[rows, cols]` sweep; only the corner differs). Equal corners —
    // every column-blocked chunk the front end mints — emit exactly the bytes the shared-corner
    // form did: same offset on every operand, same [`crate::addr::ColWindow`].
    //
    // ⭐ TWO FORMS, ONE BUILDER, PICKED BY THE ROW CORNER. At row 0 the window is a whole-stick
    // COLUMN window of a same-rows stick-major parent — a [`crate::addr::ColWindow`], so the
    // piece is presented stick-major and the work-division law reaches it (#622; the 12b gemma-4
    // chunk family, card-proven). At a nonzero row corner — a BANDED program's window, one
    // core's row band of a taller view — the window is not a column window of a same-rows tensor
    // at all; it is a rectangle at `(r_start, c_start)` of the parent, addressed flat through
    // [`band_corner`] (`In::sliced` at an `rc_of` corner, the write at `OutCorner::At`). That
    // form is exactly the card-proven rotate-half geometry — a `[rows, 64]` window at a column
    // corner of a wider view — with the row term the corner adds.
    let out_off: crate::addr::OutCorner = if out.r_start == 0 {
        chunk_window(name, rows, &out)?.into()
    } else {
        crate::addr::OutCorner::At(band_corner(name, rows, &out)?)
    };
    // HANDLE-FLOW, `silumul`'s own discipline: the handles are bound once and the operands
    // borrow them, because an `EwOperand` holds a `&Stk` that must outlive the builder call.
    // ⭐ AND THE HANDLE DECLARES THE **STORAGE**, NOT THE WINDOW: each region's own view
    // `[v_rows, v_cols]` — the parent the corner arithmetic (`chunk_window`/`band_corner` over
    // `rc_of`'s nest) addresses within. A handle declared at the window `[rows, cols]` makes the
    // corner itself past-the-end (chunk 1 of the `[25, 15360]` MLP intermediate: corner 204800
    // against a 179200-element declaration), which the operand mint refuses — the same
    // annotation-only dims the `rbo` comment promises nothing reads but the checks.
    // ⭐ WALL 13 — each handle at its region's OWN df (the mint prices bytes at the handle's word
    // length; `rb` IS `rb_df(Dp16)`, so every f16 op is byte-identical).
    let out_h = rb_df(
        &out.name(),
        RowCount::of_token_rows(out.v_rows),
        BlockCols::of_feature_cols(out.v_cols),
        out.df,
    );
    let in_handles: Vec<_> = ins
        .iter()
        .map(|x| {
            rb_df(
                &x.name(),
                RowCount::of_token_rows(x.v_rows),
                BlockCols::of_feature_cols(x.v_cols),
                x.df,
            )
        })
        .collect();
    let ew: Vec<_> = in_handles
        .iter()
        .zip(ins.iter())
        .map(|(h, x)| {
            Ok(if x.r_start == 0 {
                In::col_window(h, chunk_window(name, rows, x)?).ew()
            } else {
                In::sliced(h, band_corner(name, rows, x)?).ew()
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let tile_op = pointwise_tile_op(rows, cols, 1, arity as u32 + 1, out.df);
    Ok(vec![assemble_pointwise_broadcast_off_from_tile(
        &op_name,
        &tile_op,
        op_func,
        rows,
        cols,
        &ew,
        &out_h,
        out_off,
        sym_id_base,
        layout,
    )])
}

/// ⛔⛔⛔ EVERY OPERAND A COLUMN-BLOCKED PIECE NAMES IS ADDRESSED AT THAT PIECE's WINDOW — the class law,
/// checked on what was EMITTED, not on what a body meant to emit.
///
/// The front end column-tiles a wide pointwise node into pieces that share one output tensor
/// (`subtile_ir.rs`: `Elementwise(Silu | Mul | Add)`, `ScalarMul`, `SiluMul`). Each piece's [`Region`]s state their corner; a body that drops it emits every
/// piece at the buffer base, and the columns past the first piece are never written. That happened
/// twice: `elementwise` (the gemma-4 12b gated-act product, half zero) and `scalarmul_scaled` (granite's
/// logits past column 8191, never written). Both bodies passed every test of their own, because each
/// test asked the body and the body agreed with itself.
///
/// So this asks the emission. For every op of the node and every operand binding naming a tensor the
/// program windows at a corner — a COLUMN corner, or the ROW corner of a banded window, which the
/// windowed pointwise emission places the same way — the binding's `offset_elems` must be that
/// window's corner in the tensor's storage (`rc_of` over the region's own view, the plane term
/// `c/64·v_rows·64` plus the row term; at row 0 it is the column corner alone); the node's ops must
/// start at least one write at the output's corner. A body that ignores a stated corner cannot
/// produce a bundle: the door returns this `Err` and the bake fails naming the op.
///
/// A whole-tensor node (every corner 0) passes without looking, so nothing that is not windowed
/// pays for it.
pub fn column_windows_are_honoured(
    name: &str,
    r: &[Region],
    ops: &[EmittedOp],
) -> Result<(), Error> {
    let windowed: Vec<&Region> = r
        .iter()
        .filter(|x| x.c_start != 0 || x.r_start != 0)
        .collect();
    if windowed.is_empty() {
        return Ok(());
    }
    for x in windowed {
        let rows = x.v_rows;
        let want =
            crate::addr::rc_of(rows, x.v_cols, x.r_start, x.c_start, Df::Fp16).into_raw_elems();
        let buf = x.name();
        let mut seen = false;
        for op in ops {
            for b in op.arg_bindings.iter().filter(|b| b.buffer == buf) {
                seen = true;
                if b.offset_elems != want {
                    return err(format!(
                        "{name}: op `{}` addresses t{} at element {} of its buffer, but the program \
                         windows that tensor at rows {}..{} × columns {}..{} — element {want} of its \
                         `[{rows}, {}]` storage. A windowed piece that does not address its own \
                         window recomputes another piece's bytes and leaves its own unwritten.",
                        op.op_name,
                        x.tid,
                        b.offset_elems,
                        x.r_start,
                        x.r_start + x.r_len,
                        x.c_start,
                        x.c_start + x.c_len,
                        x.c_start + x.c_len,
                    ));
                }
            }
        }
        if x.is_out && !seen {
            return err(format!(
                "{name}: the program writes t{} at rows {}..{} × columns {}..{}, and no emitted op \
                 names it",
                x.tid,
                x.r_start,
                x.r_start + x.r_len,
                x.c_start,
                x.c_start + x.c_len,
            ));
        }
    }
    Ok(())
}

/// ⭐ A COLUMN-BLOCKED PIECE's WINDOW as a [`crate::addr::ColWindow`] — the device element offset of
/// column `c_start` in its stick-blocked `[rows, full]` storage, `(c/64)·(rows·64)` through the
/// address nest (`full` the storage's width `c_start + c_len`, because the nest's row stride is a
/// property of the storage, not of this piece's view), carrying the fact that a column-blocked
/// piece is a whole-stick window of its stick-major storage, so the door presents it rank-2
/// stick-major and the work-division law reaches it (#622). `Err`, naming the op, when the piece's
/// corner is not a whole stick.
fn chunk_window(name: &str, rows: u32, out: &Region) -> Result<crate::addr::ColWindow, Error> {
    crate::addr::col_window_of(
        rows,
        out.c_start + out.c_len,
        pointwise_chunk_out_offset(out.c_start),
        Df::Fp16,
    )
    .map_err(|e| Error {
        message: format!("{name}: {e}"),
    })
}

/// ⭐ THE CORNER OF A BANDED WINDOW — a row AND column corner of a stick-blocked view, the
/// per-operand spelling the banded family's emission addresses through (`In::sliced` on the read
/// side, `OutCorner::At` on the write). A rope copy's window is `[band, 64]` at `(r_start, 0)` or
/// `(r_start, 64)` of its `[v_rows, 128]` view; the corner is that rectangle's first element in
/// the view's own nest, `rc_of` — the parent's plane stride `v_rows·64` and row stride `64` both
/// stated by the VIEW, never by the window.
///
/// ⛔⛔ SINGLE-STICK ONLY, AND THIS IS GEOMETRY, NOT TASTE. A window at a nonzero row corner is
/// addressable only while it stays inside ONE column plane of the parent: the parent strides its
/// planes `v_rows·64` apart while the window's own `[rows, c_len]` stick-major view strides them
/// `rows·64`, so a multi-stick window's second plane lands `rows·64` from the first in the window
/// and `(v_rows − rows)·64` further in the parent — not contiguous, and no flat or stick-major
/// presentation reaches both planes at once. A sub-stick width mis-strides the rows the same way
/// (`rows·c_len` against the parent's `rows·64`). The card-proven precedents for this exact form
/// are the rotate-half halves and the per-head attention blocks — `[rows, 64]` rectangles at a
/// corner of a wider view — and the band lift adds only the row term to that corner. Anything
/// wider than one stick at a row corner is refused BY NAME, here, where the geometry says so.
fn band_corner(name: &str, rows: u32, x: &Region) -> Result<crate::addr::DevOff, Error> {
    if x.c_len != FP16_ELEMS_PER_STICK || !x.c_start.is_multiple_of(FP16_ELEMS_PER_STICK) {
        return err(format!(
            "{name}: t{}'s window is `[{}, {}]` at column {} of its `[{}, {}]` view, at row \
             corner {} — a banded window is addressable only as ONE WHOLE \
             {FP16_ELEMS_PER_STICK}-element stick at a whole-stick column: the parent strides its \
             column planes `v_rows·64` elements apart while a window of {rows} rows would stride \
             its own `rows·64`, so a multi-stick window's planes are not contiguous in the parent, \
             and a sub-stick width mis-strides every row. Widen the sweep or keep the window at \
             row 0, where the whole-stick column-window form reaches any width.",
            x.tid, x.r_len, x.c_len, x.c_start, x.v_rows, x.v_cols, x.r_start,
        ));
    }
    Ok(crate::addr::rc_of(
        x.v_rows,
        x.v_cols,
        x.r_start,
        x.c_start,
        Df::Fp16,
    ))
}

/// ⭐⭐⭐ THE PER-ROW CLASS — `[rows, 1]` elementwise, the running-max/alpha/denominator chain of an
/// online softmax.
///
/// The seeded path above refuses `cols=1` because `assemble_pointwise` emits no `coordinateMasking_`
/// and a one-column output is not a whole stick. But the hardware-proven attention body HAS this
/// class and lowers it at ONE STICK WIDE, not one column: `attn.rs`'s `newm`/`corrsubt`/`corr`/
/// `lcorr`/`ladd` all state `BlockCols::of_one_stick(Lanes::FP16)`, the value rides in LANE 0 and
/// lanes 1..63 are never read (every consumer reads the buffer through `In::col`, the one-stick
/// broadcast mode, or a reduce accumulator that keeps the same lane-0 law). That is proven emission
/// on card, not a new shape being invented here.
///
/// The footprint already agrees: `synth_footprint_bytes` rounds a `[rows, 1]` intermediate's inner
/// extent up to a whole stick, so the reserved buffer is `rows × 64` fp16 — exactly what an op at
/// `cols = 64` spans. Emitting one column would UNDERWRITE the buffer, not preserve it.
///
/// ⛔ ONLY THE DENSE PER-ROW CHAIN TAKES THIS ARM. A broadcast operand still needs the operand-mode
/// path below (`elementwise_broadcast`), because a spray is an addressing mode, not a width. And
/// `rows` comes from [`node_rows`], the same source the dense path reads, so a windowed program is
/// spanned rather than re-blocked.
fn elementwise_per_row(
    name: &str,
    kind: Elementwise,
    ins: &[Region],
    out: &Region,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let (op_func, arity) = elementwise_op_func(name, kind)?;
    // EVERY operand must be the same per-row shape — this is the dense path's law ([`pointwise_extents_agree`])
    // restated for the one-stick emission: the op computes lane 0 of every row of every operand.
    for (i, x) in ins.iter().enumerate() {
        if (x.v_rows, x.c_len) != (out.v_rows, out.c_len) {
            return err(format!(
                "Elementwise({kind:?}) {name}: input {i} t{} is `[{}, {}]` but the output t{} is \
                 `[{}, {}]` — a per-row op computes LANE 0 of matching rows, so a mismatched operand \
                 is the same out-of-end read the dense path refuses",
                x.tid, x.v_rows, x.c_len, out.tid, out.v_rows, out.c_len,
            ));
        }
    }
    // ⭐ WALL 13 — the ONE-STICK width is the data's own df's stick (`Df::elems_per_stick`), not
    // the f16 hardcode: an f32 per-row chain writes lane 0 of a 32-lane IEEE_FP32 stick.
    let cols = out.df.elems_per_stick();
    let rows = pointwise_rows(name, out)?;
    let op_name = format!("{op_func}_o{}", out.tid);
    let tile_op = pointwise_tile_op(rows, cols, 1, arity as u32 + 1, out.df);
    let in_names: Vec<String> = ins.iter().map(|x| x.name()).collect();
    let in_refs: Vec<&str> = in_names.iter().map(|s| s.as_str()).collect();
    let op = assemble_pointwise_seeded_from_tile(
        &op_name,
        &tile_op,
        op_func,
        &in_refs,
        &out.name(),
        sym_id_base,
        layout,
    );
    Ok(vec![op])
}

/// ⭐⭐⭐ [`elementwise`] WITH A BROADCAST OPERAND — the arm [`pointwise_extents_agree`] names in its
/// own refusal, now built.
///
/// A broadcast is an ADDRESSING MODE, not an op: the AIU has no broadcast primitive and needs none.
/// [`EwOperand::scale`] turns [`In::col`] into `out = Scale::RedStick` (a one-stick `alpha_=0`
/// broadcast READ) and [`In::mb`] into `mb = Scale::RedNonStick`, so the DESCRIPTOR states which axis
/// is sprayed and the operand is read at its own extent instead of the output's. That is the whole
/// difference from the seeded path, which computes `device_dims` once off the output tile and hands
/// the same dims to every operand — which is why it has to refuse a degenerate one.
///
/// This is the same emission the hardware-proven bodies already use: `rmsnorm.rs` reads
/// `In::col_at(&rinv, …)` for the per-row `1/rms` (element 8, the first element of the exx2
/// output's E[x²] half-stick — same RedStick
/// broadcast, one-element view step in) and `In::mb(&gamma)` for the `[1, cols]` gain, and
/// `attn.rs`'s softmax reads `In::col(&hm(run_m))` for the per-row max. So the softmax a KTIR producer
/// spells longhand lowers to the operand modes the fused body was already proving on card.
///
/// ⭐ THE OPERAND'S BUFFER IS ALREADY STICK-WIDE, which is what makes a `[m, 1]` region addressable
/// as a per-row scalar. [`synth_footprint_bytes`](crate::placement::synth_footprint_bytes) rounds the
/// INNER extent up to a whole stick, so the `[m, 1]` intermediate a row reduction writes reserves
/// `m × 64` fp16 — byte-identical to the `rb(name, rows, 64)` handle `rmsnorm.rs` mints for `rinv`.
/// The per-row value sits in lane 0 of each row and `Scale::RedStick` reads exactly that.
///
/// ⛔ THE EXTENTS ARE RE-CHECKED HERE, NOT TRUSTED. The caller proves the axis from the program's
/// attributes; this door proves it from the operand's REGION. The two are independent, and the
/// emission is only correct when both hold — a `Col` flag over an operand that is really `[m, n]`
/// would read one lane and spray it over the whole output, silently.
fn elementwise_broadcast(
    name: &str,
    kind: Elementwise,
    ins: &[Region],
    out: &Region,
    bcast: &[Option<BcastAxis>],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    // THE OP FUNC IS A FUNCTION OF THE KIND, so it is named here rather than passed. The caller holds
    // it already, but [`elementwise_op_func`] is that one `match` and nothing else, so handing it over
    // as well would buy a parameter and no fact — and the call already proved it Ok for this `kind`.
    let (op_func, _) = elementwise_op_func(name, kind)?;
    // ⛔ A UNARY'S ONLY INPUT CANNOT BE THE BROADCAST. `f(vector) -> tile` is not a pointwise op at
    // all — it is a broadcast MATERIALIZATION with an `f` applied, and emitting it as one would put
    // the whole output's worth of work on a descriptor whose only input is one lane per row. Nothing
    // produces it, so it is named rather than given a meaning here.
    if ins.len() < 2 {
        return err(format!(
            "Elementwise({kind:?}) {name}: the only input of a unary op is marked broadcast, so the \
             output would be a SPRAY of a vector rather than a function of a tile. A broadcast is an \
             operand mode of an op that also reads a dense operand; a materialization is a different \
             op and this door does not invent one"
        ));
    }
    // EVERY OPERAND'S EXTENT AGAINST WHAT ITS FLAG CLAIMS — the dense ones at the output's extent
    // (the law [`pointwise_extents_agree`] states), the broadcast ones degenerate on the sprayed axis
    // and matching the output on the other. A `Col` operand must still have the output's ROW count:
    // it supplies one value per row, so a different row count would run off its end exactly as a
    // dense mismatch does.
    //
    // ⭐ THE ROW BASIS IS THE WINDOW'S when the store windows do not tile the view ([`pointwise_rows`])
    // — the epilogue `out / l[:, None]` of an unrolled flash block writes a `[64, 128]` window of a
    // `[1024, 128]` parameter view, and the dense operand beside the broadcast one is the `[64, 128]`
    // intermediate this block computed. The view is the row-blocked basis; the window is the grid
    // one; comparing a window's inputs against a view's extent is the mismatch that reads rows
    // nobody wrote.
    let rows = pointwise_rows(name, out)?;
    // ⭐⭐⭐ THE ROW-SPLIT (Y) MODE'S FRAME, DERIVED BEFORE THE WANT LOOP — because the want row
    // for a `Y` operand (`rows/y`) is only honest when `y` DIVIDES `rows`, and that divisibility
    // is itself half of the frame law: the output's own device frame `[rows, c_len]` must split
    // into exactly `y` sprays of `rows/y` rows with the `out` dim the ONE stick (the source is
    // single-stick). [`RegionFrame::split_rows`] checks all of it and returns the op's
    // `[mb, out, y]` facts; its `Err` is the refusal, so a row-split the bytes do not support is
    // a build failure here, not a wrong-byte bake (the mb re-row's own defect class).
    let ymode: Option<u32> = bcast.iter().find_map(|b| match b {
        Some(BcastAxis::Y(y)) => Some(*y),
        _ => None,
    });
    let split = match ymode {
        Some(y) => Some(
            crate::sdsc_abstract::RegionFrame::stick_blocked(
                rows as usize,
                out.c_len as usize,
                crate::superdsc_opspec::Df::Fp16,
            )
            .split_rows(y as usize, (rows / y) as usize)
            .map_err(|e| Error {
                message: format!("Elementwise({kind:?}) {name}: output t{}: {e}", out.tid),
            })?,
        ),
        None => None,
    };
    for (i, x) in ins.iter().enumerate() {
        let want = match bcast[i] {
            None => (rows, out.c_len),
            Some(BcastAxis::Col) => (rows, 1),
            Some(BcastAxis::Mb) => (1, out.c_len),
            // ⭐ THE ROW-SPLIT SOURCE IS THE `[mb, out]` TILE — one spray's worth of the output's
            // rows, at the output's own column extent. The divisibility that makes `rows/y`
            // honest is the frame law above, already proven.
            Some(BcastAxis::Y(y)) => (rows / y, out.c_len),
        };
        // ⭐⭐⭐ AND THE ROW-SPLIT SOURCE MAY BE A PER-TRIP WINDOW of a region-stated view — the
        // prefill mask: `[REGION·W4·mq, 64]` staged page-major, loaded per trip at its own row
        // corner, sprayed over the batch. The dense law's own pair ([`pointwise_extents_agree`]:
        // whole view OR window) stated for the spray: one spray's worth of rows is the WINDOW's
        // rows (`r_len`), and the corner rides the emission below (`In::y_at`) exactly as a
        // banded dense operand's row corner does — the source is single-stick by the frame law,
        // so a row corner is a plain element offset that composes with the mb-row addressing.
        // Every other mode keeps the verbatim equality.
        let ok = match bcast[i] {
            Some(BcastAxis::Y(_)) => {
                (x.v_rows, x.c_len) == want
                    || (x.c_start == 0
                        && x.r_len == want.0
                        && x.c_len == want.1
                        && x.r_start + x.r_len <= x.v_rows)
            }
            _ => (x.v_rows, x.c_len) == want,
        };
        if !ok {
            return err(format!(
                "Elementwise({kind:?}) {name}: input {i} t{} is `[{}, {}]` but its operand mode \
                 ({:?}) requires `[{}, {}]` against the `[{}, {}]` output t{}. A `Col` operand is one \
                 value per row (`[rows, 1]`), an `Mb` operand one row of values (`[1, cols]`), a `Y` \
                 operand one spray's worth of rows (`[rows/y, cols]`), and a \
                 dense operand the whole tile; the descriptor addresses it as its mode says, so an \
                 extent that disagrees reads bytes nobody wrote.",
                x.tid, x.v_rows, x.c_len, bcast[i], want.0, want.1, out.v_rows, out.c_len, out.tid,
            ));
        }
    }
    // ⭐ A `[rows, 1]` OUTPUT IS THE PER-ROW CHAIN at one stick, the same law the dense arm routes
    // through [`elementwise_per_row`]: the value rides in lane 0 of a one-stick output, which is the
    // width `attn.rs`'s own `lcorr`/`ladd` emissions state. A broadcast OPERAND of a per-row op is
    // still an operand mode (a spray is addressing, not width), so only the width is folded here.
    // The Y mode's width is the frame's `out` — the ONE stick the split law allows — which is the
    // same pad-up a `[rows, 1]` output takes: the mint's footprint already reserved it.
    let cols = if let Some(s) = split {
        s.out as u32
    } else if out.c_len == 1 {
        FP16_ELEMS_PER_STICK
    } else {
        out.c_len
    };
    check_pointwise_cols(cols, out.df, "Elementwise", out.tid)?;
    let op_name = format!("{op_func}_o{}", out.tid);
    // The handles: `rb` is RowBlocked, which is what `head_major == false` means on the seeded path —
    // the residual token stream, not the per-head attention layout. An INPUT handle's extents are
    // annotation-only (`In::ew` keeps the name and the two broadcast flags), and the OUTPUT's are too
    // (the builder reads `o.name()` and `O::kind()`), so the emission cannot depend on them; they are
    // stated at the operand's own shape so a reader is not misled.
    let out_h = rb(
        &out.name(),
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let in_names: Vec<String> = ins.iter().map(Region::name).collect();
    let in_handles: Vec<_> = in_names
        .iter()
        .zip(ins)
        .zip(bcast)
        .map(|((n, x), b)| match b {
            // ⭐ A `Col` OPERAND'S STORAGE IS STICK-WIDE, not `[rows, 1]`: the row reduction that
            // wrote it reserved `rows × 64` (`synth_footprint_bytes` rounds the inner extent up to
            // a whole stick — the `rinv` law this arm's own header cites), and the RedStick read
            // sweeps ONE STICK PER ROW. The operand mint checks that swept window against the
            // handle's declaration, so the annotation now states the footprint the storage always
            // had — `rb(name, rows, 64)`, the same handle `rmsnorm.rs` mints for `rinv`.
            Some(BcastAxis::Col) => rb(
                n,
                RowCount::of_token_rows(x.v_rows),
                BlockCols::of_one_stick(Lanes::FP16),
            ),
            // ⭐⭐⭐ A CORNERED `Mb` SOURCE IS DECLARED AT THE FULL VIEW, not the window. The
            // operand mint bounds-checks the swept window (one row of `c_len` values) AT the
            // operand's `col_offset` — the corner — against this declaration, so a `[1, c_len]`
            // handle would fail its own mint at any nonzero corner. Declaring `[v_rows, v_cols]`
            // states the buffer the window sits in, the same "true shapes" law `scalarmul_scaled`
            // states for its chunked tile (the walk's flat Mb frame presents the parent's whole
            // extent here, window and all).
            Some(BcastAxis::Mb) if x.c_start != 0 => rb(
                n,
                RowCount::of_token_rows(x.v_rows),
                BlockCols::of_feature_cols(x.v_cols),
            ),
            // ⭐⭐⭐ AND IN THE Y MODE EVERY OPERAND IS STICK-WIDE for the same reason: the op's
            // `out` dim IS the one stick the frame law allows, so the dense operand's storage is
            // `[v_rows, 64]` — the footprint its `(v_rows, c_len < 64)` mint already reserved —
            // and the swept window the mint checks (`rows × stick` dense, `rows/y × stick` for
            // the spray) fits it exactly.
            _ => rb(
                n,
                RowCount::of_token_rows(x.v_rows),
                BlockCols::of_feature_cols(if split.is_some() { cols } else { x.c_len }),
            ),
        })
        .collect();
    let ew: Vec<_> = in_handles
        .iter()
        .zip(ins)
        .zip(bcast)
        .map(|((h, x), b)| match b {
            None => Ok(In::full(h).ew()),
            Some(BcastAxis::Col) => Ok(In::col(h).ew()),
            // ⭐⭐⭐ THE Mb SOURCE AT A WHOLE-STICK COLUMN CORNER — the per-block slice law
            // (`In::mb_at`, the card-proven prefix-mask shape): one row of values read at the
            // corner of the parent's flat frame, broadcast over the output's rows. The corner is
            // built by `col_window_of`, the ONE typed constructor — it refuses a sub-stick
            // corner BY NAME (a mid-plane window is not a stick-major tensor of its own, #622's
            // law), so "where the slice starts" is never re-derived from an integer. A
            // corner-free source keeps the base `In::mb` — byte-identical to the whole-tensor
            // emission.
            Some(BcastAxis::Mb) if x.c_start != 0 => {
                let df = if x.is_fp8 { Df::Fp8 } else { Df::Fp16 };
                let w =
                    crate::addr::col_window_of(x.v_rows, x.v_cols, x.c_start, df).map_err(|e| {
                        Error {
                            message: format!("Elementwise({kind:?}) {name}: input t{}: {e}", x.tid),
                        }
                    })?;
                Ok(In::mb_at(h, w.dev()).ew())
            }
            Some(BcastAxis::Mb) => Ok(In::mb(h).ew()),
            // ⭐⭐⭐ THE Y SOURCE AT A PER-TRIP ROW WINDOW — the corner form of the row-split
            // source: the window's own device offset in the parent's stick-blocked storage
            // (`rc_of` at `(r_start, 0)`), the same typed corner law `In::mb_at` states for the
            // Mb source's column window. The source is single-stick by the frame law, so one
            // stick per row means the corner composes additively with the mb-row addressing and
            // the spray reads exactly the window's rows. A corner-free source keeps the base
            // `In::y_broadcast` — byte-identical to the whole-tensor emission.
            Some(BcastAxis::Y(y)) => Ok(if x.r_start == 0 {
                In::y_broadcast(h, *y).ew()
            } else {
                let df = if x.is_fp8 { Df::Fp8 } else { Df::Fp16 };
                In::y_at(
                    h,
                    *y,
                    crate::addr::rc_of(x.v_rows, x.v_cols, x.r_start, x.c_start, df),
                )
                .ew()
            }),
        })
        .collect::<Result<Vec<_>, Error>>()?;
    // The SAME `TileOp` the dense arm builds, so the two arms share one LX-tiling contract and a
    // broadcast op is not silently exempt from it. In the Y mode the tile states the OP's own
    // `[mb, out, y]` dims — `mb` one spray's worth of rows, `y` the spray count — which is the
    // iteration space the y-major TensorArgs (`["y", "mb", "out"]`) walk.
    let tile_op = pointwise_tile_op(
        match &split {
            Some(s) => s.mb as u32,
            None => rows,
        },
        cols,
        ymode.unwrap_or(1),
        ins.len() as u32 + 1,
        out.df,
    );
    Ok(vec![assemble_pointwise_broadcast_off_from_tile(
        &op_name,
        &tile_op,
        op_func,
        rows,
        cols,
        &ew,
        &out_h,
        // WHOLE-TENSOR OUTPUT: [`elementwise`] refuses any corner on the OUTPUT before routing
        // here (the one corner these arms can place — a whole-stick column window on an
        // Mb-broadcast SOURCE — rides the operand itself as `In::mb_at`), so the output's offset
        // is 0 by proof, not by default.
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    )])
}

/// main's `lower_silumul_node` (main 8466-8604): `out = silu(gate) · up` DECOMPOSED into two
/// pointwise ops — `silu(gate) → <out>_silu`, then `multiply(<out>_silu, up) → out`.
///
/// ⭐ THIS IS THE ONE BODY THE DEVICE'S OWN OP MAKES NECESSARY. The AIU has `OpFunc::Silu` and
/// KTIR's `math.*` set does not, so the producer writes silu longhand (negate → exp → 1+ → divide);
/// emitting that longhand would be five descriptors where the device has one primitive.
pub fn silumul(
    name: &str,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let (ins, o) = split_out(name, r, layout, 2)?;
    let (gate_r, up_r) = (ins[0], ins[1]);
    check_pointwise_cols(o.c_len, o.df, "SiluMul", o.tid)?;
    let rows = node_rows(name, &o)?;
    let cols = o.c_len;
    let gate = gate_r.name();
    let up = up_r.name();
    let out = o.name();
    let tmp_id = PlaceId::Act(o.tid).synth(SynthRole::Silu);
    let tmp = syn(layout, tmp_id);
    // ⛔ DECLARE BEFORE USE — `silu(gate) -> tmp` then `multiply(tmp, up) -> out`, so `tmp` is a
    // real intermediate. It was never declared, so it took the bump allocator's shared address
    // alongside rope's.
    //
    // ⛔ AT THE OUTPUT TENSOR'S WIDTH, NOT THIS CHUNK'S. `cols` is the column block this call
    // lowers; `tmp` is the WHOLE intermediate, shared by every chunk exactly as gate/up/out are.
    // See [`BundleLayout::synth_like`] — declaring `[rows, cols]` reserved granite-3.1-8b's first
    // chunk (8192 of 12800) and the second chunk then wrote 9216 B past it.
    if let Some(l) = layout {
        l.synth_like(tmp_id, o.tid, &[rows, cols], Df::Fp16);
    }
    let silu_name = format!("silu_o{}", o.tid);
    let mul_name = format!("mulsilu_o{}", o.tid);
    // COLUMN-BLOCK OFFSETS — the tape splits a wide SiluMul (granite MLP intermediate
    // 12800) into col-blocks that SHARE one output tensor; each chunk's region carries
    // its `cols.start` (`ds_name` collapses to `t{tid}`, so WITHOUT this every chunk
    // writes/reads the whole-tensor base 0 → later blocks' columns left as ZEROS).
    // `gate`/`up`/`out` all share the same chunk offset (verified: the tape aligns their
    // regions). See [`pointwise_chunk_out_offset`] + `silumul_chunks_cover_output`.
    let gate_off = pointwise_chunk_out_offset(gate_r.c_start);
    let up_off = pointwise_chunk_out_offset(up_r.c_start);
    let out_off = pointwise_chunk_out_offset(o.c_start);
    // The FULL tensor width, not this chunk's. `cols` is the chunk extent (the op's view); the
    // gate/up/tmp/out tensors are the whole intermediate, and a column corner must be taken against
    // the storage that actually holds it. Passing `cols` made the second chunk of granite-3.1-8b
    // (start 8192, width 4608 of a 12800-wide intermediate) address column 8192 of a 4608-wide
    // tensor — caught by the footprint guard. This is the same "op's view standing in for the
    // tensor's storage" the address module exists to stop, committed inside that module's own
    // migration.
    let full_cols = o.c_start + o.c_len;
    // ⭐ EACH CORNER IS A [`crate::addr::ColWindow`] (#622): the column pieces of one `[rows, full]`
    // stick-major tensor start on whole sticks (the tiler cuts at `nb = 8192`), so each piece is a
    // stick-major tensor of its own and the door presents it rank-2 — the work-division law then reaches
    // the remainder piece too (granite-3.1-8b's `[m, 4608]` at column 8192). Same element offsets as
    // `col_of`; a sub-stick corner is a build error naming the op.
    let window = |c: u32| {
        crate::addr::col_window_of(rows, full_cols, c, Df::Fp16).map_err(|e| Error {
            message: format!("SiluMul {name}: {e}"),
        })
    };
    let (gate_w, up_w, out_w) = (window(gate_off)?, window(up_off)?, window(out_off)?);
    // HANDLE-FLOW: bind gate/up/tmp/out once; `tmp` flows from the silu output INTO the mul input.
    // ⭐ THE HANDLES DECLARE THE **STORAGE**, like the corners above them: each tensor is the WHOLE
    // `[rows, v_cols]` intermediate (the region's own view — `full_cols` is the chunk-relative
    // width the corner arithmetic takes, which for a non-final chunk under-states the view). A
    // handle at the chunk's `[rows, cols]` makes chunk 1's own corner past-the-end, which the
    // operand mint refuses — the dims are annotation-only, so this changes no byte.
    let gate = rb(
        &gate,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(gate_r.v_cols),
    );
    let up = rb(
        &up,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(up_r.v_cols),
    );
    let tmp = rb(
        &tmp,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(o.v_cols),
    );
    let out = rb(
        &out,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(o.v_cols),
    );
    Ok(vec![
        assemble_pointwise_broadcast_off(
            PointwiseFacts {
                op_name: &silu_name,
                op_func: "silu",
                rows: crate::sdsc_abstract::RowCount::of_token_rows(rows),
                cols: crate::sdsc_abstract::BlockCols::of_feature_cols(cols),
                inputs: &[In::col_window(&gate, gate_w).ew()],
                o: &tmp,
                // The WRITE needs the same nest the READS above go through. A raw `cols_start` is the
                // flat element index; in the stick-blocked output, column `c` of a `[rows, cols]` tensor
                // starts at `(c/64)*(rows*64)`. Equal only at rows == 1, so decode was right and prefill
                // wrote every later column block on top of the first one.
                out_offset: out_w.into(),
            },
            sym_id_base,
            layout,
        ),
        assemble_pointwise_broadcast_off(
            PointwiseFacts {
                op_name: &mul_name,
                op_func: "multiply",
                rows: crate::sdsc_abstract::RowCount::of_token_rows(rows),
                cols: crate::sdsc_abstract::BlockCols::of_feature_cols(cols),
                inputs: &[
                    In::col_window(&tmp, out_w).ew(),
                    In::col_window(&up, up_w).ew(),
                ],
                o: &out,
                out_offset: out_w.into(),
            },
            sym_id_base,
            layout,
        ),
    ])
}

/// main's `lower_rmsnorm_node` (main 8604-8666): the RMS sequence IBM's `torch_spyre` decomposes
/// (`decompositions.py:999 spyre_rms_norm`) — assembled by `assemble_rmsnorm` with the device's
/// own fused `exx2` head (5 ops: the vendor's elementary `x·x`+`mean` pair is one reduce).
///
/// Its door: `eps` was an attribute of the node and is now the CARRIED
/// [`KtirNode::rmsnorm_eps_idx`] — its slot in `BundleLayout::scalarmul_scales`, resolved by the
/// producer where the node's value is in scope.
///
/// ⛔ NOT RECOVERED FROM A BOUND PARAMETER. It was, briefly: the program loaded its epsilon from the
/// reserved tid so this body could read the tid off a parameter. That made the PROGRAM's arithmetic
/// differ from `subtile→superdsc`'s — which states the epsilon inline — and the emulator's output
/// moved. The device's constant flow is unchanged either way (the descriptor reads the registry), so
/// the index rides on the node and the program keeps main's immediate.
pub fn rmsnorm(
    name: &str,
    k: &KtirNode,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    // x, gamma and the output — the parameters the rmsnorm program mints. Its constants are
    // immediates, so none of them is a parameter.
    let (tensors, out) = split_out(name, r, layout, 2)?;
    // ⭐⭐⭐ THE EPSILON COMES FROM THE PROGRAM. It was `KtirNode::rmsnorm_eps_idx`, a slot the producer
    // resolved and hung on the node, so the value the emulator adds and the value the descriptor reads
    // were two facts with nothing obliging them to agree — and a third-party producer had to hand over
    // an index into a registry it does not own. The program states the epsilon as an immediate
    // (`f32_splat(eps)` into the `mean + eps` add), which is the number the emulator actually uses; the
    // slot is then looked up BY THAT VALUE, exactly as the attention body looks up its multiplier.
    let eps = program_rmsnorm_eps(&k.func).ok_or_else(|| Error {
        message: format!(
            "RmsNorm {name}: the program states no epsilon. The rmsnorm program splats it into the \
             `arith.addf` that feeds its one root op — `math.sqrt` there, `math.rsqrt` in a producer \
             that spells `1/sqrt` as one op (`1/sqrt(mean + eps)`) — and the descriptor's `[1,1]` \
             const is resolved from that value, so a program without it cannot be lowered."
        ),
    })?;
    let eps_idx = scale_slot(layout, eps).ok_or_else(|| Error {
        message: format!(
            "RmsNorm {name}: epsilon {eps}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor adds it as a bound `[1,1]` const, so \
             the value the program uses must have a registry slot (registry desync)"
        ),
    })?;
    check_pointwise_cols(out.c_len, out.df, "RmsNorm", out.tid)?;
    let rows = node_rows(name, &out)?;
    let cols = out.c_len;
    // ⭐ THE GAIN OFFSET RIDES THE EPSILON'S LAW. `rmsnorm_oneplus_fwd` states the
    // offset as ONE `arith.addf` over the loaded gain row with a constant splat;
    // the door resolves the SAME value's `[1,1]` registry slot by bits (the census
    // collected `GainConvention::OnePlusScale.offset()` for exactly this node) and
    // `assemble_rmsnorm` emits ONE `add` ahead of the gain multiply — the same
    // formula the program runs, op for op. `None` is the Scale program (no addf),
    // and any value other than the convention's own 1.0 is refused by name: it is
    // not a convention `GainConvention::from_offset` admits, so no program may
    // state it.
    let ones_const = match program_gain_offset(&k.func) {
        None => None,
        Some((1.0, _)) => {
            let ones_idx = scale_slot(layout, 1.0).ok_or_else(|| Error {
                message: format!(
                    "RmsNorm {name}: the program states gain offset 1.0 (the OnePlusScale \
                     convention), which is absent from `BundleLayout::scalarmul_scales` — the \
                     descriptor adds it as a bound `[1,1]` const ahead of the gain multiply, so \
                     the value the program uses must have a registry slot (registry desync)"
                ),
            })?;
            Some(crate::place::act_name(scalarmul_scale_tid(ones_idx)))
        }
        Some((other, _)) => {
            return Err(Error {
                message: format!(
                    "RmsNorm {name}: the program states gain offset {other}, which is not a \
                     gain convention — `GainConvention::from_offset` admits 0.0 (Scale) and 1.0 \
                     (OnePlusScale) only, so a program stating any other offset is a producer bug"
                ),
            });
        }
    };
    // rms_norm_eps (config) flows via the scalarmul registry (compute_bundle_layout collected it) —
    // the same reserved `[1,1]` const `subtile→superdsc` adds to the mean-of-squares.
    let eps_const = crate::place::act_name(scalarmul_scale_tid(eps_idx));
    let x = tensors[0].name();
    let gamma = tensors[1].name();
    let t = out.tid;
    Ok(assemble_rmsnorm(
        RmsnormFacts {
            prefix: &format!("o{t}"),
            rows,
            cols,
            x: &x,
            gamma: &gamma,
            out_id: PlaceId::Act(t),
            eps_const: &eps_const,
            ones_const: ones_const.as_deref(),
        },
        sym_id_base,
        layout,
    ))
}

/// THE POINTWISE DEVICE-WIDTH PAD, CAPPED AT THE WIDTH THE OUTPUT'S PLACEMENT ACTUALLY HOLDS.
///
/// # THE SECOND INSTANCE OF A NAMED CLASS, NOT A ONE-OFF
///
/// [`DeviceWidth::for_pointwise`] pads a ScalarMul's column count to a stick count the gemm can split
/// across cores. That pad is width a SECOND PARTY has to make real: for the lm-head logits it IS real,
/// because the producer matmul emitted the padded width and the bundle layout reserved the padded
/// footprint. For an INTERMEDIATE that a producer mints at its logical 2-D shape, nobody reserved it —
/// so the descriptor addresses past the buffer.
///
/// This is exactly the defect `2c5abe0a5` fixed for the matmul's `out` width
/// ([`super::super::ir::bridge::tiled_op_sdsc_op::matmul::assemble`]'s `out_width_the_weight_holds`),
/// and W1's invented-width audit PREDICTED this site before any config reached it: "`for_pointwise` …
/// INVENTED, and WORSE than the matmul's: it is UNCONDITIONAL — no `macs >= 2^20` gate, so it pads
/// every ScalarMul. NOT REACHABLE TODAY … Same fix shape (cap at the OUTPUT placement's whole-stick
/// width) … the width is chosen at the read-only call site and I cannot construct a config that
/// reaches it, so I did not change it."
///
/// ⚠️ THE CONFIG THAT MEASURED IT IS NOT IN THIS BRANCH, AND THAT IS STATED RATHER THAN ELIDED. The
/// over-run was measured through the multi-op whole-function door, which is staged and lands with its
/// first caller: `decoder_layer_one_flat`'s `qk · QK_SCALE` writes a `[64, 64]` intermediate,
/// `for_pointwise` bumps 64 columns to 512 (eight sticks), and the emission asked for `0B + 65536B` of
/// a buffer declared `8192B` — caught by `resolve_seg_base`'s footprint guard as "would alias the next
/// intermediate". On the per-`Program` path this crate takes today the cap is a PROVEN NO-OP (see the
/// reproduction rule below), which is the whole reason it is safe to land ahead of that door.
///
/// # WHY HERE AND NOT IN THE ASSEMBLER
///
/// W1 named `assemble_pointwise_broadcast_off_from_tile` because the call site was read-only to it.
/// Both are ours now, and the width is CHOSEN here — it is baked into the `TileOp` two lines below, so
/// capping at the point of choice keeps the `TileOp` and the assembler in agreement by construction
/// rather than by two matching adjustments.
///
/// # ⛔ IT ONLY EVER DROPS A PAD IT CAN PROVE IS THIS RULE'S OWN
///
/// The same discipline that makes the matmul's version safe: the reserved width is re-padded and must
/// REPRODUCE the width handed in. So a genuine over-run — an emission wider than the placement for any
/// other reason — still reaches the footprint guard and is refused BY NAME, rather than quietly
/// satisfied by a narrower descriptor that computes fewer columns than the program states. And it is a
/// no-op wherever the layout does reserve the pad, which is the lm-head logits case the rule exists
/// for.
/// Returns the width PAIR a pointwise op over `[rows, ·]` may emit at:
///
/// * **the op width** — the floored device width, dropped to what the placement holds when the
///   excess is provably the floor's own pad (unchanged law); and
/// * **the storage width** — what the placement holds PER ROW (the whole-view width the tensors
///   are stored at, not any one chunk's), which is what the operand handles declare so the
///   operand mint checks every swept window against the storage that actually holds it.
///
/// # ⛔⛔⛔ A PLACEMENT SMALLER THAN THE PROGRAM'S OWN WIDTH IS AN `Err`, NEVER A CAP
///
/// `view_cols` is the program's own width — the region's view, the columns the program
/// states. When the narrowest placement among `names` holds LESS than that, the emission would
/// have to compute fewer columns than the program states to fit — the SILENT wrong answer this
/// function used to produce whenever its reproduction proof passed vacuously (the measured
/// case: a `[512, 512]` whole-view mask add over a validity-shaped `[1, 139776]` reservation —
/// `held = 279552/512` B/row → 273 elems → floor-stick 256, and `for_pointwise(256) == 512`
/// "reproduced" the asked width, so the add baked at `[512, 256]`, half the causal triangle
/// unswept and half the output unwritten while downstream max/sub/exp read all 512). Under the
/// Win ruling that is unreachable: the refusal names the tensor, its placement and the width
/// the program states, and the build fails instead of the descriptor lying.
///
/// The floor-drop band is unchanged and still legal: `view_cols ≤ held < cols` with the
/// reproduction proof (`for_pointwise(held) == cols`) drops only the pad this rule's own floor
/// added. A placement that holds AT LEAST the program's width but not the floor is the
/// measured decoder-intermediate case (a `[64, 64]` score tile whose 64 columns `for_pointwise`
/// bumps to the 8-stick occupancy floor) — the op emits at `held` and every byte is one the
/// program states.
fn pointwise_width_the_output_holds(
    layout: Option<&BundleLayout>,
    names: &[&str],
    rows: u32,
    view_cols: u32,
    cols: u32,
) -> Result<(u32, u32), Error> {
    let Some(l) = layout else {
        // No layout (every unit test; the gate probes): the op keeps its floored width and the
        // storage is the whole view at the same padding law.
        return Ok((cols, DeviceWidth::for_pointwise(view_cols).get()));
    };
    if rows == 0 {
        return Ok((cols, cols));
    }
    // ⭐ EVERY TENSOR THE OP ADDRESSES AT `cols`, NOT JUST THE OUTPUT. `cols` sizes the input view and
    // the output view alike, so the pad is only real if BOTH hold it — and MEASURED, the one that did
    // not was the INPUT: the guard fired on `synth 't32'` (the `qk` intermediate this op reads) while
    // the output `t33` was a separate buffer. Taking the narrowest is what makes the check about the
    // emission rather than about one operand of it.
    //
    // ⛔ AND IT READS THE SYNTH ALLOCATOR AS WELL AS `placements`. An INTERMEDIATE is declared through
    // `BundleLayout::synth`, which records its footprint in `SynthAlloc::sizes` and NOT in
    // `placements` — `resolve_seg_base`'s case 2, "the name is in the synth allocator's map". Reading
    // only `placements` found nothing for exactly the tensors this cap exists for, so the cap silently
    // did not fire; the first version of this function had that bug and a footprint guard caught it.
    let held_of = |name: &str| -> Option<u32> {
        let bytes = match l.id_of(name) {
            Some(crate::place::PlaceId::Act(tid)) => l
                .placements
                .get(&tid)
                .map(|p| p.size)
                .or_else(|| l.synth.borrow().sizes.get(name).copied()),
            _ => l.synth.borrow().sizes.get(name).copied(),
        }?;
        // A footprint is `synth_footprint_bytes`: the INNER (column) axis stick-rounded, the outer
        // multiplied. So one row costs a stick-rounded column count, and the width the buffer holds is
        // its size divided by the rows, in whole fp16 sticks.
        let row_bytes = (bytes / rows as u64).max(1);
        Some(((row_bytes / 2) as u32 / FP16_ELEMS_PER_STICK) * FP16_ELEMS_PER_STICK)
    };
    let Some(held) = names.iter().filter_map(|n| held_of(n)).min() else {
        return Ok((cols, cols));
    };
    // ⛔⛔⛔ THE PROGRAM'S OWN WIDTH IS THE FLOOR OF THE LEGAL — a placement that cannot hold it is a
    // reservation defect (the prefix-continuation mask's validity-shaped reservation under a
    // whole-view score mask), and satisfying it with a narrower descriptor computes fewer columns
    // than the program states. Refused BY NAME; never capped.
    if held > 0 && held < view_cols {
        let who = names
            .iter()
            .find(|n| held_of(n) == Some(held))
            .copied()
            .unwrap_or(names.first().copied().unwrap_or("the operand"));
        return err(format!(
            "a pointwise op over a [{rows}, {view_cols}] view whose narrowest placement '{who}' \
             holds only {held} columns per row — the placement cannot hold the width the program \
             itself states, so the only descriptor that fits would compute fewer columns than the \
             program. This is a reservation defect, not a pad to drop: reserve '{who}' at the width \
             the program states (or wider), or state the program's true width."
        ));
    }
    // THE PROOF: re-pad what the layout holds and require it to REPRODUCE the width we were about to
    // emit. Only then is the excess demonstrably this rule's own pad; anything else keeps `cols` and
    // the operand mint / the footprint guard refuse the genuine over-run BY NAME.
    if held > 0 && held < cols && DeviceWidth::for_pointwise(held).get() == cols {
        Ok((held, held))
    } else {
        Ok((cols, held.max(cols)))
    }
}

/// THE CONTROL SET FOR [`pointwise_width_the_output_holds`], the pointwise twin of
/// `out_width_the_weight_holds`'s.
///
/// ⛔⛔⛔ SAME DISCIPLINE, SAME REASON: a test that only pins "it narrows" is satisfied by a
/// function that narrows ALWAYS, which is the opposite silent wrong answer. Each case moves exactly
/// one thing:
///
/// | case | `cols` asked | footprint | holds | verdict | what it pins |
/// |---|---|---|---|---|---|
/// | (a) | 512 | 8192 B | 64 | caps to **64** | the measured over-run: the pad was fictional |
/// | (b) | 512 | 65536 B | 512 | keeps **512** | the pad was REAL — the lm-head logits case |
/// | (c) | 1024 | 65536 B | 512 | keeps **1024** | 512 does not re-pad to 1024, so it is a real over-run |
/// | (d) | 512 | out 65536 B / in 8192 B | 64 | caps to **64** | EVERY tensor the op addresses, not just the output |
///
/// (a)→(b) moves only the footprint and must flip the verdict. (c) is the case that pins the
/// reproduction proof specifically: [`DeviceWidth::for_pointwise`] of 512 is 512, which does not
/// reproduce the 1024 handed in, so the shortfall is a genuine over-run and must reach
/// `resolve_seg_base` rather than be quietly satisfied. Note (c) has to be taken at a WIDER `cols`
/// than (a) and (b): below 8 sticks every width re-pads to 512, so at `cols = 512` the reproduction
/// check cannot be made to fail — which is a real asymmetry with the matmul's version and the reason
/// this case looks different from `out_width_caps`' case (c).
///
/// ⭐ (d) IS THE BUG THIS FUNCTION ALREADY HAD ONCE, in two ways at once: the operand that did not
/// hold the pad was the INPUT (`synth 't32'`, the `qk` intermediate the op READS) while the output was
/// a separate, adequately sized buffer — so reading only the output finds nothing; and that input is
/// declared through [`BundleLayout::synth`], which records its footprint in `SynthAlloc::sizes` and
/// NOT in `placements` — so reading only `placements` also finds nothing. (d) fails if either the
/// `.min()` over all names or the synth-allocator fallback is removed.
///
/// ⚠️ WHAT (b) ACTUALLY DISCRIMINATES, MEASURED BY MUTATION RATHER THAN ASSUMED — the same finding
/// as `out_width_caps`'. (b) does NOT catch a cap that narrows unconditionally: at (b) the held width
/// EQUALS `cols`, so both branches return the same answer, and deleting the narrowing condition
/// leaves (b) green (it is (c) that goes red). What (b) catches is a `held` computed WRONG — proven by
/// mutating the two-bytes-per-element divisor to four, which turns (b) red. So (b) guards the
/// FOOTPRINT ARITHMETIC and (c) guards the narrowing CONDITION; neither substitutes for the other.
#[cfg(test)]
mod pointwise_width_caps {
    use super::*;
    use crate::place::{PlaceId, act_name};
    use crate::placement::{SegRole, TensorPlacement};

    /// The measured decoder intermediate: a `[64, 64]` score tile whose 64 columns `for_pointwise`
    /// bumps to the 8-stick occupancy floor.
    const ROWS: u32 = 64;

    /// A layout with one PLACED tensor `t1` of `size` bytes — the ordinary activation case.
    fn placed(size: u64) -> BundleLayout {
        let mut l = BundleLayout::default();
        l.ids.borrow_mut().insert(act_name(1), PlaceId::Act(1));
        l.placements.insert(
            1,
            TensorPlacement {
                tid: 1,
                role: SegRole::Intermediate,
                segment: 0,
                bank: 0,
                offset: 0,
                size,
                cover: crate::placement::BindCover::Full,
            },
        );
        l
    }

    /// The OP width the guard returns for a single placed operand of `size` bytes, a program
    /// view of `view` columns and a floored ask of `cols`.
    fn width_for(size: u64, view: u32, cols: u32) -> Result<u32, Error> {
        let l = placed(size);
        pointwise_width_the_output_holds(Some(&l), &[&act_name(1)], ROWS, view, cols)
            .map(|(op, _)| op)
    }

    /// ⛔ CASE (a) — THE MEASURED OVER-RUN. `qk · QK_SCALE` writes a `[64, 64]` tile;
    /// `for_pointwise` bumps 64 columns to 512, and the buffer is the 8192 B the logical shape needs.
    /// The emission asked for `0B + 65536B` of it.
    #[test]
    fn an_under_reserved_intermediate_caps_the_pointwise_width() {
        assert_eq!(
            width_for(64 * 64 * 2, 64, 512).expect("the floor pad drops"),
            64,
            "a `[64, 64]` tile holds 64 columns; emitting 512 is the measured \
             `0B + 65536B exceeds footprint 8192B` refusal — the floor's own pad, dropped"
        );
    }

    /// ⭐ CASE (b) — THE CONTROL THAT MAKES (a) NON-VACUOUS, and the case the rule exists FOR: the
    /// padded lm-head logits, where the producer matmul emitted the padded width and the layout
    /// reserved it. The cap must be a no-op.
    #[test]
    fn a_fully_reserved_tensor_keeps_the_padded_width() {
        assert_eq!(
            width_for(u64::from(ROWS) * 512 * 2, 512, 512).expect("the pad is reserved"),
            512,
            "when the layout reserves the padded width the pad is REAL and nothing may move"
        );
    }

    /// ⛔ CASE (c) — A PLACEMENT SMALLER THAN THE PROGRAM'S OWN WIDTH IS REFUSED HERE, BY NAME.
    /// The program states 1024 columns and the placement holds 512: the only descriptor that
    /// fits would compute fewer columns than the program states — the silent wrong answer this
    /// cap used to reach the footprint guard with (and, when the reproduction proof passed
    /// vacuously, to bake — the `[512, 512]` mask add capped to `[512, 256]`). Under the Win
    /// ruling the refusal happens at the width choice itself.
    #[test]
    fn a_shortfall_below_the_programs_own_width_is_refused_by_name() {
        let Err(e) = width_for(u64::from(ROWS) * 512 * 2, 1024, 1024) else {
            panic!(
                "a placement holding 512 of a 1024-column program was ACCEPTED — the only \
                    descriptor that fits computes half the columns the program states"
            )
        };
        assert!(
            e.message.contains("reservation defect") && e.message.contains("[64, 1024]"),
            "the refusal names the defect and the shape: {}",
            e.message
        );
    }

    /// ⭐ CASE (d) — THE NARROWEST OPERAND WINS, AND THE SYNTH ALLOCATOR IS READ. The output `t1` is
    /// adequately sized; the INPUT is a synthetic intermediate declared only in `SynthAlloc::sizes`
    /// and holding one stick. `cols` sizes both views, so the pad is only real if BOTH hold it.
    #[test]
    fn the_narrowest_operand_wins_including_one_only_the_synth_allocator_knows() {
        let l = placed(u64::from(ROWS) * 512 * 2);
        let narrow = "t32_synth";
        l.synth
            .borrow_mut()
            .sizes
            .insert(narrow.to_string(), 64 * 64 * 2);
        let out = act_name(1);
        assert_eq!(
            pointwise_width_the_output_holds(Some(&l), &[&out, narrow], ROWS, 64, 512)
                .expect("the narrow input's floor pad drops")
                .0,
            64,
            "the INPUT is the operand that does not hold the pad, and it lives in the synth \
             allocator rather than in `placements` — both halves of the bug this once had"
        );
        assert_eq!(
            pointwise_width_the_output_holds(Some(&l), &[&out], ROWS, 64, 512)
                .expect("the output alone holds the pad")
                .0,
            512,
            "reading ONLY the output finds nothing wrong, which is why the minimum is taken over \
             every tensor the op addresses"
        );
    }

    /// ⛔⛔⛔ THE VACUOUS PROOF — the measured silent cap this function's `Err` exists to make
    /// unreachable. A `[512, 512]` whole-view mask add over a validity-shaped `[1, 139776]`
    /// reservation: `held = 279552/512` B/row → 273 elems → floor-stick 256, and
    /// `for_pointwise(256) == 512` "reproduced" the asked width — so the old cap narrowed the add
    /// to `[512, 256]`, sweeping half the causal triangle and leaving half the output unwritten
    /// while downstream max/sub/exp read all 512. The proof passing is EXACTLY what must not
    /// satisfy a placement smaller than the program's own 512-column view.
    #[test]
    fn a_vacuously_reproducing_reservation_is_refused_not_capped() {
        // The pod's own numbers: 279552 B over 512 rows → 273 elems/row → floor-stick 256.
        let l = placed(279552);
        let Err(e) = pointwise_width_the_output_holds(
            Some(&l),
            &[&act_name(1)],
            512,
            512,
            DeviceWidth::for_pointwise(512).get(),
        ) else {
            panic!(
                "the vacuous reproduction was ACCEPTED — the add bakes at [512, 256] and half \
                    the causal triangle is never swept"
            )
        };
        assert!(
            e.message.contains("reservation defect"),
            "the refusal names the class: {}",
            e.message
        );
    }
}

/// main's `lower_scalarmul_node` (main 10376-10441): an on-device pointwise `mul` by the bound
/// `[1,1]` scale const.
///
/// Its door: the scale was an `f32` on the node and is now the CARRIED
/// [`KtirNode::scalarmul_scale_idx`] — its slot in `BundleLayout::scalarmul_scales`.
///
/// ⛔ NOT A PARAMETER SCAN. It was: the program loaded its scale from the reserved tid so this body
/// could find the tid among the parameters. `subtile→superdsc`'s own KTIR states the value inline
/// (`st.splat(f64::from(scale), dims)`), and a bound load in its place is a `TensorExtract` the
/// emulator's Metal offload cannot treat as a resident tile — MEASURED as 51 of 216 `map window`
/// refusals, every one of them falling back to the interpreter. The device is unaffected either way:
/// the descriptor below still multiplies by the bound `[1,1]` const at this slot.
pub fn scalarmul(
    name: &str,
    k: &KtirNode,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    // ⭐⭐⭐ THE MULTIPLIER COMES FROM THE PROGRAM, like the epsilon and the attention scale. It was
    // `KtirNode::scalarmul_scale_idx`, a slot the producer resolved and hung on the node — so the
    // number the emulator multiplies by and the number the descriptor multiplies by were two facts
    // with nothing obliging them to agree, and a third-party producer had to hand over an index into a
    // registry it does not own. The program states the value inline; the slot is looked up by it.
    let scale = program_scalarmul_scale(&k.func).ok_or_else(|| Error {
        message: format!(
            "ScalarMul {name}: the program states no single multiplier. Its scalarmul arm splats the \
             scale and multiplies — once for the region, or once per row block when the region is too \
             wide for the LX — so every `arith.mulf` here must multiply by one splat and all of them \
             by the same value."
        ),
    })?;
    // ⭐ AND WHETHER THE TILE IT MULTIPLIES IS **GATHERED** is the program's statement, not the
    // caller's. `Program::ScalarMul` is the kind of the one COMPUTE op; a Triton embedding is exactly
    // that op over a row tile whose row INDEX is data (`embedding.py`'s
    // `rows = table_desc.gather(ids, 0)` then `rows * EMB_SCALE`), so the gather is addressing that
    // rides on this same node rather than a different node kind.
    //
    // `None` for every program that states no `ktdp.construct_indirect_access_tile`.
    scalarmul_at(name, scale, gather_of(k)?, r, sym_id_base, layout)
}

/// A KV HEAD INDEX, TYPED, from a loop counter — the one place a bare `usize` becomes a [`KvHead`].
///
/// A kv head, a query head, a slot and a feature are all small integers, and nothing but spelling kept
/// them apart at a call site. `?` here means a head past the pool's count is a build error rather than
/// an address in the next plane.
///
/// [`KvHead`]: crate::sdsc_abstract::KvHead
fn kv_head_of(kvh: u32, nkvh: u32) -> Result<crate::sdsc_abstract::KvHead, Error> {
    let n = std::num::NonZeroU32::new(nkvh).ok_or_else(|| Error {
        message: "a paged pool needs at least one kv head".into(),
    })?;
    crate::sdsc_abstract::KvHead::new(kvh, n).ok_or_else(|| Error {
        message: format!("kv head {kvh} is past the pool's {nkvh} head(s)"),
    })
}

/// main's `lower_attn_node` (main 9528-10376), waiting for its head geometry to become consts — the
/// consumer side of `with_config_attn_geometry`, exactly as main's `LowerAttn` was.
///
/// ⛔⛔⛔ AND IT IS MAIN'S `LowerAttn` FIELD FOR FIELD, WHICH IS WHY `AttnFacts` IS GONE. main's pack
/// is `{ node, cap, active_cap, rows_are_requests, sym_id_base, layout }`: the NODE, plus the four
/// things the node does not state. This pack replaces `node` with the node's PROGRAM — from which
/// [`attn_operands`] reads the six tensor identities and the resident capacity that main read off
/// `node.inputs` / `node.op` / `ir.tensors` — and keeps exactly main's remainder. `scale` joins them
/// because main took it off `node.op`'s `SubOp::AttnDecode { scale }` payload, the same place it took
/// the geometry the door above turns into the consts.
///
/// ⛔ NOTHING HERE IS STAPLED TO THE PROGRAM. These are ARGUMENTS, supplied per call by the caller
/// that crossed the geometry door — a third-party KTIR producer's caller states them from its own
/// configs, exactly as scratchy's does. A record hanging off [`KtirNode`] is what made the crate's
/// attention path unusable to anyone but this producer.
pub struct AttnAt<'a> {
    pub name: &'a str,
    /// The node's KTIR, and its parameters as [`Region`]s — the program, and nothing beside it.
    pub k: &'a KtirNode,
    pub r: &'a [Region],
    /// main's `LowerAttn::rows_are_requests` — TRUE when this bundle's query rows are separate
    /// requests (a batched decode) rather than consecutive positions of one sequence.
    ///
    /// ⛔ THE ONE FACT WITH NO STATEMENT ANYWHERE IN THE PROGRAM, and it cannot have one: it decides
    /// the DEVICE's row/slot laws (`attn_bundle_rows`' pad, rope's head-major collapse), which the
    /// emulator does not model at all — the attention/rope programs never receive it and
    /// emit the same ops either way.
    pub rows_are_requests: bool,
    /// Whether this bundle's attention reads the paged KV through an index — the caller-side twin of
    /// the spyre door's `BundleAttnParams::gathers_kv` (the batched rungs' per-page gather, or the
    /// one-request rung's split-KV chunk). Decides the `KV_BLOCK_INDEX_TID` admission below; the
    /// placement reserves under the SAME fact — one predicate, both halves of the gather.
    pub gathers_kv: bool,
    /// The split-KV chunk's page count — the caller-side twin of `BundleAttnParams::chunk_pages`.
    /// `Some` lowers the prefix through [`crate::sdsc_abstract::ChunkScratch`] (every resident page in
    /// one pass); `None` keeps the per-page fold.
    pub chunk_pages: Option<u32>,
    /// ⭐ WHERE THIS BUNDLE'S ATTENTION SCORE SCALE LIVES — the caller-side twin of the spyre
    /// door's `BundleAttnParams::scale_law` ([`ScoreScaleLaw`], the proof-carrying replacement
    /// for the `scale_in_weights: bool` this pack used to carry). The fold removes the score
    /// mulf from the program, so a folded program states NO multiplier; a model whose attention
    /// scale genuinely IS 1.0 states none either (the triton→ktir pipeline folds a multiply by
    /// 1.0 away). The law is what tells the door that absence is one of those two and not a
    /// broken program — and its arms are matched exhaustively below, so the two skip reasons and
    /// the multiply case cannot be conflated the way a `bool`'s single true could.
    pub scale_law: ScoreScaleLaw,
    /// ⭐ WHETHER THE FINALIZE HANDS `out` TO o_proj SLAB-MAJOR — the caller-side twin of the spyre
    /// door's `BundleAttnParams::headmajor_handoff`, and the SAME fact the matmul door swaps the o
    /// matmul's B operand on (the permuted `oproj_headmajor_tid` copy): one fact, both halves of the
    /// restructure, so neither can be emitted without the other. TRUE only for the single-row
    /// (`mq == 1`) body of a multi-slab head dim with an fp8 W8A8 o matmul.
    pub headmajor_handoff: bool,
    pub sym_id_base: &'a mut i64,
    pub layout: Option<&'a BundleLayout>,
}

/// ⭐⭐⭐ THE ATTENTION NODE'S OPERANDS AND ITS RESIDENT CAPACITY, READ OUT OF THE PROGRAM THAT
/// STATES THEM — the seven fields `AttnFacts` used to carry from the producer.
///
/// Every one is a positional join on [`KtirNode::args`], which is the join [`regions`] already
/// performs: the attention program mints its parameters in first-use order — `q`, `out`,
/// then (when this node emits the runtime length mask) the mask, then EVERY segment's K view followed
/// by EVERY segment's V view — and `crate::place::act_name(tid)` is the operand spelling every proven
/// builder agrees on.
///
/// ⭐ THE SIX IDENTITIES ARE [`PlaceId`](crate::place::PlaceId)S, the bake↔host join key — not bare
/// `u32`s. A number that is NOT a placed tensor's identity (a slot offset, a row index, a cursor)
/// cannot be handed where one is expected, and the number exits only through `.tid()` at the
/// binding that spends it.
///
/// ⛔ AND THE POSITIONS ARE CROSS-CHECKED AGAINST THE SHAPES THEY MUST HAVE, because a silent
/// positional slip REPOINTS A TENSOR: `q`/`out` are `[mq, nqh·hd]` and every K/V view is `[·, nkvh·hd]`,
/// so a q↔kv swap is caught; the two views of one SEGMENT share a row extent while a segment's K and
/// the NEXT segment's V do not, so an all-Ks-then-all-Vs order mistaken for an interleaved one is
/// caught. A mismatch is a build error naming what disagreed, never a descriptor built on a guess.
#[derive(Clone, Copy, Debug)]
pub struct AttnOperands {
    /// main's `inputs[0]` — the rotated Q.
    pub q_id: crate::place::PlaceId,
    /// main's `output`.
    pub out_id: crate::place::PlaceId,
    /// main's `kv.cache_tensor()` / `kv.v_cache_tensor()` — the resident prefix cache, segment 0.
    pub k_id: crate::place::PlaceId,
    pub v_id: crate::place::PlaceId,
    /// main's `inputs[3]` / `inputs[4]` — this step's new K and V, the last segment.
    pub new_k_id: crate::place::PlaceId,
    pub new_v_id: crate::place::PlaceId,
    /// The resident cache's structural capacity — the prefix K view's OWN row extent, which is
    /// `ir.tensors[kv.cache_tensor()].rows`, i.e. exactly what main's call site looked up. A
    /// [`SlotCount`](crate::sdsc_abstract::SlotCount): the capacity is a number of KV slots, the
    /// same unit the extent pair below spends it as.
    pub cap: crate::sdsc_abstract::SlotCount,
    /// Which PARAMETER the resident K is, so the sweep check below reads the rows of that buffer
    /// without searching for it a second time.
    pub k_param: usize,
    /// The query row count, read off `q`'s own view — not a fact the caller states. The
    /// one-mint count the scratch doors take, so it cannot transpose against the tid fields.
    pub mq: crate::sdsc_abstract::QueryRowCount,
}

/// [`AttnOperands`], with every position proven against the shape it must have.
///
/// `geom` is the witness the door minted, so the widths checked against are the model's, evaluated
/// once — and the head dim is checked against the program's own per-head window, which is the first
/// `ktdp.construct_access_tile` over the `q` view.
pub fn attn_operands<const NQH: u32, const NKVH: u32, const HD: u32>(
    name: &str,
    k: &KtirNode,
    r: &[Region],
    geom: crate::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
) -> Result<AttnOperands, Error> {
    let (q_width, kv_width) = (geom.nqh() * geom.hd(), geom.nkvh() * geom.hd());
    // `q`, then `out`: the attention program's first two parameter mints, in that order.
    let (q, out) = match (r.first(), r.get(1)) {
        (Some(q), Some(o)) => (q, o),
        _ => {
            return err(format!(
                "{name}: an attention program states {} parameter(s); its construction mints at \
                 least six (q, out, and each segment's K then V)",
                r.len()
            ));
        }
    };
    // ⭐⭐⭐ THE QUERY ROW COUNT COMES FROM THE PROGRAM. It was `KtirNode::out_shape.0`, and both
    // checks below forced it equal to this very view's row extent — so the record was a copy of what
    // `q`'s own `ktdp.construct_memory_view` states, and a third-party producer had to supply a
    // number its program already carries. The `out` check below is unchanged and still real: `q` and
    // `out` must agree on rows, and both on the geometry's column width.
    let mq = q.v_rows;
    let shape_is = |role: &str, x: &Region, rows: u32, cols: u32| -> Result<(), Error> {
        if (x.v_rows, x.v_cols) != (rows, cols) {
            return err(format!(
                "{name}: parameter {role} t{} states a `[{}, {}]` view where the node's geometry \
                 (nqh={NQH} nkvh={NKVH} head_dim={HD}) and its {mq} query row(s) make it \
                 `[{rows}, {cols}]` — the positional join on `KtirNode::args` disagrees with the \
                 shape the operand must have, which means it names a different tensor",
                x.tid, x.v_rows, x.v_cols,
            ));
        }
        Ok(())
    };
    shape_is("q", q, mq, q_width)?;
    shape_is("out", out, mq, q_width)?;
    // ⛔ THE STORE SAYS WHICH ONE IS THE OUTPUT, so a q↔out transposition cannot survive even when
    // (as here) the two share a shape.
    if q.is_out || !out.is_out {
        return err(format!(
            "{name}: parameter 0 (t{}, read as `q`) is {}written by a `ktdp.store` and parameter 1 \
             (t{}, taken as `out`) is {}— `q` is the read operand and `out` the written one, so this \
             program's parameters are not the order the attention program mints them in",
            q.tid,
            if q.is_out { "" } else { "not " },
            out.tid,
            if out.is_out { "" } else { "not " },
        ));
    }
    // ⭐ THE HEAD DIM, FROM THE PROGRAM'S OWN PER-HEAD WINDOW. The attention program reads head `h` as
    // `tile(q_view, ·, h·hd, ·, hd)`, so the first access tile over the `q` view is `hd` wide — the
    // one place the program states the head dim rather than a product with it.
    //
    // ⛔ AND IT IS THE TILE'S OWN `Shape`, NOT [`Region::c_len`]. That field is dropped to the whole
    // view unless BOTH of the tile's corner operands are `arith.constant`s, and a PREFILL attention
    // runs one head per core — its column corner is `arith.muli(pid, hd)`, so `c_len` reads back the
    // full `nqh·hd` and a check against it refused every prefill bundle (MEASURED: "the program's
    // first window on `q` t341 is 576 column(s) wide where the door's head dim is 64"). The `Shape`
    // says `hd` either way.
    let q_window_cols = param_first_tile(&k.func, &k.func.arguments[0].0).map(|(_, c)| c);
    if q_window_cols != Some(geom.hd()) {
        return err(format!(
            "{name}: the program's first window on `q` t{} is {} where the door's head dim is {} — \
             the geometry crossed does not belong to this program",
            q.tid,
            match q_window_cols {
                Some(c) => format!("{c} column(s) wide"),
                None => "not a window at all".to_string(),
            },
            geom.hd(),
        ));
    }
    // The K/V segment parameters: everything after `q`/`out` except the runtime length mask, whose
    // tensor id the node already states (it is SYNTHETIC — beyond the graph — so nothing else could).
    let mask_tid = k.mask.map(|b| b.get());
    let segs: Vec<(usize, &Region)> = r
        .iter()
        .enumerate()
        .skip(2)
        .filter(|(_, x)| Some(x.tid) != mask_tid)
        .collect();
    // main's own arity: `AttnDecode` "expects 5 inputs [q, prefix_k, prefix_v, new_k, new_v]", so
    // exactly two segments, and the program states each one's K view before any V view.
    let [(k_param, kc), (_, new_k), (_, vc), (_, new_v)] = match segs.as_slice() {
        [a, b, c, d] => [*a, *b, *c, *d],
        _ => {
            return err(format!(
                "{name}: the program takes {} K/V segment parameter(s); an attention node has \
                 exactly four (the resident prefix cache's K and V, and this step's new K and V — \
                 main's `inputs[1..5]`)",
                segs.len()
            ));
        }
    };
    for (role, x) in [
        ("resident K", kc),
        ("new K", new_k),
        ("resident V", vc),
        ("new V", new_v),
    ] {
        if x.v_cols != kv_width {
            return err(format!(
                "{name}: parameter {role} t{} states a {}-column view where the node's geometry \
                 (nqh={NQH} nkvh={NKVH} head_dim={HD}) makes a kv stream {kv_width} wide — the \
                 positional join names a different tensor",
                x.tid, x.v_cols,
            ));
        }
    }
    // ⛔ THE TWO VIEWS OF ONE SEGMENT SHARE A ROW EXTENT. This is what proves the order is
    // all-Ks-then-all-Vs and not K,V,K,V: under the wrong reading `vc` would be the PREFIX's V (the
    // full capacity) while `new_k` would be it too, and the resident/new row extents differ.
    if kc.v_rows != vc.v_rows || new_k.v_rows != new_v.v_rows {
        return err(format!(
            "{name}: the resident cache's K t{} `[{}, ·]` and V t{} `[{}, ·]`, and this step's new K \
             t{} `[{}, ·]` and V t{} `[{}, ·]`, must each be one segment's pair and so share a row \
             extent — they do not, so the K and V parameter lists are not the order the \
             construction mints them in",
            kc.tid, kc.v_rows, vc.tid, vc.v_rows, new_k.tid, new_k.v_rows, new_v.tid, new_v.v_rows,
        ));
    }
    Ok(AttnOperands {
        q_id: crate::place::PlaceId::Act(q.tid),
        out_id: crate::place::PlaceId::Act(out.tid),
        k_id: crate::place::PlaceId::Act(kc.tid),
        v_id: crate::place::PlaceId::Act(vc.tid),
        new_k_id: crate::place::PlaceId::Act(new_k.tid),
        new_v_id: crate::place::PlaceId::Act(new_v.tid),
        cap: crate::sdsc_abstract::SlotCount::new(kc.v_rows),
        k_param,
        mq: crate::sdsc_abstract::QueryRowCount::of_mq(mq),
    })
}

/// ⛔⛔⛔ AN ATTENTION PROGRAM'S SIZE MUST NOT GROW WITH ITS ROWS — the bake refuses a per-row unroll.
///
/// The continuation arm was a `tl.static_range(0, MQ)` of per-row bodies, each with its own `[qi + 1,
/// HD]` causal tiles: `MQ` bodies whose tiles summed to `MQ²/2` rows. Its program, and every pass that
/// walks it, grew with the rows: MEASURED on granite-2b, 94.7 s to lower at 256 rows (the bake's
/// widest compile) and a build-script thread past 25 min at 102 GB without finishing at 512. The bundle
/// was correct; only the build paid.
///
/// So the check is on the LOWERED program, at the door, for every multi-row attention: the number of
/// windows it takes of its query `q` (parameter 0), regions included so a loop the front end failed to
/// fold still counts its body. A whole-chunk head reads its query ONCE, so the program takes at most
/// one window per head (`heads` — the grid is unrolled into the program, so every head's body is in
/// it); a per-row form takes one per head PER ROW. That is the property a raw op-count bound cannot
/// state, because a program's absolute size also scales with its heads and head dim. Decode
/// (`mq == 1`) has one row, so the two forms coincide there.
pub fn attn_program_is_flat_in_rows(
    name: &str,
    f: &IRFunction<'static>,
    mq: u32,
    heads: u32,
) -> Result<(), Error> {
    if mq <= 1 {
        return Ok(());
    }
    let Some((q, _)) = f.arguments.first() else {
        return err(format!(
            "{name}: an attention program with no parameters — its query is parameter 0"
        ));
    };
    let tiles = param_tiles_deep(f, q);
    if tiles > heads as usize {
        return err(format!(
            "{name}: a {mq}-row attention program over {heads} head(s) takes {tiles} windows of its \
             query, where a whole-chunk head takes one — it unrolls per row, so its size, its \
             lowering and every pass over it grow with its rows. State the chunk whole: one score tile \
             per segment and the causal triangle in the mask."
        ));
    }
    Ok(())
}

/// The rows of one parameter's buffer the program actually READS — every `ktdp.construct_access_tile`
/// over any view of it, as the widest row extent; `0` when it takes no window at all.
///
/// ⭐ THIS IS WHERE `active_cap` MEETS THE PROGRAM. A bundle baked at `ActiveCap::NONE` sweeps no
/// resident prefix, and its program says so by naming the cache tensor (so the identity survives) and
/// taking no access tile of it. `nb == 0` on the device and "no prefix block" in the emulator are then
/// the same statement rather than two.
///
/// ⛔ [`regions`] CANNOT ANSWER THIS. Its window fields come from the FIRST access tile whose corner
/// is a pair of `arith.constant`s, and attention's kv corner is `arith.addi(col_start, kv_head·hd)` —
/// so that field falls back to the whole view, which is the capacity and not the sweep.
fn param_read_rows(f: &IRFunction<'static>, ptr: &Ssa) -> u32 {
    param_tiles(f, ptr)
        .filter_map(|t| shape_2d(t).map(|(rows, _)| rows))
        .max()
        .unwrap_or(0)
}

/// The `[rows, cols]` of the FIRST window the program takes of one parameter's buffer — its access
/// tile's own `Shape`, which is stated whatever its corner is made of.
/// PUBLIC for the door: the geometry mint reads a program's head dim here, the same way rope's
/// head dim is read — the door (`ktir_superdsc_door::attn`) mints `ModelAttnGeometry` off the
/// program's own views because a hybrid model's bundle carries more than one.
pub fn param_first_tile(f: &IRFunction<'static>, ptr: &Ssa) -> Option<(u32, u32)> {
    param_tiles(f, ptr).find_map(shape_2d)
}

/// Every `ktdp.construct_access_tile` over any view of one parameter, in program order.
fn param_tiles<'f>(
    f: &'f IRFunction<'static>,
    ptr: &Ssa,
) -> impl Iterator<Item = &'f Operation<'static>> {
    let views: Vec<Ssa> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(ptr))
        .filter_map(|o| o.result)
        .collect();
    f.operations.iter().filter(move |o| {
        o.op_type == OpKind::KtdpConstructAccessTile
            && o.operands.first().is_some_and(|v| views.contains(v))
    })
}

/// [`param_tiles`] over `ops_deep()` — REGIONS INCLUDED.
///
/// ⭐ IT EXISTS ONLY TO BE COMPARED WITH `param_tiles`, in [`regions`]. Nothing here reads a nested
/// tile's extents, because a body that spans one descriptor over a time-tiled program would be
/// describing the trip and calling it the node. The comparison turns "no window at the top level"
/// into a refusal that says WHY, instead of the silent whole-view fallback.
fn param_tiles_deep(f: &IRFunction<'static>, ptr: &Ssa) -> usize {
    let views: Vec<Ssa> = f
        .ops_deep()
        .into_iter()
        .filter(|o| o.op_type == OpKind::KtdpConstructMemoryView && o.operands.first() == Some(ptr))
        .filter_map(|o| o.result)
        .collect();
    f.ops_deep()
        .into_iter()
        .filter(|o| {
            o.op_type == OpKind::KtdpConstructAccessTile
                && o.operands.first().is_some_and(|v| views.contains(v))
        })
        .count()
}

/// The attention multiplier the PROGRAM states, as an immediate: the `arith.constant` splatted into
/// the `arith.mulf` that scales a `linalg.matmul`'s scores.
///
/// ⭐ ONE SITE, RECOGNISED BY ITS OPERANDS, not by elimination among the func's float constants —
/// the attention program also states `-1e38` and `0.0`, and telling them apart by value would be a
/// convention rather than a reading.
/// The RMSNORM epsilon THE PROGRAM STATES, read structurally.
///
/// ⭐ RECOGNISED BY ITS SHAPE, NOT ITS VALUE. The rmsnorm program computes `1/sqrt(mean + eps)`, so
/// the epsilon is the splat operand of the `arith.addf` that feeds the one `math.sqrt`. The same body
/// also splats `0.0` (the reduce seed), `1.0` (the reciprocal's numerator) and the reciprocal row
/// count, so picking a float out by value would be a convention rather than a reading — the same
/// discipline [`program_score_scale`] follows.
///
/// `f32_splat` emits `arith.constant` (scalar) then `tensor.splat`, so the value is one hop behind the
/// operand.
///
/// ⭐ AND THE ROOT IS EITHER `math.sqrt` OR `math.rsqrt`, because a third-party producer states
/// `1/sqrt(x)` as ONE op. The builder's `KtirFunc::rms_norm` spells it `math.sqrt` then `arith.divf(1.0, ·)`; a
/// Triton front end spells the same value `math.rsqrt`, which `OpKind::MathRsqrt` already admits and
/// which this crate already emits on card as a first-class pointwise (`Elementwise::Rsqrt =>
/// ("rsqrt", 1)`). The epsilon is the SAME fact under both spellings — the splat into the add that
/// feeds the root — so the reading takes both roots and stays exactly as structural: one root op in
/// the whole program, exactly one splat among the two operands of the add that feeds it.
///
/// ⭐⭐⭐ AND THE DESCRIPTOR THIS PROGRAM LOWERS TO ALREADY SPELLS IT `rsqrt`. Step 4 of
/// [`assemble_rmsnorm`] is
/// `pw1("rmrsqrt_…", "rsqrt", …)`, and its own comment says "ONE native `rsqrt` (torch.rsqrt), NOT
/// sqrt+reciprocal". So a program that states `math.rsqrt` matches what this crate EMITS more
/// closely than the builder's `math.sqrt` + `arith.divf` does, and the reader was the only thing in
/// the path narrower than both. (Step 2 lines up the same way: it is one native `mean` reduce that
/// "folds 1/N into the reduce scale", which is exactly a producer that multiplies by a splatted
/// `1/cols` rather than dividing — see the mean's operand below.)
///
/// ⛔ THE ALTERNATIVE WAS TO NORMALISE `math.rsqrt` AWAY IN THE PRODUCER, AND IT IS UNSOUND HERE.
/// [`elementwise`] takes its kind as an ARGUMENT and never reads the program's op, so rewriting every
/// `math.rsqrt` into `math.sqrt` + a reciprocal would leave an `Elementwise(Rsqrt)` node whose program
/// says "sqrt then divide" while its descriptor computes `rsqrt` — the program/descriptor divergence
/// the epsilon door itself exists to prevent. Rewriting only inside an rmsnorm requires recognising an
/// rmsnorm, which is the pattern-matching this file's readings are written to avoid. So the reader is
/// what widens, and no descriptor changes: a program that spells the root `math.sqrt` takes the
/// identical path and emits the identical bytes.
///
/// ⭐ `pub` BECAUSE THE CALLER HAS TO REGISTER WHAT THIS READS. The value is looked up in
/// [`BundleLayout::scalarmul_scales`] BY BITS (`scale_slot`), so a caller building that registry must
/// put in exactly the float this function returns — and a caller that cannot call it has to restate
/// the reading instead. That is two matchers for one fact, which is the defect family this file's
/// other comments are a record of. The three `program_*` readers are the registry's contract, so they
/// are part of the door.
pub fn program_rmsnorm_eps(f: &IRFunction<'static>) -> Option<f32> {
    program_rmsnorm_eps_splat(f).map(|(v, _)| v)
}

/// [`program_rmsnorm_eps`] WITH the epsilon's own splat SSA — the seat the registry's format
/// column reads ([`program_stated_scales`]): the value and its df come from the SAME splat, so
/// the two columns of one registry row cannot disagree about which op stated them.
pub fn program_rmsnorm_eps_splat(f: &IRFunction<'static>) -> Option<(f32, Ssa)> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let splat_value = |s: Ssa| -> Option<f64> {
        let sp = def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(kk, v)| match (kk, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    };
    // ONE root op — `math.sqrt` or `math.rsqrt` — or this is not the shape this reading assumes. Two
    // roots means two rmsnorms in one program (or something that is not one at all), and picking
    // either one's epsilon would be a guess about which node is being lowered.
    let mut sqrts = f
        .operations
        .iter()
        .filter(|o| matches!(o.op_type, OpKind::MathSqrt | OpKind::MathRsqrt));
    let sqrt = sqrts.next()?;
    if sqrts.next().is_some() {
        return None;
    }
    let add = def_of(*sqrt.operands.first()?)?;
    if add.op_type != OpKind::ArithAddf {
        return None;
    }
    // The other operand is the mean — `arith.divf` here, `arith.mulf` by a splatted `1/cols` in a
    // producer that folds the divisor — and NEITHER is a splat, so exactly one of the two is. That is
    // the whole discriminator, and it holds for any spelling of the mean. The splat's OWN SSA rides
    // the answer so the format column reads the same op the value came from.
    let mut splats = add.operands.iter().filter(|&&s| splat_value(s).is_some());
    let &eps_splat = splats.next()?;
    if splats.next().is_some() {
        return None;
    }
    Some((splat_value(eps_splat)? as f32, eps_splat))
}

/// The SCALARMUL multiplier the program states, read structurally.
///
/// ⭐ EVERY `arith.mulf` IN THE PROGRAM, AND THEY MUST AGREE. The scalarmul arm splats the
/// scale and multiplies, once for the whole region or once per row block when the region is too wide
/// for the LX — so a program states the same multiplier one or many times, never two different ones.
/// Reading all of them and requiring agreement is what makes the many-block form safe to lower from
/// the program rather than from a record.
/// ⭐ `pub` FOR THE SAME REASON AS [`program_rmsnorm_eps`]: the caller builds the registry this
/// value is looked up in, by bits, so it must be able to read the same value rather than restate the
/// reading.
pub fn program_scalarmul_scale(f: &IRFunction<'static>) -> Option<f32> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let splat_value = |s: Ssa| -> Option<f64> {
        let sp = def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(kk, v)| match (kk, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    };
    let mut found: Option<f64> = None;
    for m in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithMulf)
    {
        // Exactly one operand of a scalarmul's multiply is the splat; the other is the loaded tile.
        let mut splats = m.operands.iter().filter_map(|&s| splat_value(s));
        let v = splats.next()?;
        if splats.next().is_some() {
            return None;
        }
        match found {
            None => found = Some(v),
            Some(v0) if v0.to_bits() == v.to_bits() => {}
            Some(_) => return None,
        }
    }
    found.map(|v| v as f32)
}

/// The registry slot holding `value`, found BY VALUE — the same lookup the attention body already does
/// for its multiplier. The program states the constant and the descriptor reads a bound `[1,1]` const,
/// so the slot is the one thing that has to be looked up rather than read.
fn scale_slot(layout: Option<&BundleLayout>, value: f32) -> Option<usize> {
    layout.and_then(|l| scale_slot_in(&l.scalarmul_scales, value))
}

/// [`scale_slot`]'s two queries over the registry list itself, so the completion pass below
/// resolves absence with the DOOR'S OWN LOOKUP rather than a second rule free to disagree.
fn scale_slot_in(scales: &[f32], value: f32) -> Option<usize> {
    // ⭐ EXACT BITS FIRST, THEN THE f16 IMAGE — and the fallback is not a tolerance.
    //
    // `compute_bundle_layout` registers the tape's f32 constant, and the worker binds ONE
    // fp16 per slot (`SCALE_BYTES = 2`), so the value the descriptor reads is the f16 image
    // of the registry entry either way. A Triton producer's program states its epsilon as
    // the f16 image directly: `LegalizeTypes::step_2b_island_constants` ROUNDS the splat
    // value (`f.to_f16()`) when it feeds an op with an f16 operand — MEASURED, the
    // triton-spyre layout docs record `1e-5` arriving as `0.00001001358`. Matching that
    // against the f32 registry by bits alone refused the CORRECT descriptor, so the f16
    // image is the second query — exact equality of the bound value, not an approximation.
    // The builder path's program keeps its f32 constant, so it still matches on the first
    // query and this changes nothing for it.
    scales
        .iter()
        .position(|s| s.to_bits() == value.to_bits())
        .or_else(|| {
            let v16 = half::f16::from_f32(value);
            scales.iter().position(|s| half::f16::from_f32(*s) == v16)
        })
}

/// ⭐⭐⭐ THE SLOT'S OWN FORMAT — the element type the constant's `tensor.splat` states for its
/// result, read at the SAME SEAT the value came from (the splat op itself), so the registry's
/// format column cannot drift from its value column. The runtime stages a scale const through
/// exactly two channels — the existing f16 `[1,1]` fill, and the raw f32 word — so the df law is:
///
/// * no `result_type` → `Fp16`, the incumbent: every producer this crate has states its scale
///   splats at f16 or untyped, and the untyped form is the f16 one the existing channel writes;
/// * `F16` → [`Df::Fp16`], `F32` → [`Df::Fp32`] — one whole word at the slot's own width;
/// * ⛔ ANY OTHER element type (fp8, int, bool, an address type) IS A BAKE PANIC naming the
///   value and the program: a slot stated at a format no channel writes is a bundle that
///   cannot bake, and the refusal is the whole point — never a nearest-neighbor staging.
pub fn scale_slot_df(f: &IRFunction<'static>, splat: Ssa, value: f32) -> Df {
    let name = f.name;
    let elem = f
        .operations
        .iter()
        .find(|o| o.result == Some(splat))
        .and_then(|o| o.result_type)
        .map(|t| match t {
            ktir_core::irtype::IrType::Tensor { elem, .. }
            | ktir_core::irtype::IrType::MemRef { elem, .. }
            | ktir_core::irtype::IrType::Scalar(elem) => elem,
            ktir_core::irtype::IrType::AccessTile { .. } | ktir_core::irtype::IrType::Index => {
                panic!(
                    "{name}: the scale slot for {value} is stated by a splat whose result is an \
                     ADDRESS type — a scale constant is a value, never an address, so no staging \
                     channel can write it"
                )
            }
        });
    let Some(elem) = elem else {
        return Df::Fp16;
    };
    match super::whole_function::df_of_dtype(elem) {
        Some(Df::Fp16) => Df::Fp16,
        Some(Df::Fp32) => Df::Fp32,
        Some(other) => panic!(
            "{name}: the scale slot for {value} states element format {other:?} — the scale \
             registry stages only IEEE f16/f32 slots (f16: the existing [1,1] const fill; f32: \
             the raw-word channel), so this bundle cannot bake"
        ),
        None => panic!(
            "{name}: the scale slot for {value} states element type {elem:?}, which has no stick \
             geometry — the scale registry stages only IEEE f16/f32 slots, so this bundle cannot \
             bake"
        ),
    }
}

/// ⭐⭐⭐ THE PROGRAM-STATED HALF OF THE SCALE REGISTRY — the values THIS program's ops will be
/// looked up BY VALUE in `BundleLayout::scalarmul_scales`, read with the door's own readers so the
/// collector cannot drift from the door — each value PAIRED WITH ITS OWN `df`, read at the same
/// splat the value came from, because the slot's staging channel is chosen by the format the
/// program states, never by a producer-side guess.
///
/// The census that fills the registry (`scalar_registry` in `scratchy-target-spyre`) is a pure walk
/// of the graph's node payloads: it sees every config-derived scalar a PAYLOAD names. A program can
/// state a constant no payload names — a front-end FOLD of two config constants into one (the exp2
/// restatement's `qk * (SCALE * LOG2E)`, where `LOG2E` is a kernel-local constexpr), which arrives
/// as one splat the census never saw. The door still resolves the value through [`scale_slot`] and
/// refuses without a slot ("registry desync"), so the registry has to be COMPLETED from the
/// programs — at the seam where they are built, never inside the door.
///
/// ⛔ THE READERS ARE THE DOOR'S OWN, ONE PER DOOR. A whole-kernel program is lowered per-op by the
/// whole-function walk, which classifies an `arith.mulf` with exactly one splatted constant as a
/// `ScalarMul` whose multiplier [`scalarmul_at`] resolves by value — and, the same tiebreak, an
/// `arith.maxnumf`/`arith.minnumf` with exactly one splatted constant as the extremum's own scalar
/// form, whose bound [`scalarextremum_at`] resolves by value — and, the same tiebreak again, an
/// `arith.subf` with exactly one splatted constant as the subtraction's own scalar form, whose
/// constant [`scalarsub_at`] resolves by value — so the per-op reader is
/// [`super::whole_function::splat_scale_of`], the same one that walk calls for both tiebreaks. A
/// one-node program is lowered by the per-`Program` door, whose rmsnorm arm resolves its epsilon
/// ([`program_rmsnorm_eps_splat`]) and its `(1+w)` offset ([`program_gain_offset`]) through the same
/// [`scale_slot`] — so those readers are here too. A constant no door looks up (an
/// `Elementwise(Mul)` with no splat operand, a fused body's internal `1/cols`) is never collected:
/// the collector enumerates LOOKUPS, not constants.
pub fn program_stated_scales(f: &IRFunction<'static>) -> Vec<(f32, Df)> {
    let mut stated = Vec::new();
    for op in f.operations.iter() {
        if matches!(
            op.op_type,
            OpKind::ArithMulf | OpKind::ArithMaxnumf | OpKind::ArithMinnumf | OpKind::ArithSubf
        ) && let Some(sf) = super::whole_function::splat_scale_of(f, op)
        {
            stated.push((sf.value, scale_slot_df(f, sf.splat, sf.value)));
        }
        // ⭐ A STORED SPLAT IS A STATED VALUE TOO — the fill's own constant rides the SAME
        // registry the constant-operand ops read ([`scalarfill_at`] resolves it by bits), so a
        // program whose only constants are stored (the online-softmax carry's seed) completes
        // the registry exactly like one whose constants multiply tiles.
        if op.op_type == OpKind::KtdpStore
            && let Some((v, sp)) = super::whole_function::stored_splat_of(f, op)
        {
            stated.push((v, scale_slot_df(f, sp, v)));
        }
    }
    if let Some((eps, sp)) = program_rmsnorm_eps_splat(f) {
        stated.push((eps, scale_slot_df(f, sp, eps)));
    }
    if let Some((off, sp)) = program_gain_offset(f) {
        stated.push((off, scale_slot_df(f, sp, off)));
    }
    stated
}

/// One program-stated scale WITH ITS PROVENANCE — the walk's ordinal for the program that stated
/// it and that program's own name — so a format disagreement at one reserved slot can name BOTH
/// staters in the refusal ([`complete_scale_registry`]). The ordinal is the SEAM's fact (where the
/// program sits in the walk that collected it), never the function's own.
#[derive(Clone, Copy, Debug)]
pub struct StatedScale {
    pub walk: usize,
    pub name: &'static str,
    pub value: f32,
    pub df: Df,
}

/// ⭐⭐⭐ THE COMPLETION — append every stated value the registry lacks, in an order NO WALK
/// CHOOSES. The registry's index IS the device tid (`SCALARMUL_SCALE_BASE - i`), and TWO walks
/// build it: the bake's layout (`compute_bundle_layout`) and the wiring's registries
/// (`graph_wiring`) — the worker binds one const per slot, so a value the two disagreed
/// about would resolve to a slot nobody bound. The payload half is one shared census; the
/// program-stated half is completed per walk, so its ORDER is the hazard: the bake walks
/// prefix → bodies → suffix and the wiring walks nodes, and two folded constants appearing in a
/// different walk order would take different slots. Sorting the appended values by BITS makes the
/// completed registry a function of the SET of stated values alone — every walk order, and every
/// consumer of the list, computes the identical bytes.
///
/// ⛔ APPEND-ONLY. A value the registry already holds — by bits, or by f16 image ([`scale_slot`]'s
/// two queries, applied through [`scale_slot_in`] so the completion and the door cannot disagree
/// about what "already held" means) — is dropped, never renumbered: existing slots are addresses
/// shipped bundles' descriptors already carry. Which is also the no-op law: completing over a
/// program whose constants the census registered leaves the registry byte-identical — pinned by
/// test.
///
/// ⛔⛔ AND THE FORMAT COLUMN MUST AGREE. `dfs` enters parallel to `scales` — the census half's own
/// dfs (all `Fp16`, the staging-twin law) or an earlier completion's — and a stated value that
/// resolves to an ALREADY-HELD slot at a DIFFERENT df is a bake refusal naming the slot, both
/// formats, and the program(s) that stated them: one reserved slot carries one staging channel,
/// and a completion that silently picked either would bind the other program's const at the wrong
/// word width. Two programs disagreeing over one NEW slot is refused the same way, both named.
pub fn complete_scale_registry(scales: &mut Vec<f32>, dfs: &mut Vec<Df>, stated: Vec<StatedScale>) {
    assert_eq!(
        scales.len(),
        dfs.len(),
        "the scale registry's value and format columns are out of parallel — the caller built \
         one without the other"
    );
    let mut missing_vals: Vec<f32> = Vec::new();
    let mut missing_prov: Vec<(usize, &'static str, Df)> = Vec::new();
    for s in stated {
        if let Some(i) = scale_slot_in(scales, s.value) {
            if dfs[i] != s.df {
                panic!(
                    "scale slot {} (value {}) is stated at two formats: the registry holds it at \
                     {:?} and program `{}` (walk #{}) states it at {:?} — one reserved slot \
                     carries one staging channel, so this bundle cannot bake",
                    crate::reserved_tids::scalarmul_scale_tid(i),
                    s.value,
                    dfs[i],
                    s.name,
                    s.walk,
                    s.df
                );
            }
            continue;
        }
        if let Some(j) = scale_slot_in(&missing_vals, s.value) {
            let (walk, name, df) = missing_prov[j];
            if df != s.df {
                panic!(
                    "the scale slot for {} is stated at two formats by two programs: `{}` (walk \
                     #{}) states {:?} and `{}` (walk #{}) states {:?} — one reserved slot carries \
                     one staging channel, so this bundle cannot bake",
                    s.value, name, walk, df, s.name, s.walk, s.df
                );
            }
            continue;
        }
        missing_vals.push(s.value);
        missing_prov.push((s.walk, s.name, s.df));
    }
    // Bits-sorted append — the two-walk law — with the format column riding the same order.
    let mut order: Vec<usize> = (0..missing_vals.len()).collect();
    order.sort_unstable_by_key(|&i| missing_vals[i].to_bits());
    for i in order {
        scales.push(missing_vals[i]);
        dfs.push(missing_prov[i].2);
    }
}

#[cfg(test)]
mod scale_completion_tests {
    use super::*;
    use ktir_core::arena::Arena;
    use ktir_core::ir::Operation;
    use ktir_core::irtype::IrType;

    /// One `x * <constant>` multiply — the whole-kernel form a front-end fold leaves behind: the
    /// constant is a PRODUCT of config constants (`SCALE * LOG2E`), so no graph payload names it
    /// and the payload census never registers it. THE WALL-3 SHAPE ITSELF.
    fn folded(constant: f64) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (x, c, sp, mul) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
        let ops = vec![
            Operation::new(a, Some(c), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Float(constant),
            ),
            Operation::new(a, Some(sp), OpKind::TensorSplat, &[c]),
            Operation::new(a, Some(mul), OpKind::ArithMulf, &[x, sp]),
        ];
        IRFunction {
            name: "folded_probe",
            arguments: a.args(vec![(x, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// The same folded multiply with the splat's RESULT TYPED — the form a producer that states
    /// its formats leaves behind (the f32 m-merge's own constants): the registry's format column
    /// reads THIS type, at the same splat the value came from.
    fn folded_at(constant: f64, elem: ktir_core::dtypes::DType) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (x, c, sp, mul) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
        let ops = vec![
            Operation::new(a, Some(c), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Float(constant),
            ),
            Operation {
                result_type: Some(IrType::Tensor {
                    dims: a.ints(vec![1, 1]),
                    elem,
                }),
                ..Operation::new(a, Some(sp), OpKind::TensorSplat, &[c])
            },
            Operation::new(a, Some(mul), OpKind::ArithMulf, &[x, sp]),
        ];
        IRFunction {
            name: "folded_typed_probe",
            arguments: a.args(vec![(x, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// The reader's pairs zipped with one program's provenance — the walk ordinal and name the
    /// seam supplies, so a df disagreement can name BOTH staters.
    fn stated_of(walk: usize, pairs: Vec<(f32, Df)>) -> Vec<StatedScale> {
        pairs
            .into_iter()
            .map(|(value, df)| StatedScale {
                walk,
                name: "probe",
                value,
                df,
            })
            .collect()
    }

    /// `x * y` — two tensors, no splat: an `Elementwise(Mul)` the door never resolves by value, so
    /// the collector must state nothing for it.
    fn elementwise_mul() -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (x, y, mul) = (Ssa(0), Ssa(1), Ssa(2));
        let ops = vec![Operation::new(a, Some(mul), OpKind::ArithMulf, &[x, y])];
        IRFunction {
            name: "elementwise_probe",
            arguments: a.args(vec![(x, IrType::Index), (y, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// `maximum(x, C)` / `minimum(x, C)` — the extremum's own scalar form, the tiebreak the mulf
    /// arm states applied to the two ops whose identity lives at the range's edge. The online
    /// softmax's running-max initial (−65504, the f16 floor, stated by `tl.full`) is exactly this
    /// shape, and no graph payload names the value.
    fn clamped(constant: f64, max: bool) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (x, c, sp, m) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
        let ops = vec![
            Operation::new(a, Some(c), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Float(constant),
            ),
            Operation::new(a, Some(sp), OpKind::TensorSplat, &[c]),
            Operation::new(
                a,
                Some(m),
                if max {
                    OpKind::ArithMaxnumf
                } else {
                    OpKind::ArithMinnumf
                },
                &[x, sp],
            ),
        ];
        IRFunction {
            name: "extremum_probe",
            arguments: a.args(vec![(x, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// THE EXTREMUM'S OWN WALL-3 HALF. A clamp's stated constant is looked up by value exactly
    /// like a multiplier, and the f16-floor initial is a constant no payload names — both the
    /// maximum and the minimum forms state it, both get exactly one appended slot.
    #[test]
    fn an_extremum_constant_the_census_never_saw_gets_one_appended_slot() {
        for max in [true, false] {
            let mut scales = vec![0.5f32];
            let mut dfs = vec![Df::Fp16];
            let stated = program_stated_scales(&clamped(-65504.0, max));
            assert_eq!(
                stated,
                vec![(-65504.0f32, Df::Fp16)],
                "the {} form states its bound at the incumbent f16 staging",
                if max { "maximum" } else { "minimum" }
            );
            complete_scale_registry(&mut scales, &mut dfs, stated_of(0, stated));
            assert_eq!(scales.len(), 2, "one appended slot, nothing renumbered");
            assert_eq!(
                dfs,
                vec![Df::Fp16, Df::Fp16],
                "the format column rides the append"
            );
            assert_eq!(scales[0], 0.5, "append-only: slot 0 is untouched");
            assert_eq!(
                scale_slot_in(&scales, -65504.0),
                Some(1),
                "the door's own lookup now resolves the bound"
            );
        }
    }

    /// A clamp whose bound the registry lacks is REFUSED as the desync it is — and the refusal
    /// names the extremum kind and the value, the same law `scalarmul_at`'s desync states.
    #[test]
    fn an_unregistered_extremum_bound_is_refused_by_name() {
        let (Err(e), kind) = (
            scalarextremum_at("m0", -65504.0, true, None, &[], &mut 0, None),
            "Maximum m0",
        ) else {
            panic!("no registry means no slot: the extremum must refuse")
        };
        assert!(
            e.message.contains(kind) && e.message.contains("-65504"),
            "names the kind, the op and the bound: {}",
            e.message
        );
        let Err(e) = scalarextremum_at("m0", 448.0, false, None, &[], &mut 0, None) else {
            panic!("no registry means no slot: the extremum must refuse")
        };
        assert!(
            e.message.contains("Minimum m0") && e.message.contains("448"),
            "names the kind, the op and the bound: {}",
            e.message
        );
    }

    /// THE HEADLINE. A folded constant the census never saw gets exactly one appended slot, and
    /// every value the registry already held keeps its slot — the index is a device tid, so a
    /// completion that renumbered existing entries would silently move every shipped constant.
    #[test]
    fn a_folded_constant_the_census_never_saw_gets_one_appended_slot() {
        let mut scales = vec![0.5f32, 0.25];
        let mut dfs = vec![Df::Fp16, Df::Fp16];
        let stated = program_stated_scales(&folded(0.022537231));
        assert_eq!(
            stated,
            vec![(0.022537231f32, Df::Fp16)],
            "the reader states the fold at its own format"
        );
        complete_scale_registry(&mut scales, &mut dfs, stated_of(0, stated));
        assert_eq!(scales.len(), 3, "one appended slot, nothing renumbered");
        assert_eq!(dfs.len(), 3, "the format column rides the append");
        assert_eq!(scales[0], 0.5, "append-only: slot 0 is untouched");
        assert_eq!(scales[1], 0.25, "append-only: slot 1 is untouched");
        assert_eq!(
            scale_slot_in(&scales, 0.022537231),
            Some(2),
            "the door's own lookup now resolves the folded constant"
        );
    }

    /// The no-op law, in its MEASURED form: a Triton producer's program states its epsilon as the
    /// f16 IMAGE of the config constant (`LegalizeTypes` rounding), and the registry holds the f32.
    /// The completion must drop it — the door's f16-image query already resolves it — or every
    /// spliced norm would grow the registry by one unused slot.
    #[test]
    fn a_value_the_registry_holds_by_f16_image_is_not_re_registered() {
        let config: f32 = 1e-5;
        let image = half::f16::from_f32(config).to_f32();
        assert_ne!(config.to_bits(), image.to_bits(), "the probe must round");
        let mut scales = vec![config];
        let mut dfs = vec![Df::Fp16];
        complete_scale_registry(
            &mut scales,
            &mut dfs,
            stated_of(0, program_stated_scales(&folded(image as f64))),
        );
        assert_eq!(
            scales.len(),
            1,
            "the f16 image resolves against the f32 entry: no append"
        );
        assert_eq!(
            dfs,
            vec![Df::Fp16],
            "and no format disagreement: the untyped \
            splat is the incumbent f16 staging, so the f16-image no-op stays a no-op"
        );
    }

    /// ⛔ THE TWO-WALK LAW. The bake completes prefix → bodies → suffix and the wiring completes in
    /// node order; two folded constants met in different orders must still land in the SAME slots,
    /// or the worker would bind a const at a tid no descriptor reads. Sorting the appended half by
    /// bits is what makes the registry a function of the set alone.
    #[test]
    fn the_appended_order_is_independent_of_the_walk_order() {
        let (a, b) = (0.75f32, 0.125);
        let mut bake_order = Vec::new();
        let mut bake_dfs = Vec::new();
        complete_scale_registry(
            &mut bake_order,
            &mut bake_dfs,
            stated_of(0, program_stated_scales(&folded(b as f64)))
                .into_iter()
                .chain(stated_of(1, program_stated_scales(&folded(a as f64))))
                .collect(),
        );
        let mut wiring_order = Vec::new();
        let mut wiring_dfs = Vec::new();
        complete_scale_registry(
            &mut wiring_order,
            &mut wiring_dfs,
            stated_of(0, program_stated_scales(&folded(a as f64)))
                .into_iter()
                .chain(stated_of(1, program_stated_scales(&folded(b as f64))))
                .collect(),
        );
        assert_eq!(
            bake_order, wiring_order,
            "both walks compute the identical registry"
        );
        assert_eq!(
            bake_dfs, wiring_dfs,
            "both walks compute the identical formats"
        );
        assert_eq!(
            bake_order,
            {
                let mut v = vec![a, b];
                v.sort_unstable_by_key(|x| x.to_bits());
                v
            },
            "the appended half is bits-sorted, not walk-ordered"
        );
    }

    /// `x − C` / `C − x` — the subtraction's own scalar form, the non-commutative member of the
    /// constant-operand class. Both STATED ORDERS are built: the online softmax's trip-0 rescale
    /// reads the `tl.full` initial (−65504) on the subtract's FIRST side, and nothing obliges a
    /// producer to put it there — the collector must state the value for either order, and only
    /// the value (the order is the emission's fact, not the registry's).
    fn subtracted(constant: f64, const_first: bool) -> IRFunction<'static> {
        let a: &'static Arena = Arena::global();
        let (x, c, sp, sub) = (Ssa(0), Ssa(1), Ssa(2), Ssa(3));
        let operands = if const_first { &[sp, x] } else { &[x, sp] };
        let ops = vec![
            Operation::new(a, Some(c), OpKind::ArithConstant, &[]).with_attr(
                a,
                AttrKey::Value,
                Attr::Float(constant),
            ),
            Operation::new(a, Some(sp), OpKind::TensorSplat, &[c]),
            Operation::new(a, Some(sub), OpKind::ArithSubf, operands),
        ];
        IRFunction {
            name: "subtract_probe",
            arguments: a.args(vec![(x, IrType::Index)]),
            operations: a.ops(ops),
            grid: (1, 1, 1),
            return_type: None,
        }
    }

    /// THE SUBTRACTION'S OWN WALL-3 HALF. A subtract's stated constant is looked up by value
    /// exactly like a multiplier, from EITHER stated order, and the f16-floor initial is a
    /// constant no payload names — both orders state it, both get exactly one appended slot.
    #[test]
    fn a_subtraction_constant_the_census_never_saw_gets_one_appended_slot() {
        for const_first in [true, false] {
            let mut scales = vec![0.5f32];
            let mut dfs = vec![Df::Fp16];
            let stated = program_stated_scales(&subtracted(-65504.0, const_first));
            assert_eq!(
                stated,
                vec![(-65504.0f32, Df::Fp16)],
                "the {} order states its constant",
                if const_first {
                    "C − tile"
                } else {
                    "tile − C"
                }
            );
            complete_scale_registry(&mut scales, &mut dfs, stated_of(0, stated));
            assert_eq!(scales.len(), 2, "one appended slot, nothing renumbered");
            assert_eq!(scales[0], 0.5, "append-only: slot 0 is untouched");
            assert_eq!(
                scale_slot_in(&scales, -65504.0),
                Some(1),
                "the door's own lookup now resolves the constant"
            );
        }
    }

    /// A subtract whose constant the registry lacks is REFUSED as the desync it is — and the
    /// refusal names the subtraction and the value, the same law `scalarmul_at`'s desync states.
    #[test]
    fn an_unregistered_subtraction_constant_is_refused_by_name() {
        let Err(e) = scalarsub_at("s0", -65504.0, true, None, &[], &mut 0, None) else {
            panic!("no registry means no slot: the subtraction must refuse")
        };
        assert!(
            e.message.contains("Subtraction s0") && e.message.contains("-65504"),
            "names the op and the constant: {}",
            e.message
        );
        let Err(e) = scalarsub_at("s0", 448.0, false, None, &[], &mut 0, None) else {
            panic!("no registry means no slot: the subtraction must refuse")
        };
        assert!(
            e.message.contains("Subtraction s0") && e.message.contains("448"),
            "names the op and the constant, whichever side it was stated on: {}",
            e.message
        );
    }

    /// A multiply with no splat operand is not a scalar multiply and no door resolves anything for
    /// it — the collector enumerates lookups, not constants.
    #[test]
    fn a_mulf_with_no_splat_operand_states_no_scale() {
        assert!(
            program_stated_scales(&elementwise_mul()).is_empty(),
            "an Elementwise(Mul) states no value-looked-up constant"
        );
        let mut scales = vec![0.5f32];
        let mut dfs = vec![Df::Fp16];
        complete_scale_registry(
            &mut scales,
            &mut dfs,
            stated_of(0, program_stated_scales(&elementwise_mul())),
        );
        assert_eq!(scales, vec![0.5], "completion over it is the identity");
        assert_eq!(dfs, vec![Df::Fp16], "and the format column too");
    }

    /// ⭐ THE FORMAT COLUMN'S OWN APPEND: a folded constant stated at f32 (the m-merge's own
    /// form — the splat's result is typed `tensor<1x1xf32>`) completes with [`Df::Fp32`], so the
    /// slot's staging channel is the one the program stated, never a producer-side guess.
    #[test]
    fn an_f32_stated_scale_completes_with_its_own_df() {
        let mut scales = vec![0.5f32];
        let mut dfs = vec![Df::Fp16];
        let stated = program_stated_scales(&folded_at(0.022537231, ktir_core::dtypes::DType::F32));
        assert_eq!(
            stated,
            vec![(0.022537231f32, Df::Fp32)],
            "the reader states the fold AND its own f32 format"
        );
        complete_scale_registry(&mut scales, &mut dfs, stated_of(0, stated));
        assert_eq!(scales.len(), 2, "one appended slot");
        assert_eq!(dfs, vec![Df::Fp16, Df::Fp32], "the appended slot is Fp32");
    }

    /// ⛔ THE SAME-BITS COLLISION, REFUSED: two programs stating the SAME value at DIFFERENT dfs
    /// (an f16-stated −65504 beside the f32 m-merge's own −65504 initial) cannot share one
    /// reserved slot — one tid carries one staging channel. The refusal names BOTH programs.
    #[test]
    #[should_panic(expected = "is stated at two formats by two programs")]
    fn a_same_bits_different_df_collision_is_refused_naming_both_programs() {
        let mut scales = Vec::new();
        let mut dfs = Vec::new();
        complete_scale_registry(
            &mut scales,
            &mut dfs,
            vec![
                StatedScale {
                    walk: 0,
                    name: "f16_stater",
                    value: -65504.0,
                    df: Df::Fp16,
                },
                StatedScale {
                    walk: 1,
                    name: "f32_stater",
                    value: -65504.0,
                    df: Df::Fp32,
                },
            ],
        );
    }

    /// ⛔ AND THE HELD HALF TOO: a stated value resolving to an ALREADY-HELD slot at a different
    /// df is the same refusal with the registry's own holder named — the census's f16 staging
    /// against a program stating f32 at the same bits.
    #[test]
    #[should_panic(expected = "the registry holds it at Fp16")]
    fn a_stated_f32_against_a_held_f16_slot_is_refused() {
        let mut scales = vec![-65504.0f32];
        let mut dfs = vec![Df::Fp16];
        complete_scale_registry(
            &mut scales,
            &mut dfs,
            stated_of(
                0,
                program_stated_scales(&folded_at(-65504.0, ktir_core::dtypes::DType::F32)),
            ),
        );
    }

    /// ⛔ THE FOREIGN-df LAW: a scale splat typed at a format NO channel writes (fp8) is a bake
    /// panic naming the value and the program — fail closed, never nearest-neighbor staging.
    #[test]
    #[should_panic(expected = "the scale registry stages only IEEE f16/f32 slots")]
    fn a_foreign_df_at_a_scale_slot_panics_naming_the_slot() {
        program_stated_scales(&folded_at(0.5, ktir_core::dtypes::DType::Fp8E4m3));
    }
}

/// THE NORM'S GAIN OFFSET — the `(1+w)` convention's `+1`, read off the program's
/// gamma leg: the ONE `arith.addf` whose non-splat operand is a `ktdp.load` (the
/// gamma row's rank-1 load), with exactly one constant-splat operand.
///
/// The emulator executes this addition (the program states it, the oneplus kernel's
/// `w = load + GAIN_OFFSET`), and the door now states the SAME addition as a
/// descriptor: the value's `[1,1]` registry slot is resolved by bits and one `add`
/// fills the `Gw` row ahead of the gain multiply — the epsilon's law, one constant,
/// two readers, one value. `None` when there is no such addf (the `Scale`
/// convention: the whole gemma-4 per-head family and the plain Scale rows), and
/// `Some(0.0)` never occurs — the producer omits the addf entirely at offset 0, so
/// a stated zero-offset addf is a producer bug worth seeing.
fn program_gain_offset(f: &IRFunction<'static>) -> Option<(f32, Ssa)> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let splat_value = |s: Ssa| -> Option<f64> {
        let sp = def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(kk, v)| match (kk, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    };
    for add in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithAddf)
    {
        // The gamma leg's addf: one operand a loaded row, the other the offset splat. The splat's
        // OWN SSA rides the answer (the epsilon's law, one seat for the value and its format).
        let has_load = add
            .operands
            .iter()
            .any(|&s| def_of(s).is_some_and(|d| d.op_type == OpKind::KtdpLoad));
        if !has_load {
            continue;
        }
        let mut splats = add.operands.iter().filter(|&&s| splat_value(s).is_some());
        let &off_splat = splats.next()?;
        if splats.next().is_some() {
            return None;
        }
        return Some((splat_value(off_splat)? as f32, off_splat));
    }
    None
}

/// THE WINDOWED NORM'S EPSILON — every `math.sqrt`/`math.rsqrt` root's feeding `arith.addf`
/// splat, READ ACROSS ALL ROOTS AND REQUIRED TO AGREE.
///
/// [`program_rmsnorm_eps`] demands exactly ONE root, which is the whole-region norm's law
/// (a second root there means two norms in one program). The fused sandwich is different
/// BY CONSTRUCTION: one independent chain per head, so `H` roots stating the SAME epsilon —
/// the same one-or-many-agree shape [`program_scalarmul_scale`] already established for a
/// program that states one constant once per block. A root with no addf, an addf with no
/// splat, or two roots disagreeing by a single bit are each "this is not a windowed norm"
/// and answer `None`.
///
/// ⭐ `pub` FOR THE SAME REASON AS [`program_rmsnorm_eps`]: the caller builds the registry
/// this value is looked up in, by bits, so it must read the same value rather than restate
/// the reading.
pub fn program_windowed_eps(f: &IRFunction<'static>) -> Option<f32> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let splat_value = |s: Ssa| -> Option<f64> {
        let sp = def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(kk, v)| match (kk, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    };
    let mut roots = f
        .operations
        .iter()
        .filter(|o| matches!(o.op_type, OpKind::MathSqrt | OpKind::MathRsqrt));
    let _first = roots.next()?;
    if roots.next().is_none() {
        // One root is the whole-region form — `program_rmsnorm_eps`'s own reading, which
        // this reader must agree with so the two norms resolve the same registry slot.
        return program_rmsnorm_eps(f);
    }
    let mut found: Option<f64> = None;
    for root in f
        .operations
        .iter()
        .filter(|o| matches!(o.op_type, OpKind::MathSqrt | OpKind::MathRsqrt))
    {
        let add = def_of(*root.operands.first()?)?;
        if add.op_type != OpKind::ArithAddf {
            return None;
        }
        let mut splats = add.operands.iter().filter_map(|&s| splat_value(s));
        let eps = splats.next()?;
        if splats.next().is_some() {
            return None;
        }
        match found {
            None => found = Some(eps),
            Some(v0) if v0.to_bits() == eps.to_bits() => {}
            Some(_) => return None,
        }
    }
    found.map(|v| v as f32)
}

/// ⭐ `pub` FOR THE SAME REASON AS [`program_rmsnorm_eps`]: the caller builds the registry this
/// value is looked up in, by bits, so it must be able to read the same value rather than restate the
/// reading.
///
/// TWO FACTS ABOUT THE VALUE, BOTH READ FROM THE PROGRAM AND NEVER ASSUMED:
/// * the score dot appears rank-2 (`linalg.matmul`) or rank-3 (`linalg.batch_matmul`), and the
///   multiplier mulf sits directly on either — see `is_matmul` below;
/// * the attention family states the multiplier EXP2-RESTATED (`qk · (SCALE · LOG2E)`, a
///   front-end fold of two config constants — the softmax is `exp2` end to end), so the value
///   returned is the product, never the raw config scale, and at a scale of one it is `log2e`
///   and not `1.0`. Consumers therefore compare EXISTENCE and LAW, never magnitude against a
///   payload — the one fact that stays a value ([`crate::score_scale`]'s module doc).
///
/// ⭐ DOES THIS PROGRAM COMPUTE SCORES AT ALL — the paged-fold family's other half. The fold
/// is THREE programs (seed, body, finalize): the body computes scores (`qk · scale` over the
/// QK contraction), while the seed initializes the online-softmax carry (`-65504` / `0` /
/// `0` — stores only) and the finalize normalizes it (`acc / l` — a division, no
/// contraction), so neither has a multiplier to state and neither may be held to the door's
/// score-law cross-check. The door demands a stated multiplier only where a QK contraction
/// exists: a contraction-bearing program with no stated multiplier is the `t = 1.0` wall, and
/// a carry program is not that.
pub fn program_has_score_contraction(f: &IRFunction<'static>) -> bool {
    f.operations
        .iter()
        .any(|o| matches!(o.op_type, OpKind::LinalgMatmul | OpKind::LinalgBatchMatmul))
}

pub fn program_score_scale(f: &IRFunction<'static>) -> Option<f32> {
    let const_of: std::collections::HashMap<Ssa, f64> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithConstant)
        .filter_map(|o| {
            let v = o.attributes.iter().find_map(|(kk, v)| match (kk, v) {
                (AttrKey::Value, Attr::Float(x)) => Some(*x),
                _ => None,
            })?;
            Some((o.result?, v))
        })
        .collect();
    // `tensor.splat` result -> the scalar it splatted, when that scalar is a float constant.
    let splat_of: std::collections::HashMap<Ssa, f64> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::TensorSplat)
        .filter_map(|o| Some((o.result?, *const_of.get(o.operands.first()?)?)))
        .collect();
    // ⭐ THE QK DOT HAS TWO SPELLINGS, AND BOTH ARE THE DOT. The flash family's is rank-2
    // (`linalg.matmul`, `[MQ, SWEPT]`); the batched decode/prefill/fold family's is rank-3
    // (`linalg.batch_matmul`, `[B, GQA, 64]` — the batched arm of `dot_to_linalg`), and the
    // score mulf sits directly on either. Admitting only the rank-2 form read the whole
    // batched family as "no multiplier" — the exact silence the door's `Program::Attn`
    // cross-check exists to catch, pointed at a program that states one.
    let is_matmul: std::collections::HashSet<Ssa> = f
        .operations
        .iter()
        .filter(|o| matches!(o.op_type, OpKind::LinalgMatmul | OpKind::LinalgBatchMatmul))
        .filter_map(|o| o.result)
        .collect();
    f.operations
        .iter()
        .filter(|o| o.op_type == OpKind::ArithMulf)
        .find_map(|o| {
            let l = o.operands.first()?;
            let rr = o.operands.get(1)?;
            if !is_matmul.contains(l) {
                return None;
            }
            Some(*splat_of.get(rr)? as f32)
        })
}

/// The CAP a [`Program::TanhSoftCap`](crate::ktir_node::Program::TanhSoftCap)
/// program states, read structurally: the ONE `math.tanh` root, then the
/// `arith.divf` that feeds it — whose non-tile operand is the splatted cap.
///
/// ⭐ `pub` FOR THE SAME REASON AS [`program_rmsnorm_eps`]: the caller builds
/// the registry the `[1,1]` const is resolved in BY BITS (`scale_slot`), so a
/// producer-side value and an emit-side reading must be ONE fact, not two
/// matchers that can drift.
///
/// ONE CAP, however many `math.tanh` roots state it. [`KtirFunc::tanhsoftcap`]
/// emits one `divf → tanh → mulf` chain per ROW BLOCK — prefill's vocab-wide
/// logits exceed the emulator's LX budget as one tile, so a prefill program is
/// N blocks of the same softcap, and the reader must resolve the ONE value all
/// of them state. Two roots with DIFFERENT caps is still refused: that is two
/// softcaps in one program (or something that is not one at all), and picking
/// either's cap would be a guess about which node is being lowered. The divisor
/// is pinned to the `arith.divf` FEEDING each root — not "any divf" — so a
/// program that divides by something else first cannot hand this reader the
/// wrong constant.
pub fn program_tanhsoftcap_cap(f: &IRFunction<'static>) -> Option<f32> {
    let def_of = |s: Ssa| f.operations.iter().find(|o| o.result == Some(s));
    let splat_value = |s: Ssa| -> Option<f64> {
        let sp = def_of(s)?;
        if sp.op_type != OpKind::TensorSplat {
            return None;
        }
        let c = def_of(*sp.operands.first()?)?;
        if c.op_type != OpKind::ArithConstant {
            return None;
        }
        c.attributes.iter().find_map(|(kk, v)| match (kk, v) {
            (AttrKey::Value, Attr::Float(x)) => Some(*x),
            _ => None,
        })
    };
    let mut cap: Option<f64> = None;
    for tanh in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::MathTanh)
    {
        let div = def_of(*tanh.operands.first()?)?;
        if div.op_type != OpKind::ArithDivf {
            return None;
        }
        // Exactly one operand of the divide is the splatted cap; the other is
        // the tile being capped.
        let mut splats = div.operands.iter().filter_map(|&s| splat_value(s));
        let this = splats.next()?;
        if splats.next().is_some() {
            return None;
        }
        match cap {
            None => cap = Some(this),
            Some(c) if c == this => {}
            // Two softcaps in one program — the cap each block states must be
            // the same value or this is not one softcap at all.
            Some(_) => return None,
        }
    }
    Some(cap? as f32)
}

/// main's `lower_rope_node` (main 8696-9528), waiting for its head dim to become a const — the
/// consumer side of `with_config_head_dim`, exactly as main's `LowerRope` was.
pub struct RopeAt<'a> {
    pub name: &'a str,
    /// x, cos, sin, out — main's `inputs[0..3]` and `output`.
    pub x: String,
    pub cos: String,
    pub sin: String,
    pub out: String,
    pub t: u32,
    pub mq: u32,
    pub total: u32,
    /// ⭐ THE ROPE-P CLASS TID this rope's `matmul(x, P)` names. One `[hd,hd]` P per DISTINCT rope
    /// head dim in the tape (a hybrid model carries two, and a composite single P is impossible: the
    /// big class's contraction reads the small class's nonzero block as spurious ±1 terms). The DOOR
    /// resolves the class index off the shared `rope_class_hds` registry, so the emitter and the
    /// placement pass cannot disagree about which table a class reads.
    pub p_tid: u32,
    pub sym_id_base: &'a mut i64,
    pub layout: Option<&'a BundleLayout>,
    pub rows_are_requests: bool,
}

/// main's `lower_rope_node` (main 8696-9528) BODY, at its const head dim. Everything below the
/// `let x = …` binding is main's text.
///
/// ⭐⭐ `HD` IS A CONST GENERIC HERE, and that is the point of this function's whole shape.
///
/// The head dim is a constant of the MODEL, but this emitter is ONE binary serving every model, so inside
/// it the value only becomes known when the proc macro runs. That is why `Shape::<0,0,0,0>` and the
/// `_of(hd, df)` twins existed: an escape hatch for "the const generic cannot be used here". The escape
/// hatch is what made the head-dim-dependent fork below a RUNTIME branch, and a runtime branch is what no
/// compile-time guard can hold onto.
///
/// The fix is a SINGLE dispatch from the value to the const (see the caller), after which everything here
/// is const. `shape.rs`'s own module doc asked for exactly this: "any branch on them would have to be
/// written as a branch on a const, which shows up as a special case in review instead of hiding inside an
/// offset expression."
///
/// What it buys immediately: the collapsed RoPE form and the slab RoPE form become TWO INSTANTIATIONS
/// rather than two arms of one function, so "a head_dim-128 bundle takes the slab path" is a fact the
/// compiler knows.
pub fn rope_at<const HD: u32>(a: RopeAt<'_>) -> Result<Vec<EmittedOp>, Error> {
    let RopeAt {
        name: _name,
        x,
        cos,
        sin,
        out,
        t,
        mq,
        total,
        p_tid,
        sym_id_base,
        layout,
        rows_are_requests,
    } = a;
    let hd = HD;
    if hd == 0 || !total.is_multiple_of(hd) {
        return err(format!(
            "RopeRotate t{t}: output cols {total} not a multiple of head_dim {hd}"
        ));
    }
    let heads = total / hd;
    // The rope plane's own dims, typed once at the door: the chunk's query rows and the whole
    // `[mq, total]` width every rope buffer declares. The `rb` mints below spend these; the
    // per-block arms cross their own quantities through the doors they name.
    let rope_rows = crate::sdsc_abstract::RowCount::of_query_rows(
        crate::sdsc_abstract::QueryRowCount::of_mq(mq),
    );
    let rope_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(total);
    // ROW COUNT: 1 = decode, mq>1 = PREFILL (a [mq, total] roped Q/K). The rope emits one mb=1 [1,hd]
    // block per (row r, head h) — the PROVEN mb=1 primitive. Looping ONLY over heads (mq implicitly 1)
    // was THE prefill bug: it wrote row 0 for every head and NEVER rows 1..mq → roped Q/K collapsed to
    // row 0 → K-cache 1 slot → garbage (measured on-card). CRUCIAL: a [mq,total] activation is
    // STICK-SCATTERED (dev_off: [r,c]→(c/64)·(mq·64)+r·64+c%64), so the (r,h) block's DEVICE offset is
    // rope_prefill_block_offset(r,h,mq,hd)=h·mq·hd+r·hd, NOT the naïve flat r·total+h·hd (that wrote
    // where the attention matmul never reads for r>0 — a baked-but-still-1-row false start). Kani
    // `rope_prefill_bijection_*`: off == dev_off([mq,total],1,[r,h·hd]) (producer==consumer) + bijection.
    // hd>stick in PREFILL: head h spans hd/stick NON-contiguous device sticks (each row r-scattered by
    // mq·stick), so a contiguous [1,hd] block can't address it — hd>stick prefill takes the SLAB path
    // below (per-(row,head,slab) [1,stick] ops). The rotate-half is realized there as a signed slab-swap
    // (rot slab s = ±x slab s±n_slabs/2), which needs half=hd/2 to be a whole number of sticks, i.e.
    // n_slabs even (hd a multiple of 2·stick) — true for every real hd>64 (128/256/512). Guard only the
    // odd-n_slabs case (never a real head_dim) as a build error (not an on-card crash).
    // ASKED OF THE SHAPE, once for this whole emission: the slab count the guard, its refusal and
    // the slab form below all read, through the ONE law's const-generic form — `HD` is this
    // function's own const, so the count is compile-time and `hd / stick` is never spelled here.
    let nslab = crate::addr::Shape::<HD, 0, 0, 0>::slabs(Df::Fp16);
    if hd > Fp16::ELEMS_PER_STICK && !nslab.get().is_multiple_of(2) {
        return err(format!(
            "RoPE prefill (t{t}, mq={mq}): head_dim {hd} > stick {stk} with an ODD stick count \
             ({ns}); the rotate-half slab-swap needs half=hd/2 to be a whole number of sticks (hd a \
             multiple of 2·stick). Real head_dims (128/256/512) satisfy this.",
            stk = Fp16::ELEMS_PER_STICK,
            ns = nslab.get(),
        ));
    }
    // ⭐ THE CLASS TID, NOT THE BARE SENTINEL — the ONE P this rope's head dim class reads. A uniform
    // model's door resolves class 0, which IS `ROPE_P_TID`, so its ops and bytes are unchanged.
    let p = crate::place::act_name(p_tid);
    use SynthRole as R;
    let out_id = PlaceId::Act(t);
    let rot = syn(layout, out_id.synth(R::Rot)); // x · P  (rotate-half, synthetic seg3)
    let xc = syn(layout, out_id.synth(R::Xc)); // x · cos
    let rs = syn(layout, out_id.synth(R::Rs)); // rot · sin
    // PER-(row,head) [1,hd] at the DEVICE offset rope_prefill_block_offset(r,h,mq,hd) — the roped Q/K
    // tensor is [mq, heads·hd] sticked on its last dim, so head h row r's hd dims are one 64-stick at
    // h·mq·hd+r·hd (hd==STK). A single FULL [mq,heads,hd] op would device-stick differently and mismatch
    // the attention matmul's [mq,total] read; per-(r,h) [1,hd] blocks match it exactly (Kani-proven).
    // cos/sin are the worker's [mq,total] per-position head-tiled table, read at row r's head-0 slice.
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(mq as usize * heads as usize * 4);

    // ⭐ THE EMITTER CONTEXT, MINTED ONCE AT THIS DOOR — the sink this function owns plus its own
    // pair parameters as the counter and the placement authority (the mint-once-at-door pattern,
    // the same mint as `assemble_attn` and `lower_function`). Every emitter below threads the ONE
    // `&mut ctx` — the three closures take it as their first parameter in place of the split
    // `ops` + `sib` pair, so the op stream, the ids and the placements are identical.
    let mut ctx = super::EmitCtx {
        ops: &mut ops,
        sym_id_base,
        layout,
    };
    let ew_ph = |ctx: &mut super::EmitCtx<'_>,
                 name: String,
                 f: &'static str,
                 a: &str,
                 aoff: crate::addr::DevOff,
                 b: &str,
                 boff: crate::addr::DevOff,
                 o: &str,
                 ooff: crate::addr::DevOff|
     -> Result<(), Error> {
        // MULTI-STICK AT head_dim > 64, AND THAT IS A REAL DIFFERENCE: at rows==1 a `[1,hd]` op is two
        // sticks at head_dim 128, and the work splitter divides a multi-stick `out` ACROSS CORES — 2
        // cores where head_dim 64 uses 1. No working model has run these ops in that mode, and this
        // file documents a dxp defect in exactly that regime (`cols > 64` ⇒ split across cores).
        //
        // Splitting them stick-wide the way prefill's RoPE already is does NOT work: `rot` is produced
        // by a `[1, hd]` matmul, so reading it as `[1, stick]` slices declares a second arrangement for
        // one tensor and the arrangement authority rejects the bundle. Making decode stick-wide means
        // making the P matmul stick-wide too, which is a different rotate formulation, not a slice.
        //
        // ⭐ TRUE EXTENTS, not `rbo`'s `1×64` annotation: every rope buffer — x/cos/sin (the worker's
        // `[mq, total]` plane) and the rot/xc/rs intermediates — is `[mq, total]` in elements, and this
        // op's `[1, hd]` window rides a `rope_prefill_block_offset` corner of it. The mint reads the
        // handle's dims as the BOUND (#731's door), so the annotation lie is now a refusal; the builder
        // reads name + kind only, so the emission is byte-identical.
        let op = pointwise_broadcast_opspec_df(
            op_func_from_str(f),
            1,
            hd,
            &[
                In::sliced(&rb(a, rope_rows, rope_cols), aoff).ew(),
                In::sliced(&rb(b, rope_rows, rope_cols), boff).ew(),
            ],
            o,
            ooff.into(),
            false, // RoPE per-(row,head) [1,hd] block (rows==1): rank-3 flat
            Df::Fp16,
            ctx.layout,
            EwDivision::Law,
        )
        .map_err(|e| Error { message: e })?;
        ctx.ops.push(emit_sdsc_tiled(
            &name,
            &op,
            &SdscFoldSet::new(op.iter.cores_used()),
            ctx.sym_id_base,
            ctx.layout,
        )?);
        Ok(())
    };
    // ── HEAD-MAJOR COLLAPSE, AT ANY ROW COUNT ───────────────────────────────────────────────────
    // This used to be `mq == 1` only, and the note below measures what that costs: ~40 per-(row,head)
    // ops per op-type, ~150 extra ops a layer, "the single largest unaccounted-for chunk of the perf
    // gap". It is also a fixed cost — the body goes 271 ops a layer at one request to 372 at two, and
    // then only +16 per further request — so a batch pays it in full and amortizes none of it.
    //
    // The collapse generalizes. Head `h` row `r` of the roped `[mq, heads*hd]` tensor sits at
    // `h*mq*hd + r*hd`, and a `[heads*mq, hd]` view indexed `h*mq + r` gives exactly `(h*mq+r)*hd` —
    // the SAME bytes at any `mq`, which is `addr_eq`'s row-expansion law (Kani
    // `row_expansion_is_byte_identical`), not an mq=1 coincidence.
    //
    // cos/sin need one change: at one row every head shares the angles, so they are read mb-broadcast
    // from a single row; above one row they vary per POSITION and must be read per-row. The worker's
    // staging already suits it — element `(p, h*hd+d)` lands at `h*mq*64 + p*64 + d`, which IS row
    // `h*mq+p`, column `d` of the `[heads*mq, hd]` view.
    //
    // Scoped to a decode batch: prefill's own bundles are proven on hardware and stay byte-identical.
    let collapse_rows = if mq == 1 {
        Some(heads)
    } else if rows_are_requests {
        Some(heads * mq)
    } else {
        None
    };
    if let Some(rows) = collapse_rows
        // ⭐ FROM THE TYPE. `HD` is a const generic now, so this is a compile-time constant and the two
        // RoPE forms below are two instantiations of this function rather than two runtime arms.
        .filter(|_| crate::addr::Shape::<HD, 0, 0, 0>::head_major_collapse_valid_here(Df::Fp16))
    {
        // ⛔ DECLARE BEFORE USE. This branch (`hd == stick`, the head-major collapse) referenced
        // `rot`/`xc`/`rs` without ever declaring them, so they fell to `resolve_seg_base`'s bump —
        // which handed all three the SAME address. Rope is `rot = x·P`, `xc = x·cos`,
        // `rs = rot·sin`, `out = xc + rs`: three of its four intermediates were one buffer.
        if let Some(l) = ctx.layout {
            for r in [R::Rot, R::Xc, R::Rs] {
                l.synth(out_id.synth(r), &[rows, hd]);
            }
        }
        // rot = x[rows,hd] @ P[hd,hd] — the same P kernel for every row (m=rows GEMM, k=n=hd).
        ctx.ops.push(assemble_matmul_off(
            &format!("rope_rot_o{t}"),
            crate::sdsc_abstract::MatM::of_head_major_rows(rows),
            crate::sdsc_abstract::MatN::of_head_dim(hd),
            crate::sdsc_abstract::MatK::of_head_dim(hd),
            crate::sdsc_abstract::MatY::unbatched(),
            crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                bmm_site::TapeLoweringSite::witness(),
            ),
            &rb(
                &x,
                RowCount::of_head_major_rows(rows),
                BlockCols::of_head_dim(hd),
            ),
            crate::addr::DevOff::ZERO,
            &Stk::<KernelTag>::head_dim_square(hd, &p),
            crate::addr::DevOff::ZERO,
            &rb(
                &rot,
                RowCount::of_head_major_rows(rows),
                BlockCols::of_head_dim(hd),
            ),
            crate::addr::DevOff::ZERO,
            ctx.sym_id_base,
            ctx.layout,
        ));
        let hh = |name: &str| fl(name, rows, hd);
        // One row of angles shared by every head at mq==1; per-row above that.
        let (cos_h, sin_h) = (hh(&cos), hh(&sin));
        let cos_in = if mq == 1 {
            In::mb_at(&cos_h, crate::addr::DevOff::ZERO)
        } else {
            In::full(&cos_h)
        };
        let sin_in = if mq == 1 {
            In::mb_at(&sin_h, crate::addr::DevOff::ZERO)
        } else {
            In::full(&sin_h)
        };
        // The collapse's typed extents: `rows` is the head-major-collapsed view's row extent
        // (`heads*mq`, where `heads` may be nqh or nkvh), swept head-dim wide.
        let hm_rows = crate::sdsc_abstract::RowCount::of_head_major_rows(rows);
        let hd_cols = crate::sdsc_abstract::BlockCols::of_head_dim(hd);
        ctx.ops.push(pw2(
            &format!("rope_xc_o{t}"),
            "multiply",
            hm_rows,
            hd_cols,
            In::full(&hh(&x)),
            cos_in,
            &hh(&xc),
            ctx.sym_id_base,
            ctx.layout,
        ));
        ctx.ops.push(pw2(
            &format!("rope_rs_o{t}"),
            "multiply",
            hm_rows,
            hd_cols,
            In::full(&hh(&rot)),
            sin_in,
            &hh(&rs),
            ctx.sym_id_base,
            ctx.layout,
        ));
        ctx.ops.push(pw2(
            &format!("rope_add_o{t}"),
            "add",
            hm_rows,
            hd_cols,
            In::full(&hh(&xc)),
            In::full(&hh(&rs)),
            &hh(&out),
            ctx.sym_id_base,
            ctx.layout,
        ));
        return Ok(ops);
    }
    // ── PREFILL (mq>1) hd>stick: SLAB rope ──────────────────────────────────────────────────────────
    // A head's hd dims span n_slabs = hd/stick device STICKS, each a [1,stick] block r-scattered by
    // mq·stick in the stick-scattered [mq,total] tensor — a contiguous [1,hd] block (and thus the [hd,hd]
    // P-matmul) can't address them. So rope per-(row r, head h, slab s): out_s = x_s·cos_s + rot_s·sin_s,
    // where the rotate-half `rot_s` = (s < n/2 ? -x_{s+n/2} : +x_{s-n/2}) is a SIGNED SLAB-SWAP — half=hd/2
    // is n_slabs/2 whole sticks (guarded even above), so no cross-stick P-matmul is needed. All offsets are
    // dev_off([mq,total],1,·) (Kani dev_off); cos/sin read head-0's slab s (the head-tiled table serves
    // every head). At mq==1 or hd==stick this reduces to the loop below (one slab), so hd>stick is the
    // only caller. rot_s's sign is folded into the final add/subtract (rs_s carries the magnitude).
    // ANY `mq`, not just prefill. The slab form emits STICK-WIDE ops; the `[1, hd]` form below is
    // MULTI-STICK at head_dim > 64, and the work splitter divides a multi-stick `out` across CORES —
    // 2 cores at head_dim 128 where head_dim 64 uses 1. That per-core mode is one no working model has
    // run these ops in, and this file documents a dxp defect in exactly it. The slab form also drops
    // the `[hd,hd]` P matmul entirely: the rotate becomes a signed SLAB SWAP, so there is no `rot`
    // tensor and no worker-staged permutation on this path at all.
    //
    // The body is already row-batched over `mq`, so `mq == 1` is one row and needs no special case.
    // head_dim 64 never reaches here (`hd > stick` is false), so granite-3.1-2b is untouched.
    if hd > Fp16::ELEMS_PER_STICK {
        let stk = Fp16::ELEMS_PER_STICK;
        let hd_u = hd as usize;
        let n_slabs = nslab.get();
        // `dev_off(&[mq,total],1,[r,col])` IS `Nest(["row","feat"],[mq,total])`; going through the
        // nest means this loop names a COORDINATE and never writes a stride.
        let doff =
            |r: usize, col: usize| crate::addr::rc_of(mq, total, r as u32, col as u32, Df::Fp16);
        let ew_slab = |ctx: &mut super::EmitCtx<'_>,
                       name: String,
                       f: &'static str,
                       a: &str,
                       aoff: crate::addr::DevOff,
                       b: &str,
                       boff: crate::addr::DevOff,
                       o: &str,
                       ooff: crate::addr::DevOff|
         -> Result<(), Error> {
            // ⭐ TRUE EXTENTS (`[mq, total]`), not `rbo`'s `1×64` annotation — the slab windows are
            // corners of the same `[mq, total]` plane `ew_ph` reads (see its note); the mint reads the
            // dims as the bound and the builder reads name + kind only.
            let op = pointwise_broadcast_opspec_df(
                op_func_from_str(f),
                mq,
                stk,
                &[
                    In::sliced(&rb(a, rope_rows, rope_cols), aoff).ew(),
                    In::sliced(&rb(b, rope_rows, rope_cols), boff).ew(),
                ],
                o,
                ooff.into(),
                // cols == stk, and `stickmajor` needs cols > 64, so it cannot fire either way — this
                // is the same rank-3 flat form regardless of the row count.
                true,
                Df::Fp16,
                ctx.layout,
                EwDivision::Law,
            )
            .map_err(|e| Error { message: e })?;
            ctx.ops.push(emit_sdsc_tiled(
                &name,
                &op,
                &SdscFoldSet::new(op.iter.cores_used()),
                ctx.sym_id_base,
                ctx.layout,
            )?);
            Ok(())
        };
        // ⭐⭐⭐⭐⭐ THREE WHOLE-TENSOR OPS + ONE ROTATE PER HEAD, instead of 3·heads·n_slabs stick blocks.
        // At granite-8b (hd=128, nqh 32 + nkvh 8) that is 240 ops a layer → 43, and RoPE was the single
        // largest op family in the decode body (38% of its 611 non-weight ops).
        //
        // ⭐ WHY IT IS WORTH THE OPS AND NOTHING ELSE: fitting both models' measured layer time against
        // their op counts and `wstride` gives `layer_ms = ops·0.83 µs + weight_MB / 137 GB/s`, and the
        // weight stream is already at 91% of the 150 GB/s IBM's own cost model fits as peak. So per-op
        // overhead is the ONLY bs=1 lever left, and 197 fewer ops a layer is ~6.5 ms/token at 8b.
        //
        // ⛔ SO DO NOT PRICE A COLLAPSE IN MACs. `c1b7e3b7`'s message blames its 8·64·64 identity-matmul
        // MACs; the mode data says that explanation is unsupported. This one ADDS a `[mq,hd]·[hd,hd]` P
        // matmul per head (655 K MACs a layer at 8b, against ~200 M for the projections) and wins. Judge
        // a collapse on LAUNCHES and on the region draw, and measure it paired.
        //
        // THE ROTATE IS THE P MATMUL AGAIN, which this path had dropped in favour of a signed slab
        // swap. The swap is what forced the per-block form: it reads slab `s ± n_slabs/2`, a
        // PERMUTATION of the stick planes, and a pointwise op has one offset and one uniform stride per
        // operand — so the read cannot be expressed whole-tensor (the same rank-2 wall the finalize and
        // the mask hit). P carries the swap AND its sign inside a kernel, so every consumer downstream
        // reads its own block: `rs = rot·sin` and `out = xc + rs` become plain elementwise ops, and the
        // ∓ that used to split the combine by half is gone with it.
        //
        // At hd == stick this branch is not taken at all (the head-major collapse above owns that case,
        // and granite-2b is untouched byte-for-byte).
        let _ = &ew_slab; // the per-block emitter stays for the paths below
        // Declare the intermediates at their TRUE full extent, for the reason the mq>1 branch below
        // records: sized by whichever access declares first, a per-head matmul window would reserve ONE
        // head and under-reserve by `heads` — the bump-allocator defect `synth` exists to kill.
        if let Some(l) = ctx.layout {
            for r in [R::Rot, R::Xc, R::Rs] {
                l.synth(out_id.synth(r), &[mq, total]);
            }
        }
        // ⛔⛔⛔ DECODE ONLY (`mq == 1`), AND THE PREFILL EVIDENCE IS WHY. Emitted at EVERY `mq`, this form
        // is measured WRONG on granite-8b's prompt: the reply reads fluently but says the PROMPT looks
        // jumbled ("the text is a bit jumbled, I'll help you rephrase it"), and TTFT DOUBLES (279.7 vs
        // 124.7 ms) while the decode ITL improves — a mis-encoded prompt that decode then continues
        // fluently from. The addresses are not the cause and that is checked, not assumed:
        // `tests/zz_rope_rot_addr_equiv.rs` proves the ±I block is the right (in-slab, out-slab)
        // coordinate (P is NOT symmetric, so a transposed one would silently negate the rotation), that
        // the slab form reproduces the per-head rotate element for element, and that the offsets step one
        // plane per slab and `mq*hd` per head. So the defect is in the FORM at `mq > 1` — the same shape
        // of finding as the score leg's ("every emitted address is correct and decode is still
        // incoherent"), and the row axis being degenerate at `mq == 1` is exactly what hides it.
        let rot_slab_form = mq == 1;
        for s in (0..n_slabs).filter(|_| rot_slab_form) {
            // P's nonzero block for this output slab: `rope_p_entry` pairs `o` with `o ± hd/2`, and
            // `hd/2` is `n_slabs/2` whole sticks (guarded even above), so the partner is a whole slab.
            let p_in = if s < n_slabs / 2 {
                s + n_slabs / 2
            } else {
                s - n_slabs / 2
            };
            // The ±I block as a COORDINATE on P's own `[in, out]` kernel nest — the same law `attn_krep`
            // reads the identity's diagonal block through, never the product it works out to.
            let p_off = crate::addr::Nest::new(&["row", "feat"], &[hd, hd], Df::Fp16)
                .view()
                .at(crate::addr::Idx::<crate::addr::Row>::n(p_in * stk))
                .slab(s)
                .dev();
            // REQUEST 0, and the op sweeps every row from there: RoPE's rotate covers the whole chunk,
            // so the stream's row axis is swept work rather than a coordinate this site names.
            let a_place = crate::sdsc_abstract::OperandPlacement::of_token_stream_by_slab(
                crate::sdsc_abstract::QueryRowCount::of_mq(mq),
                heads,
                hd,
                0,
                0,
                p_in,
                Df::Fp16,
            );
            let o_place = crate::sdsc_abstract::OperandPlacement::of_token_stream_by_slab(
                crate::sdsc_abstract::QueryRowCount::of_mq(mq),
                heads,
                hd,
                0,
                0,
                s,
                Df::Fp16,
            );
            ctx.ops
                .push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                    &if n_slabs == 1 {
                        format!("rope_rot_o{t}")
                    } else {
                        format!("rope_rot_s{s}_o{t}")
                    },
                    crate::sdsc_abstract::MatM::of_query_rows(
                        crate::sdsc_abstract::QueryRowCount::of_mq(mq),
                    ),
                    crate::sdsc_abstract::MatN::one_stick(crate::sdsc_abstract::Lanes::FP16),
                    crate::sdsc_abstract::MatK::one_stick(crate::sdsc_abstract::Lanes::FP16),
                    crate::sdsc_abstract::MatY::of_gqa_group(heads, a_place, o_place),
                    crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::of_head_batched(
                        bmm_site::TapeLoweringSite::witness(),
                    ),
                    &rb(&x, rope_rows, rope_cols),
                    a_place,
                    &Stk::<KernelTag>::head_dim_square(hd, &p),
                    p_off,
                    &rb(&rot, rope_rows, rope_cols),
                    o_place,
                    ctx.sym_id_base,
                    ctx.layout,
                ));
        }
        // PREFILL KEEPS THE PER-HEAD P MATMUL — the form measured coherent on granite-8b (`7d27d734`),
        // one `[mq,hd]` window per head with the whole head dim contracted. `heads` ops instead of
        // `n_slabs`, and the prompt is right.
        for h in (0..heads as usize).filter(|_| !rot_slab_form) {
            let off_h = doff(0, h * hd_u);
            ctx.ops.push(assemble_matmul_off(
                &format!("rope_rot_h{h}_o{t}"),
                crate::sdsc_abstract::MatM::of_query_rows(
                    crate::sdsc_abstract::QueryRowCount::of_mq(mq),
                ),
                crate::sdsc_abstract::MatN::of_head_dim(hd),
                crate::sdsc_abstract::MatK::of_head_dim(hd),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(&x, rope_rows, rope_cols),
                off_h,
                &Stk::<KernelTag>::head_dim_square(hd, &p),
                crate::addr::DevOff::ZERO,
                &rb(&rot, rope_rows, rope_cols),
                off_h,
                ctx.sym_id_base,
                ctx.layout,
            ));
        }
        // The three elementwise legs, each over the WHOLE tensor: same bytes the per-block ops covered,
        // one op instead of `heads * n_slabs`. cos/sin are the worker's head-TILED table at this site's
        // full width (`tile_one` replicates the hd-wide row across `total/hd` heads), so a whole-tensor
        // read sees, for every head, exactly the head-0 slice the per-block form read.
        let rows_all = crate::sdsc_abstract::RowCount::of_query_rows(
            crate::sdsc_abstract::QueryRowCount::of_mq(mq),
        );
        let cols_all = crate::sdsc_abstract::BlockCols::of_feature_cols(total);
        ctx.ops.push(pw2(
            &format!("rope_xc_o{t}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rb(&x, rope_rows, rope_cols)),
            In::full(&rb(&cos, rope_rows, rope_cols)),
            &rb(&xc, rope_rows, rope_cols),
            ctx.sym_id_base,
            ctx.layout,
        ));
        ctx.ops.push(pw2(
            &format!("rope_rs_o{t}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rb(&rot, rope_rows, rope_cols)),
            In::full(&rb(&sin, rope_rows, rope_cols)),
            &rb(&rs, rope_rows, rope_cols),
            ctx.sym_id_base,
            ctx.layout,
        ));
        // ADD, unconditionally: P already carries the rotate's sign, so there is no first-half/
        // second-half split left to make this op two ops (which the stick planes' interleaving would
        // have forced — first-half slabs are every OTHER plane, not a contiguous run).
        ctx.ops.push(pw2(
            &format!("rope_out_o{t}"),
            "add",
            rows_all,
            cols_all,
            In::full(&rb(&xc, rope_rows, rope_cols)),
            In::full(&rb(&rs, rope_rows, rope_cols)),
            &rb(&out, rope_rows, rope_cols),
            ctx.sym_id_base,
            ctx.layout,
        ));
        return Ok(ops);
    }
    // ── PREFILL (mq>1) hd==stick: PER-HEAD ROW-BATCHED rope (2026-07-28, TTFT) ──────────────────
    // The general per-(row,head) loop below emits 4 ops per (row,head) — at granite prefill that is
    // mq(31)*nqh(32)*4 = 3968 ops for Q plus mq*nkvh*4 = 992 for K, i.e. ~83% of the ENTIRE 5990-op
    // prefill body. Decode dodges this via the mq==1 head-batch short-circuit above; prefill had no
    // equivalent.
    //
    // Heads CANNOT be batched at mq>1: head h row r lives at `h*mq*hd + r*hd`, so consecutive heads
    // are mq*hd apart while an op's own row stride is hd. But for a FIXED head the rows ARE
    // contiguous at stride hd, which is exactly what a [mq, hd] op wants:
    //   * x/rot/xc/rs/out: base `h*mq*hd`, internal dev_off(r,d) = r*hd + d  (cols=hd=stick, so the
    //     op is rank-3 flat and the address is base + r*hd + d) -> matches
    //     rope_prefill_block_offset(r,h,mq,hd) = h*mq*hd + r*hd exactly, for every r.
    //   * cos/sin: the worker stages them [mq,total] stick-scattered, so element (r,i) sits at
    //     (i/64)*(mq*64) + r*64 + (i%64); for head-0's slice (i<hd=64) that is just r*hd + i — a
    //     contiguous [mq,hd] block at offset 0. The table is head-tiled, so head 0's slice serves
    //     every head (same fact the per-row path relies on via rope_prefill_cos_offset).
    // So one [mq,hd] op per head replaces mq of them: 4*mq*heads -> 4*heads ops (Q: 3968 -> 128,
    // K: 992 -> 32). Mathematically identical work, just not unrolled over rows.
    // Gated to hd==stick (the multi-slab hd>stick case is handled by the slab path above) and to
    // mq>1, so decode (mq==1) never reaches here and its emission is untouched.
    // ⭐ FROM THE TYPE — `HD` is this function's const generic, so the gate is a compile-time constant.
    // ⛔ AND NOTE THE `mq > 1` TERM IS A *WIDTH* TEST, NOT A KIND TEST: it reads "more than one query row",
    // which is true of a prefill chunk AND of a decode batch. The comment below claims "decode (mq==1)
    // never reaches here", and that is FALSE for a BATCHED decode, whose mq is the batch width. The kind
    // is `rows_are_requests`; the count cannot stand in for it.
    if mq > 1 && crate::addr::Shape::<HD, 0, 0, 0>::head_major_collapse_valid_here(Df::Fp16) {
        // Declare the rope intermediates at their TRUE full extent. Without this they are sized by
        // whichever access declares first, which was the per-head `[mq,hd]` matmul — ONE head, so the
        // tensor under-reserved by a factor of `heads` (the exact bump-allocator defect
        // `BundleLayout::synth` exists to kill). It also makes the whole-tensor `[heads*mq, hd]` reads
        // below the ones that speak for the layout, since a per-head access no longer covers the
        // footprint and so no longer declares. Scoped to the mq>1 branch so decode's allocation —
        // and therefore its bundle fingerprint — is untouched.
        if let Some(l) = ctx.layout {
            for r in [R::Rot, R::Xc, R::Rs] {
                l.synth(out_id.synth(r), &[heads * mq, hd]);
            }
        }
        let ew_rows = |ctx: &mut super::EmitCtx<'_>,
                       name: String,
                       f: &'static str,
                       rows: u32,
                       a: &str,
                       aoff: crate::addr::DevOff,
                       b: &str,
                       boff: crate::addr::DevOff,
                       o: &str,
                       ooff: crate::addr::DevOff|
         -> Result<(), Error> {
            // ⭐ TRUE EXTENTS, not `rbo`'s `1×64` annotation: this branch reads every buffer as the
            // `[rows, hd]` image — the same elements the `[heads*mq, hd]` synth above reserves (the
            // callers pass `all = heads*mq`) — so the handle states the swept window's own extent. The
            // mint reads the dims as the bound; the builder reads name + kind only.
            let op = pointwise_broadcast_opspec_df(
                op_func_from_str(f),
                rows,
                hd,
                &[
                    In::sliced(
                        &rb(
                            a,
                            RowCount::of_head_major_rows(rows),
                            BlockCols::of_head_dim(hd),
                        ),
                        aoff,
                    )
                    .ew(),
                    In::sliced(
                        &rb(
                            b,
                            RowCount::of_head_major_rows(rows),
                            BlockCols::of_head_dim(hd),
                        ),
                        boff,
                    )
                    .ew(),
                ],
                o,
                ooff.into(),
                // cols == hd == stick, so `stickmajor` cannot fire (it needs cols>64) and this stays
                // the SAME rank-3 flat form the per-row path used — only the row count differs.
                false,
                Df::Fp16,
                ctx.layout,
                EwDivision::Law,
            )
            .map_err(|e| Error { message: e })?;
            ctx.ops.push(emit_sdsc_tiled(
                &name,
                &op,
                &SdscFoldSet::new(op.iter.cores_used()),
                ctx.sym_id_base,
                ctx.layout,
            )?);
            Ok(())
        };
        // ⭐⭐⭐⭐⭐ ONE MATMUL FOR EVERY HEAD AND EVERY ROW — no per-head loop and NO BATCH AXIS.
        //
        // `heads` ops become ONE (32 -> 1 at granite). `P` is the SAME rotation for every head and every
        // row, so this never needed a `y` axis at all: sweep `m = heads*mq` rows against the shared 2-D
        // kernel `[hd, hd]` and every head is covered.
        //
        // ⛔ THE `y`-BATCHED FORM IS WHAT DOES NOT WORK, AND THE REASON IS ARRANGEMENT, NOT STRIDES.
        // Measured on granite-3.1-2b fp8 (mq=7):
        //
        //   tensor 't729': device arrangement StickLayout { rows: 32, cols: 448, Flat }
        //   conflicts with the earlier StickLayout { rows: 7, cols: 2048 }
        //
        // A `y`-batched rank-3 view makes `y` the LEADING (rows) axis, and a tensor gets ONE device
        // arrangement per bundle. The rank-2 sweep has no such problem: `[heads*mq, hd]` is EXACTLY the
        // arrangement the three pointwise legs below already read, and the comment there proves the
        // bytes line up — row `j = h*mq + r` lands at `j*hd = h*mq*hd + r*hd`, which is
        // `rope_prefill_block_offset(r, h, mq, hd)` for every `(h, r)`.
        //
        // So the per-head loop was never buying correctness; it was one op per head doing what one op
        // over all rows does, because the row axis already enumerates (head, row) contiguously.
        ctx.ops.push(assemble_matmul_off(
            &format!("rope_rot_o{t}"),
            crate::sdsc_abstract::MatM::of_token_rows(heads * mq),
            crate::sdsc_abstract::MatN::of_head_dim(hd),
            crate::sdsc_abstract::MatK::of_head_dim(hd),
            crate::sdsc_abstract::MatY::unbatched(),
            crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                bmm_site::TapeLoweringSite::witness(),
            ),
            &rb(
                &x,
                RowCount::of_token_rows(heads * mq),
                BlockCols::of_head_dim(hd),
            ),
            crate::addr::DevOff::ZERO,
            &Stk::<KernelTag>::head_dim_square(hd, &p),
            crate::addr::DevOff::ZERO,
            &rb(
                &rot,
                RowCount::of_token_rows(heads * mq),
                BlockCols::of_head_dim(hd),
            ),
            crate::addr::DevOff::ZERO,
            ctx.sym_id_base,
            ctx.layout,
        ));
        // ── The three POINTWISE legs collapse to ONE op EACH, across all heads. ──────────────────
        // These were `heads` ops apiece (160 of the 402 trips in a granite prefill layer body — the
        // single largest family, 40%) purely because they inherited the per-head loop. They do not
        // need it: at `rows = heads*mq` the op's own row stride is `hd`, so row `j = h*mq + r` lands
        // at `j*hd = h*mq*hd + r*hd` — EXACTLY `rope_prefill_block_offset(r,h,mq,hd)`, for every
        // (h,r). So x/rot/xc/rs/out tile the whole tensor contiguously with no gaps and no reorder.
        //
        // cos/sin line up too, WITHOUT restaging: the worker writes element (r,i) of its [mq,total]
        // table at `(i/64)*(mq*64) + r*64 + (i%64)`, and for i = h*hd+d (hd==stick) that is
        // `h*mq*hd + r*hd + d` — the same row-major `[heads*mq, hd]` image this op reads. The table
        // is head-tiled (every head holds the same row), so head h's copy already sits at row h*mq+r.
        // Byte-identical bytes, read whole instead of `heads` times at offset 0.
        //
        // The rank-3 flat form is unchanged: `cols == hd == stick`, so `stickmajor` still cannot fire
        // (it needs cols>64) — only the row count differs, exactly as when this loop went per-row →
        // per-head. Decode (mq==1) never reaches this branch.
        let all = heads * mq;
        ew_rows(
            &mut ctx,
            format!("rope_xc_o{t}"),
            "multiply",
            all,
            &x,
            crate::addr::DevOff::ZERO,
            &cos,
            crate::addr::DevOff::ZERO,
            &xc,
            crate::addr::DevOff::ZERO,
        )?;
        ew_rows(
            &mut ctx,
            format!("rope_rs_o{t}"),
            "multiply",
            all,
            &rot,
            crate::addr::DevOff::ZERO,
            &sin,
            crate::addr::DevOff::ZERO,
            &rs,
            crate::addr::DevOff::ZERO,
        )?;
        ew_rows(
            &mut ctx,
            format!("rope_add_o{t}"),
            "add",
            all,
            &xc,
            crate::addr::DevOff::ZERO,
            &rs,
            crate::addr::DevOff::ZERO,
            &out,
            crate::addr::DevOff::ZERO,
        )?;
        return Ok(ops);
    }
    for r in 0..mq {
        // Row r's cos/sin slice = dev_off row r head 0 (per-position, head-tiled ⇒ serves every head).
        let coff = crate::addr::rc_of(mq, total, r, 0, Df::Fp16);
        for h in 0..heads {
            // DEVICE offset of row r, head h's [1,hd] block in the stick-scattered [mq,total] tensor
            // (== dev_off([mq,total],1,[r,h·hd]) for hd==STK). mq=1 ⇒ h·hd (decode, byte-identical).
            // `[row, head, feat]` corner. `rope_prefill_block_offset`'s `h*mq*hd + r*hd` is that
            // nest's hd == stick special case; the nest is right at every hd.
            let off = crate::addr::Nest::new(&["row", "head", "feat"], &[mq, heads, hd], Df::Fp16)
                .view()
                .at(crate::addr::Idx::<crate::addr::Row>::n(r))
                .at(crate::addr::Idx::<crate::addr::Head>::n(h))
                .dev();
            // rot[r,h] = x[r,h]·P  (per-head [1,hd] @ [hd,hd] kernel; same P kernel, mb=1)
            ctx.ops.push(assemble_matmul_off(
                &format!("rope_rot_r{r}_h{h}_o{t}"),
                crate::sdsc_abstract::MatM::single_row(),
                crate::sdsc_abstract::MatN::of_head_dim(hd),
                crate::sdsc_abstract::MatK::of_head_dim(hd),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(&x, RowCount::of_token_rows(1), BlockCols::of_head_dim(hd)),
                off,
                &Stk::<KernelTag>::head_dim_square(hd, &p),
                crate::addr::DevOff::ZERO,
                &rb(&rot, RowCount::of_token_rows(1), BlockCols::of_head_dim(hd)),
                off,
                ctx.sym_id_base,
                ctx.layout,
            ));
            ew_ph(
                &mut ctx,
                format!("rope_xc_r{r}_h{h}_o{t}"),
                "multiply",
                &x,
                off,
                &cos,
                coff,
                &xc,
                off,
            )?; // x·cos
            ew_ph(
                &mut ctx,
                format!("rope_rs_r{r}_h{h}_o{t}"),
                "multiply",
                &rot,
                off,
                &sin,
                coff,
                &rs,
                off,
            )?; // rot·sin
            ew_ph(
                &mut ctx,
                format!("rope_add_r{r}_h{h}_o{t}"),
                "add",
                &xc,
                off,
                &rs,
                off,
                &out,
                off,
            )?; // xc+rs
        }
    }
    Ok(ops)
}

/// main's `lower_attn_node` (main 9528-10376) BODY, at its const geometry — BATCHED over q-heads
/// (BatchMatmul, batch=`num_q_heads`), reading the resident transposed-K + replicated K/V cache
/// (seg2, worker-filled) over the full `cap` masked by `t{ATTN_MASK_TID}`. Decode (mq=1): per head
/// `scores[1,cap] = Q·Kᵀ·scale + mask`, softmax, `out=probs·V`. Viewed as `[nqh, cap]` for the
/// ew/softmax ops (mb=nqh, out=cap; `[nqh,1,cap]`≡`[nqh,cap]` bytes). Softmax reduce-outputs are one
/// stick (like rmsnorm's mean).
pub fn attn_at<const NQH: u32, const NKVH: u32, const HD: u32>(
    a: AttnAt<'_>,
    geom: crate::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
) -> Result<Vec<EmittedOp>, Error> {
    let AttnAt {
        name: _name,
        k,
        r,
        rows_are_requests,
        gathers_kv,
        chunk_pages,
        scale_law,
        headmajor_handoff,
        sym_id_base,
        layout,
    } = a;
    // ⭐⭐⭐ THE OPERANDS COME FROM THE PROGRAM. main read them off `node.inputs` / `node.op` /
    // `ir.tensors[..].rows`; the program states every one of them, so this is the same reading through
    // the door this file's header describes — not a record the producer filled in beside it.
    let ops_in = attn_operands(_name, k, r, geom)?;
    let mq = ops_in.mq.get();
    attn_program_is_flat_in_rows(_name, &k.func, mq, geom.nqh())?;
    let (k_id, v_id) = (ops_in.k_id.tid(), ops_in.v_id.tid());
    let (nqh, nkvh, hd) = (geom.nqh(), geom.nkvh(), geom.hd());
    let cap = ops_in.cap.get();
    let stick = Fp16::ELEMS_PER_STICK; // 64
    // ── PAGED-ATTENTION COMPUTE EXTENT, READ OFF THE PROGRAM ──
    //
    // ⭐⭐⭐ IT WAS `AttnAt::active_cap`, THE RUNG, RESOLVED HERE. The rung is a PRODUCER input: it
    // decides how many rows of the resident cache the attention program takes an access tile of, so the
    // program already carries the swept extent and the lowering can read it back. `param_read_rows`
    // is that reading — the widest row extent any `ktdp.construct_access_tile` over the cache
    // parameter takes, and 0 when the program takes none (`ActiveCap::NONE`'s zero-length segment).
    //
    // ⛔ AND THIS WAS MEASURED, NOT ARGUED. The old code resolved the rung and then checked only
    // whether the two AGREED ABOUT ZERO — never that the extents matched — so whether the field was
    // derivable had never actually been established. Making `active_cap != prefix_rows` a hard error
    // and building granite-3.1-8b fp8 on the card, whose bake loops the whole
    // `ActiveCap::decode_ladder` (`codegen.rs:9470`, one bundle per rung), it never fired: every rung
    // of the ladder resolved to exactly what its program reads. The emulator alone would NOT have
    // shown this — it bakes `ActiveCap::FULL`, where `resolve` is trivially `cap`, which is the same
    // blind spot that let a wrong rope reading through.
    let prefix_rows = param_read_rows(&k.func, &k.func.arguments[ops_in.k_param].0);
    // ⛔⛔⛔ THE RUNG AND THE PROGRAM MUST AGREE ABOUT WHETHER THERE IS A PREFIX AT ALL, because
    // that is the one place the two consumers could compute different things and neither would fault.
    // `ActiveCap::NONE` means `nb == 0` here — the new-token block alone seeds and finalizes the
    // softmax — and the producer says the same thing by taking NO access tile of the resident cache
    // (the tensor is still named, so its identity survives; see the attention program's zero-length
    // segment). A rung that swept nothing against a program that reads the prefix — or the reverse —
    // is an emulator that cannot be an oracle for this node, so it is a build error naming both.
    // The swept extent cannot exceed the buffer it sweeps: the cache parameter's own view states
    // `cap` rows, and a program taking a window past that addresses bytes nothing wrote.
    if prefix_rows > cap {
        return err(format!(
            "AttnDecode t{}: the program reads {prefix_rows} row(s) of the resident cache t{k_id} \
             whose own view states only {cap} — the capacity read off that view is not the buffer \
             this program addresses.",
            ops_in.out_id.tid(),
        ));
    }
    // ⭐ THE EXTENT PAIR, MINTED ONCE — storage from the cache's own view, sweep from the
    // program's own access tile, AFTER the agreement check above (this is the party that can
    // name the node in the refusal). `assemble_attn` takes the pair as this carrier, so no call
    // site can hand the storage extent in the sweep slot or the reverse.
    let extent = crate::sdsc_abstract::KvExtent::new(
        ops_in.cap,
        crate::sdsc_abstract::SlotCount::new(prefix_rows),
    );
    // ⭐⭐⭐ THE MULTIPLIER COMES FROM THE PROGRAM, AND IT IS THE ONLY READING OF IT. The attention
    // program states `attention_multiplier` inline (`self.scalar(f64::from(scale))`) because the emulator
    // interprets these ops and the three attention optimizers recognise the scale by pattern; the
    // DEVICE reads it as a bound `[1,1]` const, whose slot is resolved below FROM THIS VALUE.
    //
    // ⛔ IT WAS A CALLER-STATED `f32` CHECKED AGAINST THIS DERIVATION, WHICH IS A SIDECAR. The check
    // could only ever pass — every program that lowered at all agreed with its caller — so the value
    // the caller passed was never the source of anything, only a second copy of one config number
    // travelling `SubtileIR → params → SuperDSC` past the IR. A third-party producer had to supply a
    // number its own program already states. There is one reading now, and no default: a program with
    // no multiplier is a build error naming it.
    //
    // ⭐ EXCEPT THE TWO CASES WHERE THE PROGRAM CANNOT STATE IT. One is THE FOLD: when the bundle's
    // `√attention_multiplier` was folded into `W_q`/`W_k`, the node's scale was rewritten to 1.0
    // before the splice minted this program, and the triton→ktir pipeline folds a multiply by 1.0
    // away entirely — the program TRUTHFULLY states no multiplier, because no device multiply is
    // left to do (the weights carry `√scale` each). The other is THE UNIT SCALE: a model whose
    // attention scale genuinely is 1.0 (gemma-4's arch defaults `attention_multiplier` to it) has
    // no multiply to state in the first place, and the same pipeline folding takes it — and the
    // skip arm below is the correct lowering for both, because `√1.0` leaves the weights carrying
    // the whole scale either way. `scale_law` is the caller's proof that one of these
    // is the case; a multiplier-less program without it is still the build error below, and a
    // program that states a real multiplier WHILE the caller says the weights carry the scale is
    // the half-applied fold — the multiplies would double-scale pre-scaled weights, so it is
    // refused, not best-efforted.
    let scale_val = match program_score_scale(&k.func) {
        None if matches!(
            scale_law,
            ScoreScaleLaw::CommutedIntoWeights | ScoreScaleLaw::UnitByModel
        ) =>
        {
            None
        }
        None => {
            return Err(Error {
                message: format!(
                    "AttnDecode t{}: the program states no score multiplier. The attention program \
                     scales its scores with an `arith.mulf` against an `arith.constant`, and the \
                     descriptor's `[1,1]` scale const is resolved from that value, so a program \
                     without one cannot be lowered — unless the bundle's score scale is already \
                     carried by its staged weights (folded into W_q/W_k, or a unit attention \
                     scale), which the caller must then state in `BundleAttnParams::scale_law`.",
                    ops_in.out_id.tid(),
                ),
            });
        }
        Some(v)
            if v != 1.0
                && matches!(
                    scale_law,
                    ScoreScaleLaw::CommutedIntoWeights | ScoreScaleLaw::UnitByModel
                ) =>
        {
            return Err(Error {
                message: format!(
                    "AttnDecode t{}: the bundle states its attention scale is carried by the staged \
                     weights, but the program still states a score multiplier of {v}. Both cases \
                     that carry the scale (the fold, a unit multiplier) mint the program at 1.0, so \
                     a real multiplier here means the graph and the splice disagreed — emitting the \
                     multiplies would scale weights that already carry the scale.",
                    ops_in.out_id.tid(),
                ),
            });
        }
        some => some,
    };
    // SPAN-OVERFLOW GUARD (ported from torch-spyre's span_overflow_hint_analysis.py). Real
    // corrective re-tiling of the resident cache's PHYSICAL STORAGE (not just the compute sweep
    // `active_cap` already bounds) means paging the cache into multiple physical buffers — a
    // structural change to the resident-KV allocation, not something this one call site can retrofit
    // in place. So this guard does what torch-spyre's planner does BEFORE re-tiling: run the actual
    // search (`cheapest_split_clearing_span`) and report the split it finds, rather than a bare
    // "not implemented". Unreachable for every real model config checked so far (granite hd=64,
    // cap up to several thousand: span ~0.5 MB against a 256 MB budget — see
    // `ir::bridge::span_overflow`'s own tests) — kept as a real `cargo build` guard, not a runtime
    // assert, so a future config that DOES trip it fails loudly with the fix already computed.
    {
        use crate::ir::bridge::span_overflow::{
            MAX_SPAN_BYTES, cheapest_split_clearing_span, physical_span_bytes,
        };
        let span = physical_span_bytes(cap, hd, stick, Fp16::WORD_LENGTH);
        if span > MAX_SPAN_BYTES {
            let cap_dim = ItDim {
                name: "cap",
                size: cap,
                is_reduction: false,
                is_stick: true,
                df: Df::Fp16,
            };
            let split_report = match cheapest_split_clearing_span(&cap_dim, |split| {
                physical_span_bytes(cap / split.max(1), hd, stick, Fp16::WORD_LENGTH)
            }) {
                Ok(split) => format!(
                    "the cheapest legal split of `cap` that clears the limit is {split}-way \
                     (cap/{split}={} slots/core) — but applying it requires paging the resident \
                     KV allocation, not implemented at this call site",
                    cap / split.max(1)
                ),
                Err(e) => format!("no legal split of `cap` clears the limit either: {e}"),
            };
            return err(format!(
                "AttnDecode t{}: resident K/V cache [cap={cap}, hd={hd}] physical span {span} B \
                 exceeds the {MAX_SPAN_BYTES} B hardware addressing limit. {split_report}.",
                ops_in.out_id.tid(),
            ));
        }
    }
    // ⛔ ONE SOURCE FOR THE PADDED ROW COUNT AND THE ROW LAWS: `attn_bundle_rows` is the parse
    // boundary from the bundle's runtime width to the pad law WITH the geometry's consts in scope,
    // and the SAME carrier rides into `assemble_attn`, so the two cannot disagree. A decode width
    // (one row, or rows that are requests) must be a baked ladder rung there — the pad is
    // `Rung<MQ>`'s compile-time arithmetic and every row extent is `RungRowLaws`' const
    // (`MaskRows = NQH*MQ` by the compiler), and an unlisted width is this loud bake error, the
    // same boundary discipline as the geometry door in `lower_one_node`. A prefill chunk's width is
    // a runtime quantity and takes the runtime arm of the same law.
    let bundle_rows = crate::sdsc_abstract::attn_bundle_rows(geom, mq, rows_are_requests)
        .ok_or_else(|| Error {
            message: format!(
                "AttnDecode t{}: a decode bundle at {mq} query rows — not a width the decode \
                     ladder bakes (1, or PagedKvPool::BATCH_RUNGS). The decode width is a const of \
                     the bundle (`Rung<MQ>`); bake a listed rung, or grow the ladder and its \
                     dispatch together.",
                ops_in.out_id.tid()
            ),
        })?;
    let width = bundle_rows.width();
    let mq_pad = width.pad().rows();
    let t = ops_in.out_id.tid();
    let q = crate::place::act_name(ops_in.q_id.tid()); // roped Q [mq, nqh·hd]
    // roped new-K [mq_pad, nkvh·hd] (worker zero-pads rows)
    let new_k = crate::place::act_name(ops_in.new_k_id.tid());
    let new_v = crate::place::act_name(ops_in.new_v_id.tid()); // new-V       [mq_pad, nkvh·hd]
    let kc = crate::place::act_name(k_id); // natural K cache [nqh, cap, hd] (seg2, GQA-replicated, slab-major)
    let vc = crate::place::act_name(v_id); // natural V cache [nqh, cap, hd] (seg2, GQA-replicated, slab-major)
    // ⭐ THE Kᵀ PLANE'S NAME — minted through the reserved binding's own type, so the only tid
    // this name can resolve to is the plane the pool law placed (never the K-cache source).
    let kct = crate::place::act_name(KctResident::of(k_id).wire_tid()); // resident Kᵀ scratch [nqh, hd, cap]
    // ⭐⭐⭐ AN ATTENTION WHOSE SCALE THE WEIGHTS CARRY STATES NO SCALE AT ALL. The √scale was
    // FOLDED into W_q and W_k (the producer half's `attn_scale_weight_folds` recognition), or the
    // model's attention scale IS 1.0 and there was nothing to fold: the splice minted this program
    // at scale 1.0 either way, the triton→ktir pipeline folded the `qk · 1.0` multiply away
    // entirely, and the registry census skipped the node's arm — so `scale_val` above is `None`,
    // and the query and the roped new-K arrive over the staged weights already carrying √scale
    // each: `(q·√s)·(k·√s)ᵀ` is `(q·kᵀ)·s` with no device multiply left to do. The skip arm at the
    // multiplies below takes both: no registry lookups (a folded node pushed no slot, so looking
    // anything up is a desync error by design), no `qs`/`new_k_scaled` buffers, and the score legs
    // read `q` and the roped new-K directly.
    let pmask = crate::place::act_name(ATTN_MASK_TID); // [nqh, cap]    prefix validity (worker-tiled, mb-broadcast over mq)
    let cmask = crate::place::act_name(ATTN_CAUSAL_TID); // [mq, mq_pad] causal triu (worker-tiled)
    // [mq, nqh·hd] — the identity; every scratch name below is a rendering of it.
    let attn_id = PlaceId::Act(t);
    let n = |r: SynthRole| syn(layout, attn_id.synth(r));
    // ⭐ THE IDENTITY CLASS TID — which `[hd,hd]` identity table this attention class's krep/vrep and
    // cachewr matmuls name. One table per DISTINCT AttnDecode head dim (the bake's
    // `attn_class_hds` registry), because a hybrid model's wide class reads slab 1 of a
    // max-hd placement holding a base-hd table as ALL ZERO (dump-proven on gemma-4
    // tiny-allglobal: `newkrep` lanes 64..127 zero while lanes 0..63 equal the roped K).
    // A uniform model resolves class 0, which IS `IDENTITY_TID`, so its ops are unchanged.
    let ident = crate::place::act_name(
        layout
            .and_then(|l| l.identity_tid_of_class(hd))
            .ok_or_else(|| Error {
                message: format!(
                    "AttnDecode t{t}: head_dim {hd} absent from BundleLayout.attn_class_hds {:?} \
                     (registry desync — the placement pass and this lowering disagree about the \
                     tape's attention classes, so which identity table this node reads is not a \
                     fact either of them states)",
                    layout.map(|l| l.attn_class_hds.clone()).unwrap_or_default(),
                ),
            })?,
    );

    let mut ops: Vec<EmittedOp> = Vec::new();

    // ── (0) ZERO new_k/new_v's PADDING rows [mq..mq_pad) before anything reads them (BUG #3, see
    // ATTN_ZERO_TID's placement comment above). `new_k`/`new_v` are a lifetime-reused seg3
    // intermediate — never explicitly re-zeroed between steps/layers — so rows beyond the real `mq`
    // can alias whatever unrelated tensor last lived at that byte range. The causal mask only
    // reliably neutralizes BOUNDED garbage; this makes the padding actually zero instead of relying
    // on that. Per-kv-head (nkvh, not nqh — new_k/new_v are still nkvh-wide here, before GQA-replicate
    // expands them), one zero-copy each, matching ATTN_ZERO's own [mqp,hd] per-head-width shape.
    if mq_pad.row_axis_extent() > mq {
        let zero = crate::place::act_name(ATTN_ZERO_TID);
        let pad_rows = mq_pad.row_axis_extent() - mq;
        // THE ONE OFFSET LEFT THAT A NEST CANNOT EXPRESS, and the reason is a real layout defect, not
        // a missing abstraction. `new_k`/`new_v` are ALLOCATED `[mq_pad, nkvh*hd]` but RoPE PACKS them
        // by the REAL row count `mq`, so a per-head block holds exactly `mq` rows and has no room of
        // its own for padding: "row mq of head kvh" is not an address that exists. Framing the zero-fill
        // on the packed nest trips the footprint check at decode (mq=1) immediately, and framing it on
        // the padded nest moves every other consumer. So the zeros go where they have always gone --
        // past the packed data -- which is NOT where the Kt restickify reads this head's padding from.
        // It reads the NEXT head's real K/V there and relies on the causal mask to neutralise it.
        //
        // Reconciling that means making RoPE write `mq_pad`-framed, which is a worker change as well as
        // an emitter one. Until then this stays byte-identical rather than becoming a second convention.
        for kvh in 0..nkvh {
            ops.push(assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("attn_kzero{kvh}_o{t}"),
                    op_func: "identity",
                    rows: crate::sdsc_abstract::RowCount::of_zero_pad_rows(pad_rows),
                    cols: crate::sdsc_abstract::BlockCols::of_head_dim(hd),
                    inputs: &[In::full(&rb(
                        &zero,
                        RowCount::of_zero_pad_rows(pad_rows),
                        BlockCols::of_head_dim(hd),
                    ))
                    .ew()],
                    o: &rb(
                        &new_k,
                        RowCount::of_padded_chunk_rows(mq_pad),
                        BlockCols::of_feature_cols(nkvh * hd),
                    ),
                    // FLAT, and that is the finding: this lands past the packed data, in a region no
                    // consumer reads (see the note above), so it is placed row-major rather than through
                    // the stick law. Saying `flat` makes which arrangement was meant explicit instead of
                    // leaving `mq*nkvh*hd + kvh*hd` for a reader to infer.
                    out_offset: crate::addr::Nest::flat(
                        &["row", "feat"],
                        &[mq_pad.row_axis_extent(), nkvh * hd],
                        Df::Fp16,
                    )
                    .view()
                    .at(crate::addr::Idx::<crate::addr::Row>::n(mq))
                    .at(crate::addr::Idx::<crate::addr::Feat>::n(kvh * hd))
                    .dev()
                    .into(),
                },
                sym_id_base,
                layout,
            ));
            ops.push(assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("attn_vzero{kvh}_o{t}"),
                    op_func: "identity",
                    rows: crate::sdsc_abstract::RowCount::of_zero_pad_rows(pad_rows),
                    cols: crate::sdsc_abstract::BlockCols::of_head_dim(hd),
                    inputs: &[In::full(&rb(
                        &zero,
                        RowCount::of_zero_pad_rows(pad_rows),
                        BlockCols::of_head_dim(hd),
                    ))
                    .ew()],
                    o: &rb(
                        &new_v,
                        RowCount::of_padded_chunk_rows(mq_pad),
                        BlockCols::of_feature_cols(nkvh * hd),
                    ),
                    out_offset: crate::addr::Nest::flat(
                        &["row", "feat"],
                        &[mq_pad.row_axis_extent(), nkvh * hd],
                        Df::Fp16,
                    )
                    .view()
                    .at(crate::addr::Idx::<crate::addr::Row>::n(mq))
                    .at(crate::addr::Idx::<crate::addr::Feat>::n(kvh * hd))
                    .dev()
                    .into(),
                },
                sym_id_base,
                layout,
            ));
        }
    }

    // ── (1) GQA-replicate: NEITHER K nor V's freshly-computed new-token block replicates to nqh
    // anymore. `kct`/(the value-side equivalent for the new block) only ever need ONE representative
    // copy per kv-head group — the resident-cache side (`kc`/`vc`, still genuinely nqh-sized for
    // their own OWN separately-proven cache-write convention) is untouched, but the new-token block's
    // OWN scratch buffers (new_k_rep/new_v_rep) are fresh allocations with no such constraint, so
    // dedup them too. `assemble_attn`'s new-block reads (both K, from before, and V, now) map query
    // head h to its kv-head via `gqa_kv_head`, matching producer/consumer by construction.
    //
    // A DECODE BATCH TRIMS THEM TOO. The copies exist so a per-head reader finds its own replicated
    // slot; the GQA-group-batched attention reads the REPRESENTATIVE slot directly and never looks at
    // them, and a batched decode now runs that form. So the `nkvh*2` identity copies a layer are pure
    // work at any batch width, for the same reason they are at one row.
    //
    // Prefill keeps them — it runs the per-head reader — so its bundles stay byte-identical.
    let decode_trim =
        // ⭐ THE PARAMETER, NOT THE GLOBAL. This function already RECEIVES the kind; reading the
        // thread-local here was a second carrier for one fact, and the two could disagree.
        // ⛔ AND `mq == 1` IS NOT "SOLO DECODE": it is also a one-token prefill chunk. The kinds
        // coincide at width one, which is exactly why `bs=1` passing never proved anything about the
        // batched kind. Keep both terms, but understand the first as a WIDTH fact and the second as a
        // KIND fact — they are not two spellings of the same test.
        mq == 1 || rows_are_requests;
    let (new_k_rep, new_v_rep) = if decode_trim {
        (new_k.clone(), new_v.clone())
    } else {
        (
            syn(layout, attn_id.synth(SynthRole::NewKRep)),
            syn(layout, attn_id.synth(SynthRole::NewVRep)),
        )
    };
    if !decode_trim && let Some(l) = layout {
        for r in [SynthRole::NewKRep, SynthRole::NewVRep] {
            l.synth(attn_id.synth(r), &[mq_pad.row_axis_extent(), nkvh * hd]);
        }
    }
    // PER-KV-HEAD BASE = `kvh * mq * hd` (2026-07-28, CORRECTED from an earlier `mq_pad` version that
    // regressed decode). The stride between kv-head column-blocks in these `[*, nkvh*hd]` tensors is
    // set by their PRODUCER, RoPE, which writes head h row r at
    // `rope_prefill_block_offset(r,h,mq,hd) = h*mq*hd + r*hd` -- stride `mq`, the chunk's REAL query-row
    // count, NOT the stick-padded `mq_pad`.
    //
    // SLAB-SPLIT, for the same reason as every other head-dim-spanning op here. A single
    // `[mq_pad, hd]` copy addresses its own second stick at `mq_pad*stick` — its declared row count
    // — but RoPE, which WROTE this buffer, puts it at `mq*stick`, because it packs each (head, slab)
    // as `mq` tightly-strided rows. Those coincide only when a head is ONE stick, and at
    // head_dim 128 / mq 31 / mq_pad 64 they are 2112 elements apart: every kv-head's upper half of K
    // and V is copied from, and to, the wrong place on every prefill.
    //
    // At nslab == 1 there is one iteration, every slab term is 0, `stick == hd` makes `n`/`k`
    // identical, and the name collapses -- granite-3.1-2b emits exactly what it did before.
    let krep_nslab = crate::addr::Shape::<HD, 0, 0, 0>::slabs(Df::Fp16).get();
    for kvh in 0..nkvh {
        if decode_trim {
            break; // the copies are the identity — `new_k_rep`/`new_v_rep` ARE `new_k`/`new_v`.
        }
        for s in 0..krep_nslab {
            // ⛔⛔⛔ NAMED AXES, NOT HAND-SUMMED PRODUCTS — the same conversion as `attn.rs`'s kv stream,
            // and the same reason: `kvh * hd + s * stick` adds two products of DIFFERENT UNITS (kv-head x
            // head_dim, slab x lanes) into one column, and the two are IDENTICAL at hd == stick == 64.
            let slab_off =
                crate::addr::Nest::new(&["row", "head", "feat"], &[mq, nkvh, hd], Df::Fp16)
                    .view()
                    .at(crate::addr::Idx::<crate::addr::Head>::n(kvh))
                    .slab(s)
                    .dev();
            // The identity's (in-slab s, out-slab s) diagonal block, as a coordinate on the [in, out]
            // kernel nest rather than the product `s*stick*(hd+stick)` it works out to.
            let ident_off = crate::addr::Nest::new(&["row", "feat"], &[hd, hd], Df::Fp16)
                .view()
                .at(crate::addr::Idx::<crate::addr::Row>::n(s * stick))
                .slab(s)
                .dev();
            let nm = |base: &str| {
                if krep_nslab == 1 {
                    format!("attn_{base}{kvh}_o{t}")
                } else {
                    format!("attn_{base}{kvh}s{s}_o{t}")
                }
            };
            ops.push(assemble_matmul_off(
                &nm("krep"),
                crate::sdsc_abstract::MatM::of_padded_chunk_rows(mq_pad),
                crate::sdsc_abstract::MatN::one_stick(crate::sdsc_abstract::Lanes::FP16),
                crate::sdsc_abstract::MatK::one_stick(crate::sdsc_abstract::Lanes::FP16),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(
                    &new_k,
                    RowCount::of_padded_chunk_rows(mq_pad),
                    BlockCols::of_feature_cols(nkvh * hd),
                ),
                slab_off,
                &Stk::<KernelTag>::head_dim_square(hd, &ident),
                ident_off,
                &rb(
                    &new_k_rep,
                    RowCount::of_padded_chunk_rows(mq_pad),
                    BlockCols::of_feature_cols(nkvh * hd),
                ),
                slab_off,
                sym_id_base,
                layout,
            ));
            ops.push(assemble_matmul_off(
                &nm("vrep"),
                crate::sdsc_abstract::MatM::of_padded_chunk_rows(mq_pad),
                crate::sdsc_abstract::MatN::one_stick(crate::sdsc_abstract::Lanes::FP16),
                crate::sdsc_abstract::MatK::one_stick(crate::sdsc_abstract::Lanes::FP16),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(
                    &new_v,
                    RowCount::of_padded_chunk_rows(mq_pad),
                    BlockCols::of_feature_cols(nkvh * hd),
                ),
                slab_off,
                &Stk::<KernelTag>::head_dim_square(hd, &ident),
                ident_off,
                &rb(
                    &new_v_rep,
                    RowCount::of_padded_chunk_rows(mq_pad),
                    BlockCols::of_feature_cols(nkvh * hd),
                ),
                slab_off,
                sym_id_base,
                layout,
            ));
        }
    }

    // ── (2) qs = Q·sqrt_scale, new_k_scaled = new_k_rep·sqrt_scale (torch-spyre scales BOTH Q and K
    // by sqrt(scale), so `scores = (Q·√s)@(K·√s)ᵀ = Q@Kᵀ·s`). The resident cache stores K ALREADY
    // scaled (see the cache-write below), so the prefix score needs only qs scaled. new_k_scaled is
    // nkvh-wide now (K dedup, see above) — assemble_attn's "new block" score reads it per-query-head
    // via the SAME gqa-dedup mapping kct_base already uses (matching producer/consumer by construction).
    //
    // ⭐ WHEN THE PROGRAM STATES NO MULTIPLIER — OR STATES 1.0 (see the comment beside `scale_val`)
    // this whole section is SKIPPED: no multiplies, no `[1,1]` consts looked up, no
    // `qs`/`new_k_scaled` synthetics — the score legs read `q` and the roped new-K (which the fold
    // already pre-scaled) by their own names.
    let (qs, new_k_scaled) = match scale_val {
        None | Some(1.0) => (q.clone(), new_k_rep.clone()),
        // attention_multiplier (config) — see the ORIGINAL header doc: NO 1/sqrt(hd) recompute.
        Some(scale_val) => {
            // ⭐ ONE LOOKUP LAW — `scale_slot`, the door every other bound constant resolves
            // through (exact bits, then the f16 image). This arm once carried its own bits-only
            // copy of the lookup, which refused any config whose attention multiplier is not
            // f16-exact: the program states the multiplier's fp16 image (its `qk * SCALE` mulf
            // runs on the fp16 dot's result, so the constant is the f16-rounded spelling) while
            // the census registers the config's f32, and the two agree by bits only when the
            // value is f16-exact. The f16 image the fallback matches is the value the worker
            // binds anyway (`SCALE_BYTES = 2`).
            let scale_idx = scale_slot(layout, scale_val).ok_or_else(|| Error {
                message: format!(
                    "AttnDecode t{t}: scale {scale_val} absent from BundleLayout.scalarmul_scales \
                     (registry desync)"
                ),
            })?;
            // unused directly: torch-spyre splits into sqrt_scale on both Q and K.
            let _scale = crate::place::act_name(scalarmul_scale_tid(scale_idx));
            // ⭐ √scale IS READ OFF THE REGISTRY'S OWN ENTRY — the census pushed `scale.sqrt()`
            // of the config's f32, so the bits the lookup below wants match BY CONSTRUCTION;
            // recomputing from the program's spelling could miss the census's entry by one
            // f16 ulp of double rounding when the slot above resolved through the f16 image.
            let sqrt_scale_val = match layout {
                Some(l) => l.scalarmul_scales[scale_idx].sqrt(),
                None => scale_val.sqrt(),
            };
            let sqrt_scale_idx = layout
                .and_then(|l| {
                    l.scalarmul_scales
                        .iter()
                        .position(|s| s.to_bits() == sqrt_scale_val.to_bits())
                })
                .ok_or_else(|| Error {
                    message: format!(
                        "AttnDecode t{t}: √scale {sqrt_scale_val} absent from scalarmul_scales \
                     (registry desync)"
                    ),
                })?;
            let sqrt_scale = crate::place::act_name(scalarmul_scale_tid(sqrt_scale_idx)); // [1,1] = √attention_multiplier
            let (qs, new_k_scaled) = (n(SynthRole::Qs), n(SynthRole::NewKScaled));
            if let Some(l) = layout {
                l.synth(attn_id.synth(SynthRole::Qs), &[mq, nqh * hd]);
                l.synth(
                    attn_id.synth(SynthRole::NewKScaled),
                    &[mq_pad.row_axis_extent(), nkvh * hd],
                );
            }
            ops.push(assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("attn_qs_o{t}"),
                    op_func: "multiply",
                    // Q is `[mq, nqh*hd]`: the chunk's REAL rows, the full query feature width.
                    rows: crate::sdsc_abstract::RowCount::of_query_rows(
                        crate::sdsc_abstract::QueryRowCount::of_mq(mq),
                    ),
                    cols: crate::sdsc_abstract::BlockCols::of_feature_cols(nqh * hd),
                    // ⭐ TRUE EXTENTS, not `rbo`'s `1×64` annotation: the qs synthetic above reserves
                    // `[mq, nqh*hd]` and Q is the same plane, so both handles state the swept extent —
                    // the mint's bound. The `[1,1]` √scale keeps `rbo`: one RedStick IS its whole
                    // swept window.
                    inputs: &[
                        In::full(&rb(
                            &q,
                            RowCount::of_query_rows(crate::sdsc_abstract::QueryRowCount::of_mq(mq)),
                            BlockCols::of_feature_cols(nqh * hd),
                        ))
                        .ew(),
                        In::scalar(&rbo(&sqrt_scale)).ew(),
                    ],
                    o: &rb(
                        &qs,
                        RowCount::of_query_rows(crate::sdsc_abstract::QueryRowCount::of_mq(mq)),
                        BlockCols::of_feature_cols(nqh * hd),
                    ),
                    out_offset: crate::addr::DevOff::ZERO.into(),
                },
                sym_id_base,
                layout,
            ));
            ops.push(assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("attn_nks_o{t}"),
                    op_func: "multiply",
                    // new-K is allocated `[mq_pad, nkvh*hd]`: the scale covers the PADDED rows, zeros included.
                    rows: crate::sdsc_abstract::RowCount::of_padded_chunk_rows(mq_pad),
                    cols: crate::sdsc_abstract::BlockCols::of_feature_cols(nkvh * hd),
                    inputs: &[
                        In::full(&rb(
                            &new_k_rep,
                            RowCount::of_padded_chunk_rows(mq_pad),
                            BlockCols::of_feature_cols(nkvh * hd),
                        ))
                        .ew(),
                        In::scalar(&rbo(&sqrt_scale)).ew(),
                    ],
                    o: &rb(
                        &new_k_scaled,
                        RowCount::of_padded_chunk_rows(mq_pad),
                        BlockCols::of_feature_cols(nkvh * hd),
                    ),
                    out_offset: crate::addr::DevOff::ZERO.into(),
                },
                sym_id_base,
                layout,
            ));
            (qs, new_k_scaled)
        }
    };

    // ⭐ THE GATHER'S INDEX TENSOR, NAMED BY ITS RESERVED TID'S OWN SPELLING — `act_name` is what the
    // placement, the bind and the descriptor all resolve through, so the tensor the descriptor gathers
    // from is the tensor the worker fills. Under the SAME `gathers_kv` that places it: the reference
    // gathers its pages on every decode, so the one-request rung gathers alongside the batched rungs,
    // while prefill — rows are positions sharing one resident history, nothing per-row for an index
    // to select — keeps the sweep and its byte-identical emission.
    //
    // ⛔⛔⛔ AND UNDER THE **GEOMETRY** TOO, WHICH IS THE OTHER HALF OF THE ONE RULE. `gathers_kv`
    // alone named a tensor that `assemble_attn` may then decline (`GatherScratch::of_fold_pass` refuses a
    // head dim its flat copy cannot express) — so an 8b bundle reserved the index activation and carried a
    // `KvBlockIndex` forward step for a gather that appears in no descriptor. Asking
    // `GatherScratch::admits` HERE and at the placement (`lower_subtile_tape_to_superdsc`) makes the two
    // halves ONE predicate; see that door for why the split was invisible at hd=64.
    // ⭐⭐⭐⭐⭐ THE PAGE-GRANULAR DOOR, ASKED HERE SO THE TENSOR'S NAME AND THE EMISSION ARE ONE
    // PREDICATE — AND A REFUSAL IS A **BUILD FAILURE**, NOT A SILENTLY UNGATHERED BUNDLE.
    //
    // ⛔⛔⛔ IT WAS `.is_some()`, AND THAT IS HOW AN hd=128 BUILD PASSED ITS GATE WHILE GATHERING
    // NOTHING. `PageScratch::of_pass` refused two slabs, this gate turned the refusal into `None`,
    // `assemble_attn`'s `zip` turned `None` into the ungathered bundle, and every test that asked
    // "does the gather work" was answered by a 2b build where the door was open. A runtime refusal that
    // falls back to the form that shipped is indistinguishable from a working feature at the one
    // geometry it is exercised on.
    //
    // ⭐ `expect` IS A COMPILE-TIME FAILURE HERE. `#[forward]` runs this whole pipeline at macro
    // expansion, so a geometry the page gather cannot express now stops the BUILD and names the
    // quantity, instead of emitting a bundle that quietly drops the gather. Every geometry
    // `rows_are_requests` can present is admissible by construction — `mq <= WIDEST_BATCH_RUNG` is the
    // same 32 as `ENTRIES_PER_PASS_MAX`, `hd` is a whole number of sticks for every model, and the entry
    // count is cut to fit both the IBR stick and the LX chunk (`PageScratch::entries_per_op`) — so this
    // panic is unreachable rather than enforced, which is the only shape a panic in the emitter may have.
    //
    // ⛔ THE `KV_BLOCK_INDEX_TID` PLACEMENT RESERVES ON `gathers_kv` ALONE (see
    // `lower_subtile_tape_to_superdsc` for the measured reason it is not narrowed), and with the door
    // now total under that same predicate the two halves are the SAME set of bundles — the hd=128
    // asymmetry that had three doors giving two answers is gone.
    // ⭐⭐⭐⭐⭐ THE SPLIT-KV CHUNK, minted here so a chunk width this geometry cannot express is a BUILD
    // failure naming the quantity — `ChunkScratch::of_chunk` refuses more than one feature slab, more
    // than `PAGES_MAX` pages, or a footprint past `u32`. A chunk is a gathered read, so it demands the
    // same `gathers_kv` the placement reserved the index under: a chunk without it would gather through
    // an index no placement holds.
    // A batch's rows are requests over their own runs of the chunk (`sdsc_abstract::BatchChunk`), which
    // `assemble_attn` mints from this chunk and the width — so the row kind decides the mask's READ, not
    // whether a chunk is admissible.
    // The pool, minted ONCE for the two doors below (the chunk's and the index's) — through the
    // class-provable door, the same spelling the write below mints its own pool from.
    let pool = crate::sdsc_abstract::PagedKvPool::of_geometry(geom);
    let chunk: Option<crate::sdsc_abstract::ChunkScratch> = chunk_pages.map(|pages| {
        assert!(
            gathers_kv,
            "attention {_name}: a split-KV chunk of {pages} page(s) is a gathered read — it needs \
             `gathers_kv` (the index placement); rows_are_requests={rows_are_requests}"
        );
        crate::sdsc_abstract::ChunkScratch::of_chunk(pool, pages).unwrap_or_else(|| {
            panic!(
                "attention {_name}: a split-KV chunk of {pages} page(s) at nkvh={nkvh}, hd={hd} is not \
                 expressible — `ChunkScratch::of_chunk` refuses more than one {stick}-element feature \
                 slab, more than {max} pages, or a footprint past a u32 descriptor extent",
                stick = crate::sdsc_abstract::POOL_STICK,
                max = crate::sdsc_abstract::ChunkScratch::PAGES_MAX,
            )
        })
    });
    let kv_block_index: Option<String> = gathers_kv.then(|| {
        // A chunk's own door was asked above; the per-page door is the batched rungs'.
        if chunk.is_some() {
            return crate::place::act_name(crate::reserved_tids::KV_BLOCK_INDEX_TID);
        }
        crate::sdsc_abstract::PageScratch::of_pass(
            pool,
            crate::sdsc_abstract::QueryRowCount::of_mq(mq),
        )
        .unwrap_or_else(|| {
            panic!(
                "the paged attention bundle for a batch of {mq} request row(s) at nkvh={nkvh}, \
                 hd={hd} cannot express a page-granular KV gather, and a batched-decode bundle has \
                 no ungathered form to fall back to. `PageScratch::of_pass` refuses when hd is not a \
                 whole number of {stick}-element sticks, when the batch is wider than {max} index \
                 entries, when one entry's double-buffered pair does not fit LX, or when the plane's \
                 footprint leaves a u32 descriptor extent. Fix the geometry or widen the door — do \
                 NOT reintroduce a fallback.",
                stick = crate::sdsc_abstract::POOL_STICK,
                max = crate::sdsc_abstract::PageScratch::ENTRIES_PER_PASS_MAX,
            )
        });
        crate::place::act_name(crate::reserved_tids::KV_BLOCK_INDEX_TID)
    });
    // ── (3) the unified score/softmax/output computation — ONE algorithm for any mq (see
    // ir::bridge::tiled_op_sdsc_op::attn's module doc for the torch-spyre correspondence). The head
    // geometry travels as the minted type, not as three integers this call could reorder, and the
    // width travels fused to its row laws on the boundary's one carrier.
    ops.extend(assemble_attn(
        AttnFacts {
            t,
            geom,
            bundle_rows,
            extent,
            qs: &qs,
            new_k_scaled: &new_k_scaled,
            new_v: &new_v_rep,
            kct: &kct,
            vc: &vc,
            pmask: &pmask,
            cmask: &cmask,
            // ⭐⭐⭐⭐⭐ THE GATHER, ON — the prefix score leg's Kᵗ operand takes its KV base from this index
            // tensor instead of from the runtime's per-launch page shift.
            //
            // ⛔ GATED ON `gathers_kv`, WHICH IS THE SAME CONDITION THE PLACEMENT USES. Both halves
            // of a gather have to be decided once: `compute_bundle_layout` places `KV_BLOCK_INDEX_TID`
            // under this predicate, `BakeFacts::gathers_kv` reads that placement, and the forward tape's
            // `KvBlockIndex` step comes from that. The reference gathers on every decode — batched and
            // one-request alike — while prefill shares ONE resident history across every row, so there is
            // nothing per-row for an index to select and its emission stays byte-identical to before this
            // existed.
            //
            // ⛔ AND THE NAME IS THE RESERVED TID'S OWN SPELLING, not a synthetic. `act_name` is what the
            // placement, the bind and the descriptor all resolve through, so the tensor the descriptor
            // gathers from is the tensor the worker fills. A name no placement matches is a bind the
            // launcher SKIPS in silence, and a skipped index reads as zero — block 0, a real address — so
            // every row of the batch would answer from row 0's keys.
            //
            // ══════════════════════════════════════════════════════════════════════════════════════════
            // ⭐⭐⭐⭐⭐ ON — AND WHAT IT IS ATTACHED TO IS THE WHOLE STORY. A VENDOR CONSTRAINT MEASURED ON
            // THE CARD THREE TIMES:
            //
            // **DEEPTOOLS CANNOT SCHEDULE A GATHER ON AN OP THAT HAS A `KERNEL` — i.e. ON A MATMUL.**
            //
            // The bake refuses, and the two refusals bracket the cause exactly (granite-3.1-2b fp8, pod
            // `nickm-7db9667cdd-z2jc6`):
            //   index memOrg_ = hbm+lx  ->  `sbf-ddc: DtException: Expect a valid allocate node.,
            //                                L3DlOpsScheduler.cpp:2337`
            //   index memOrg_ = hbm     ->  `sbf-ddc: DtException: Expect LX in labeledDs memOrg_.,
            //                                L3DlOpsScheduler.cpp:2334`
            // Both lines are in `calculateFlopPerByte`, which for EVERY HBM-pinned labeledDs demands an LX
            // memOrg entry AND an allocated LX node. `allocAllMem` gives one to each STAGED operand and
            // never to an index — an index is read into the IBR, not staged — so the two requirements
            // cannot both be met and there is no third `memOrg_` to try.
            //
            // ⭐ IT ONLY RUNS FOR A REUSE OP, AND THAT IS THE WHOLE ANSWER.
            // `L3DlOpsScheduler.cpp:1550` gates it on `isReuse`, and `hasDimensionReuse` (`:303-321`) is
            // *"`primaryDsInfo_.size() > 1 && primaryDsInfo_.count(DsTypes::KERNEL)`"*. So:
            //   our score op       primaryDsInfo_ {INPUT, KERNEL, OUTPUT, KERNEL_IDX}  -> reuse, EXPLORER RUNS
            //   test_gather_1core  primaryDsInfo_ {OUTPUT, KERNEL_IDX}                 -> no KERNEL, skipped
            //   sdsc_add_paged_l3lu primaryDsInfo_ {OUTPUT, KERNEL_IDX}                -> no KERNEL, skipped
            // Neither vendor fixture gathers on a matmul. `test_gather_1core`'s op is `identity`; IBM's
            // paged attention gathers on `AddZero` — an elementwise copy — and feeds the RESULT to its
            // matmul. `a-gather-is-an-ordinary-compute-op-with-an-index-annotated-operand` says exactly
            // that, and this is the constraint that forces it.
            //
            // ⭐ SO THE SHAPE THAT BAKES IS A SEPARATE GATHER OP, AND THAT IS WHAT `assemble_attn` NOW EMITS:
            // two KERNEL-less `identity` copies that move the paged Kᵗ and V blocks through the index into
            // CONTIGUOUS scratches, with the score and value matmuls reading those scratches per-request and
            // carrying no gather at all. `emit_sdsc` still refuses a gather on a KERNEL-bearing op at build
            // time (`asking_the_shipped_prefix_fold_for_a_gather_is_refused_naming_the_kernel` drives this
            // very call path to prove the refusal reaches it), so the constraint is enforced rather than
            // remembered.
            //
            // ⛔ `Some` IS THE COUPLING, NOT A SWITCH. Passing the NAME is what makes the emitter unable to
            // gather unless a caller has a tensor to name — and the caller that names it is the same one that
            // must place it (`compute_bundle_layout`, under this same `rows_are_requests`) and stage it
            // (`BakeFacts::gathers_kv` → the forward tape's `KvBlockIndex` step). A name no placement matches
            // is a bind the launcher SKIPS in silence, and a skipped index reads as entry 0 — a REAL address,
            // page 0's first block, so every row of the batch would answer from one request's keys.
            //
            // ⛔ AND `assemble_attn` MAY STILL DECLINE IT. `GatherScratch::of_fold_pass` refuses a head dim
            // above one stick, because a 64-slot window of the V plane is then `nslab` runs `PLANE_SLOTS*64`
            // apart and a flat block copy would relayout it. That bundle emits exactly what it emits today
            // rather than a copy that bakes and scrambles the value leg.
            kv_block_index: kv_block_index.as_deref(),
            chunk,
            out_id: attn_id,
            rows_are_requests,
            // The head-major o_proj handoff — the bundle fact the finalize above and the o matmul's
            // B-operand swap key on together. See [`AttnAt::headmajor_handoff`].
            headmajor_handoff,
        },
        sym_id_base,
        layout,
    )?);

    // ── (4) + (5): the KV cache write and the Kᵀ re-transpose — see [`kv_cache_write`].
    ops.extend(kv_cache_write(
        KvCacheWrite {
            t: ops_in.out_id,
            mq: crate::sdsc_abstract::QueryRowCount::of_mq(mq),
            pool: crate::sdsc_abstract::PagedKvPool::of_geometry(geom),
            rows_are_requests,
            // main's own call always wrote the window starting at the cursor's row 0; the
            // rope arm below is the second caller, and it states its node's window.
            row_start: crate::sdsc_abstract::WriteRowStart::at_cursor(),
            // ⭐ THE WRITE'S K, TYPED BY ITS LAW — minted from the same `scale_law` the score
            // legs above lowered under, so the write and the readers cannot disagree about
            // what the pool carries: the multiply arm's `new_k_scaled` is `qs`'s twin (an
            // EMITTED multiply scaled it); the fold's skip arm hands K born scaled out of
            // weights proven to carry √s; a unit scale's bytes are raw with nothing for a
            // reader to state.
            new_k: {
                // The emitted-multiply case first: a program stating a real multiplier put
                // `qs`'s twin on the K, so the pool takes it scaled. `Some(1.0)` is the
                // identity — the multiply above elided it — so only the LAW decides. The
                // match is over `(bool, ScoreScaleLaw)`: guard-free, so exhaustiveness is
                // checked, not asserted. (A `(true, folded)` pair is the half-applied fold
                // the contract region above already refused; if it ever reached here, the
                // multiply arm's bytes really are scaled, which is what the type would say.)
                let multiplied = scale_val.is_some_and(|v| v != 1.0);
                match (multiplied, scale_law) {
                    (true, _) => NewK::sqrt_scaled_by_multiply(&new_k_scaled),
                    (false, ScoreScaleLaw::CommutedIntoWeights) => {
                        NewK::born_scaled_by_fold(&new_k_scaled)
                    }
                    (false, ScoreScaleLaw::UnitByModel | ScoreScaleLaw::StatedByProgram) => {
                        NewK::reader_stated(&new_k_scaled)
                    }
                }
            },
            new_v: &new_v_rep,
            kc: &kc,
            vc: &vc,
            kct: &kct,
            ident: &ident,
        },
        sym_id_base,
        layout,
    )?);
    Ok(ops)
}

/// What one step's KV cache write moves, and between which buffers — everything [`kv_cache_write`]
/// reads off the attention it closes.
pub struct KvCacheWrite<'a> {
    /// The attention's output tensor, which every op's name is suffixed with. ⭐ ITS IDENTITY IS
    /// [`PlaceId`](crate::place::PlaceId), the bake↔host join key — so this field cannot be handed
    /// the row index beside it (`row_start`): a cursor or a slot offset does not name a placed
    /// tensor, and the number exits only through `.tid()`.
    pub t: crate::place::PlaceId,
    /// The step's query rows: one decode token, a batch's requests, or a prompt chunk's
    /// positions — the one-mint count the scratch doors take, so it cannot be reordered
    /// against the geometry below.
    pub mq: crate::sdsc_abstract::QueryRowCount,
    /// The pool this write lands in — the geometry as the seam's minted type, not two
    /// integers this call could reorder against each other or against the rows.
    pub pool: crate::sdsc_abstract::PagedKvPool,
    /// The rows are a decode batch's requests (one op per request), not one sequence's positions.
    pub rows_are_requests: bool,
    /// ⭐ THE WINDOW'S FIRST ROW, cursor-relative — [`WriteRowStart`], minted at the cursor
    /// (main's `attn_at` call, which always passed the whole activation plane,
    /// byte-identically) or from a node's own window (the rope append arm, a per-row node of a
    /// batch, whose rows share one output tensor and whose request is the row itself).
    ///
    /// [`WriteRowStart`]: crate::sdsc_abstract::WriteRowStart
    pub row_start: crate::sdsc_abstract::WriteRowStart,
    /// This step's new K and V, `[mq, nkvh·hd]` row-blocked. ⭐ THE SCALE CONTRACT IS THE TYPE:
    /// `new_k` is [`NewK`], minted by one of three proof-named constructors, so the two callers
    /// with OPPOSITE laws — the spliced append's RAW rotated K (the fp8 k_proj's fold is refused
    /// there, so every pool reader states its own scale) and this path's `√s`-carrying K (an
    /// emitted multiply, or a fold's born-scaled staging, whose future prefix scores state
    /// nothing) — cannot hand the recipe the same bare `&str` and leave the difference to a
    /// comment. The recipe writes whichever bytes it is given; WHICH law they carry is what the
    /// argument says.
    pub new_k: NewK<'a>,
    pub new_v: &'a str,
    /// The resident natural-K, V and Kᵀ planes.
    pub kc: &'a str,
    pub vc: &'a str,
    pub kct: &'a str,
    /// The `[64, 64]` identity the copies contract against.
    pub ident: &'a str,
}

/// ── (4) KV CACHE WRITE: append this step's new K (nkvh-wide, dedup)/new_v_rep (nqh-wide)
/// into the resident cache, then (5) re-transpose the Kᵀ plane from it. K writes only the nkvh
/// representative slots (kvh*gqa) — `kctpost` (below) only ever reads those; the other slots were
/// always dead. V is UNCHANGED (nqh writes, genuinely all read via `vc`'s own nqh-replicated
/// convention). ⛔ WHAT THE K CARRIES IS THE ARGUMENT'S TYPE, NOT THIS COMMENT: [`NewK`] states
/// whether these bytes carry √s (an emitted multiply, or a fold's born-scaled staging — future
/// steps' prefix scores then state no multiplier) or are raw (every reader states the whole
/// scale in its own program).
///
/// REQUEST-MAJOR, so that one request's kv-head copies are CONSECUTIVE and fuse into a
/// single launch. A group resolves one request's page table and write cursor, so the run
/// breaks where the request changes (`GroupKind::Slot`'s `req`) — kv-head-major order would
/// alternate requests every op and leave every copy its own launch, which is what made a
/// batch of 8 slower than 8 separate forwards. One request ⇒ this loop runs once and the
/// order is exactly what it was.
///
/// Its own function, public, so the emitted parts of a wide prompt window are pinned off the op list
/// this very code returns rather than off a copy of it.
pub fn kv_cache_write(
    w: KvCacheWrite<'_>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let KvCacheWrite {
        t,
        mq,
        pool,
        rows_are_requests,
        row_start,
        new_k,
        new_v: new_v_rep,
        kc,
        vc,
        kct,
        ident,
    } = w;
    let stick = Fp16::ELEMS_PER_STICK;
    // The identity, rows and geometry, spent through their doors — the values are the ones the
    // caller minted, and the body below reads them exactly as it did.
    let t = t.tid();
    // The window's row count rides typed to the cutting law below; the number the body
    // reads is spent here.
    let mq_rows = mq;
    let mq = mq.get();
    let nkvh = pool.nkvh();
    let hd = pool.hd();
    let mut ops: Vec<EmittedOp> = Vec::new();
    let per_request = rows_are_requests && mq > 1;
    // ⛔⛔⛔ AND THE SLAB CANNOT JOIN THAT `y` AXIS — MEASURED ON CARD, NOT INFERRED. The pair
    // `(h, s)` linearizes as `h*nslab + s`, and that index steps exactly ONE STICK PLANE on BOTH
    // operands, so ONE `y` of `nkvh*nslab` should cover both loops. It passes every check this backend
    // has — the builder's y-stride check at hd 64/128/256/512, a conservation law differenced out of
    // both nests, two Kani proofs of coverage + injectivity — and granite-2b (hd=64, where nslab==1
    // makes the two forms identical) stays coherent 10/10. On granite-8b at hd=128 it produces GARBAGE
    // from the first token ("The is a 1", 10/10 runs). So whatever makes a `y` step cross a feature
    // slab is NOT the address arithmetic; do not retry the pair walk on the strength of the
    // arithmetic, which is correct and insufficient.
    let src_nest = crate::addr::Nest::new(&["row", "head", "feat"], &[mq, nkvh, hd], Df::Fp16);
    // ⛔ `PLANE_SLOTS`, NOT `PAGE_SLOTS`, AND LOAD-BEARING AT EVERY HEAD DIM — this extent IS the
    // write's kv-head stride (the `y` step is derived from it), and every READER takes its head stride
    // from `PagedKvPool::plane_block_elems()`, which is physical. Declaring the ADDRESSABLE page while
    // the pool strides by the PHYSICAL plane splits one law into two: writes land at `kvh*PAGE_SLOTS*hd`
    // and reads at `kvh*PLANE_SLOTS*hd`, so every kv head but 0 reads keys nobody wrote. MEASURED: that
    // regressed granite-2b (hd=64) from coherent to garbled, which is how it was found.
    let dst_nest = crate::addr::Nest::new(
        &["slot", "feat"],
        &[crate::sdsc_abstract::PagedKvPool::PLANE_SLOTS as u32, hd],
        Df::Fp16,
    );
    // ASKED OF THE SHAPE, once: the slab count reaches this write only through `slabs_of`, so there is
    // no `hd/64` in scope for the loop and the pitches to spell differently.
    let slabs = crate::addr::Shape::<0, 0, 0, 0>::slabs_of(hd, Df::Fp16);
    // ⭐⭐⭐ A PROMPT WINDOW WIDER THAN A PAGE IS WRITTEN IN PAGE PARTS.
    //
    // One launch resolves ONE page per group (the page holding the group's cursor), and a page plane is
    // `PAGE_SLOTS` slots of ONE kv head — so a single `[mq, hd]` copy of more rows than that would run off
    // the page into the NEXT KV HEAD's block. Instead each `(kv head, slab)` copy is cut into parts of
    // `PAGE_SLOTS` rows: part `p` reads the chunk's rows from `p·PAGE_SLOTS` and writes the SAME baked
    // destination, carrying its `PagePart` off the cut so the runtime aims it at the page that slot
    // falls in (`fold_plan::op_slot_pos`), and skips it when the request holds no such page (a short
    // last chunk's padding). A window is a whole number of pages (`PREFILL_CHUNK_SLOTS`'s module law)
    // and starts at a page boundary (`chunk_write_start`), so every part fills its page from slot 0.
    //
    // PART-MAJOR, so a part's copies are consecutive and fuse into one launch — the same reason the
    // loop is request-major. A window that fits a page is ONE part with offset 0, and every name, row
    // and op is exactly what it was.
    //
    // ⭐ THE CUTTING GOES THROUGH THE ONE LAW ([`PagedKvPool::write_parts`]) — the same
    // function the whole splice/door boundary shares, re-homed here from main's inline
    // `parts = mq.div_ceil(page)` loop. At `row_start = 0` (main's own call, and a
    // whole-plane node) the cut is BYTE-IDENTICAL to that loop: request-major one part per
    // row, position-major one part per page with `p·page` offsets.
    let parts = pool.write_parts(row_start, mq_rows, rows_are_requests);
    let part_tag = |p: usize| {
        if !rows_are_requests && parts.len() > 1 {
            format!("_p{p}")
        } else {
            String::new()
        }
    };
    for (p, part) in parts.iter().enumerate() {
        let (row0, part_rows, req) = (part.row0, part.rows, part.request);
        // ⛔ NO REQUEST TERM IN THE ADDRESS. `req` here indexes a ROW OF THE ACTIVATION and nothing more:
        // which page that row's token lands in is the host's page map, applied as a per-launch shift. A page
        // holds slots, not requests, so the address inside it is `(kv head, slot, feature)`.
        //
        // ⭐ THE `_r{req}` TAG COVERS THE PER-ROW NODE TOO: a request-major window cut by
        // `write_parts` is one part per row, and at `mq == 1` (a per-row node of a batch,
        // whose rows share ONE output tensor) `per_request` is false but the names still
        // need the row — main's whole-plane call never had two nodes over one tensor, the
        // splice's per-row decomposition does.
        let rq = if per_request || (rows_are_requests && row_start.starts_past_cursor()) {
            format!("_r{req}")
        } else {
            String::new()
        };
        // ⛔⛔⛔ THE HEADS GO ON `y` ONLY WHEN THIS OP COVERS ONE ROW, and `rows_per_op` is that test.
        //
        // The plane-walk placement declares `pitch = rows * nslab`, which describes head-outermost planes of
        // `rows` rows. The source is row-outermost — head `h` of row `r` is at `r*(nkvh*hd) + h*hd`, heads
        // INTERLEAVED inside a row — so the two readings agree only at one row, where the row axis is
        // degenerate. Decode qualifies both ways: solo decode has `mq == 1`, and a batched decode step emits
        // one op per request (`per_request`), each covering its single row. A PREFILL CHUNK does not: one op
        // spans `mq` rows, and `y` then strides heads as if each owned `mq` consecutive rows.
        //
        // MEASURED, granite-2b (hd=64, nslab=1) with the collapse ungated: every prompt from 400 characters
        // up decodes fluent garbage (`'(\x03d. .exaggeret'` for a prompt whose answer is `Paris`).
        let rows_per_op = if per_request { 1 } else { mq };
        let heads_on_y = rows_per_op == 1;
        {
            let pt = part_tag(p);
            for s in 0..slabs.get() {
                for h in 0..if heads_on_y { 1 } else { nkvh } {
                    let a_place = if heads_on_y {
                        crate::sdsc_abstract::OperandPlacement::of_plane_walk_by_head(
                            &src_nest,
                            crate::addr::Idx::<crate::addr::Row>::n(row0),
                            s,
                            slabs,
                        )
                    } else {
                        crate::sdsc_abstract::OperandPlacement::at_head(
                            &src_nest,
                            crate::addr::Idx::<crate::addr::Row>::n(row0),
                            h,
                            s,
                            slabs,
                        )
                    };
                    let o_place = if heads_on_y {
                        crate::sdsc_abstract::OperandPlacement::of_plane_walk_by_head_block_base(
                            &dst_nest, s, slabs,
                        )
                    } else {
                        crate::sdsc_abstract::OperandPlacement::at_block(
                            &dst_nest,
                            crate::sdsc_abstract::PagedKvPool::block_of(kv_head_of(h, nkvh)?),
                            s,
                            slabs,
                        )
                    };
                    // K writes only the nkvh representative slots (`kctpost` only ever reads those); V is the
                    // nkvh-wide `new_v_rep`, whose reader is kv-head indexed too. Both sides of both writes are
                    // therefore kv-head indexed, which is what lets ONE `y` axis serve them.
                    for (base, src, dst) in [("kc", new_k.name(), kc), ("vc", new_v_rep, vc)] {
                        ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                        &if heads_on_y {
                            format!("cachewr_{base}s{s}{rq}{pt}_o{t}")
                        } else {
                            format!("cachewr_{base}{h}s{s}{rq}{pt}_o{t}")
                        },
                        // The kv stream's REAL rows this copy covers: one per request-copy, else the part's.
                        crate::sdsc_abstract::MatM::of_query_rows(
                            crate::sdsc_abstract::QueryRowCount::of_mq(part_rows),
                        ),
                        crate::sdsc_abstract::MatN::one_stick(crate::sdsc_abstract::Lanes::FP16),
                        crate::sdsc_abstract::MatK::one_stick(crate::sdsc_abstract::Lanes::FP16),
                        // `y` WALKS THE KV HEADS OF THIS SLAB — the same door the score leg's GQA group uses, and
                        // for the same reason: the batch axis carries BOTH operands' head strides, so it cannot be
                        // asked for without stating where the heads are, and the builder refuses a walk that
                        // strides by neither.
                        if heads_on_y {
                            crate::sdsc_abstract::MatY::of_gqa_group(nkvh, a_place, o_place)
                        } else {
                            // The head is in both offsets, so there is no batch axis to stride — and nothing for the
                            // builder's stride check to be wrong about.
                            crate::sdsc_abstract::MatY::unbatched()
                        },
                        crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::of_kv_cache_write(
                        ),
                        &rb(
                            src,
                            RowCount::of_query_rows(
                                crate::sdsc_abstract::QueryRowCount::of_mq(mq),
                            ),
                            BlockCols::of_feature_cols(nkvh * hd),
                        ),
                        a_place,
                        // I[64,64] — the identity for EVERY slab, so the shared 2-D kernel a `y` batch demands is
                        // exactly what an identity copy needs. Offset 0: see the residency argument above.
                        &Stk::<KernelTag>::one_stick_identity(ident),
                        crate::addr::DevOff::ZERO,
                        &Stk::<KernelTag>::resident_kv_plane(hd, dst),
                        o_place,
                        sym_id_base,
                        layout,
                    ));
                        if let Some(o) = ops.last_mut() {
                            o.slot_stride_bytes = stick * 2;
                            // PAGED: the slot is the position WITHIN its page; the page comes from the block table.
                            o.kv_page_slots = crate::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32;
                            // ⭐ TAGGED WITH ITS ROW, and it has to be. Untagged, this op took the page from the
                            // LAUNCH's position — fine while the address carried a request term, because then the row
                            // was already in the address. That term is gone: a page holds slots, so the ONLY thing that
                            // says which page this row's token lands in is the tag, which `nonfold_page_delta` turns
                            // into that row's page through the host's block table.
                            o.kv_request = req;
                            // ⭐ THE PART'S OWN PAGE, off the cut — `write_parts` minted it beside the
                            // rows; this stamp COPIES it, never re-derives it (the `if rows_are_requests`
                            // that used to live here WAS the re-derivation, and it lived twice).
                            o.kv_part = part.page;
                        }
                    }
                }
            }
        }
    }

    // ── (5) RE-TRANSPOSE the resident Kᵀ cache from the natural kc the cachewr just wrote, so NEXT
    // step's prefix score reads this step's new K too (hd==stick only — guarded above).
    //
    // ONE RE-TRANSPOSE PER (REQUEST, KV-HEAD) — not one for the whole batch.
    //
    // This loop used to run over kv-heads only, so a batch of 8 re-transposed ONE page: the one
    // `pool.write_base(kvh, hd)` names, which after the shim's per-op shift is request 0's. Every
    // other request's freshly-written K therefore never reached `kct`, so its NEXT step scored the
    // prefix against a stale Kᵀ.
    //
    // AND THE REQUEST IS THE ADDRESS, NOT A TAG — so the batch's re-transposes are ONE launch.
    //
    // Flat product loop (not nested) so the body below keeps the indentation — and the shape — of the
    // whole-page form it is otherwise unchanged from.
    //
    // ⭐ AND ONE PER PAGE PART, PART-MAJOR — the window's second page holds natural K the cache write
    // just wrote, and its Kᵀ is as stale as the first page's was. A whole-page re-transpose is already
    // the per-page form, so a part is the same op aimed `p` pages on (its `PagePart`), skipped with
    // its cache write when the request holds no such page.
    //
    // ⭐ THE PARTS CARRY THE REQUESTS: at `row_start = 0` this is main's own
    // `(p, req, kvh)` flat loop exactly — request-major parts are one per row (main's
    // `kct_reqs = mq` at one part), position-major parts are one per page at `req = 0` —
    // so the op sequence and every name are what they were.
    for (p, part) in parts.iter().enumerate() {
        let req = part.request;
        for kvh in 0..nkvh {
            let rq = if per_request || (rows_are_requests && row_start.starts_past_cursor()) {
                format!("_r{req}")
            } else {
                String::new()
            };
            let pt = part_tag(p);
            // WHOLE-PAGE re-transpose, structurally the proven full-`cap` form with the page (equal to
            // that cap) substituted — NOT the slab-granular variant, whose on-card attempt regressed
            // decode. Source and destination take the identical `layer + page` shift, which is why
            // natural K lives in the page rather than a segment of its own.
            ops.push(assemble_restickify_kt_2d(
                &format!("attn_kctpost{kvh}{rq}{pt}_o{t}"),
                crate::sdsc_abstract::KtTileSlots::of_page(),
                crate::sdsc_abstract::KtTileFeats::of_head_dim(hd),
                // ⭐ BOTH HANDLES AT THE PLANE'S OWN LAW — `pool.addr` is PLANE-RELATIVE (each plane is
                // its own named tensor; the plane base is implicit in which name an op takes), so the
                // bound the mint states is the plane's own `plane_stride()` — the one number that sizes
                // the pool's placement, stated by the same function the offsets come from. The window is
                // ONE block (`[PAGE_SLOTS, hd]`), which the pool's own doctest pins inside the stride
                // for every head the block table can name; a kv head past `nkvh` now refuses HERE
                // instead of re-transposing another plane's bytes into Kᵀ.
                &rb(
                    kc,
                    RowCount::of_broadcast_row(),
                    BlockCols::of_feature_cols(pool.plane_stride() as u32),
                ),
                // The two planes are the whole content of this op: it reads natural K and writes Kᵀ, both at
                // the same `(kv head)` block of the page the launch was shifted to.
                crate::addr::DevOff::from_view_step(pool.addr(
                    crate::sdsc_abstract::KvCoord::block(
                        crate::sdsc_abstract::KvPlane::Knat,
                        kv_head_of(kvh, nkvh)?,
                    ),
                )),
                &rb(
                    kct,
                    RowCount::of_broadcast_row(),
                    BlockCols::of_feature_cols(pool.plane_stride() as u32),
                ),
                crate::addr::DevOff::from_view_step(pool.addr(
                    crate::sdsc_abstract::KvCoord::block(
                        crate::sdsc_abstract::KvPlane::Kt,
                        kv_head_of(kvh, nkvh)?,
                    ),
                )),
                sym_id_base,
                layout,
            ));
            // ⭐ TAGGED WITH ITS ROW, for the same reason the cache writes are: this op reads natural K and
            // writes Kᵀ within ONE row's page, and with no request term left in the address the tag is the only
            // thing that tells the runtime which page that is.
            if let Some(o) = ops.last_mut() {
                o.kv_request = req;
                // The part's own page, off the cut — the copies' law (main's `p·page` at
                // `row_start = 0`), copied from the same `write_parts` row that carried `row0`.
                o.kv_part = part.page;
            }
        }
    }

    Ok(ops)
}

/// main's PREFILL LAST-ROW EXTRACTION — the first half of `lower_prefill_lm_head_at_m1`
/// (main `lower_subtile_tape_to_superdsc.rs:10496-10534`), verbatim below its door.
///
/// `hidden/64` single-stick `[1,64]` identity copies lift row `selector_lastrow_col(mq)` of the
/// `[mq, hidden]` final-norm output into the `[1, hidden]` synthetic
/// [`LAST_HIDDEN_TID`](crate::reserved_tids::LAST_HIDDEN_TID), so the vocab-wide lm-head tail can run
/// at `m=1` over a tensor whose row 0 IS the last prompt token.
///
/// ⛔⛔⛔ THIS IS THE OP THE PORT DROPPED, AND ITS ABSENCE WAS A WRONG FIRST TOKEN ON EVERY PROMPT.
/// The KTIR arm replaced these copies with a one-line row slice on the matmul's activation region,
/// on the reasoning that "a KTIR view STATES its start address … so row `mq-1` of `[mq, hidden]` is an
/// address this target can simply name". The KTIR can name it — the matmul emits the corner as
/// an `arith.constant` — but [`regions`] discarded it and `assemble_matmul_seeded` has nowhere to put
/// it, so the tail read the buffer base. MEASURED against main on the same card, same ladder rung,
/// same `PREFILL_PATH` line (`m_used=14 prefill_m=15 rung=2/21`): ours `yun! How can`, main
/// `Hello! How can`.
///
/// ⭐ WHY A COPY AND NOT A ROW OFFSET, which is the thing worth not re-litigating: `hidden[mq, hidden]`
/// is `RowBlocked`, so logical `(r, c)` sits at `(c/64)·(mq·64) + r·64 + (c%64)` — row `r`'s stick-group
/// `j` is 64 CONTIGUOUS elements at `(j·mq + r)·64`. Read as the tall-narrow `[(hidden/64)·mq, 64]` it
/// physically is, that group is exactly ROW `j·mq + r` of a single-stick tensor, and a row offset there
/// is the ONE coordInfo form fp16 can represent (`StickLayout::group_stride`: a `rows·lanes` stride is
/// not, and a one-hot `sel[1,mq] @ hidden` needs `k = mq` to be a whole 64-stick, which no rung is).
/// So both ends walk at `lanes`, which is why this is `hidden/64` ops and not one.
///
/// ⛔ AND THE COPIES ARE NAMED AFTER THE NODE, NOT AFTER WHAT THEY WRITE. main's name is
/// `lmlast{j}_o{node.output.tensor}` — the logits tid its own tail matmul is named for — while this
/// program's only output is the reserved staging, so naming from `out.tid` gave every model
/// `lmlast{j}_o4294967295`. The node's tensor arrives on [`KtirNode::node_out_tid`], stated by the
/// producer that built both halves; see that field for why the program cannot state it.
pub fn lmlast(
    name: &str,
    k: &KtirNode,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let owner = k.node_out_tid.ok_or_else(|| Error {
        message: format!(
            "{name}: states no `node_out_tid`. The last-row extraction is HALF of an lm-head node's \
             lowering and main names its copies after that node's output tensor \
             (`lmlast{{j}}_o{{tid}}`, the same suffix as the tail's own `matmul_o{{tid}}`), which \
             this program cannot state — it never addresses the logits buffer. The producer that \
             built both halves must state it."
        ),
    })?;
    let outs: Vec<Region> = r.iter().copied().filter(|x| x.is_out).collect();
    let [out] = outs[..] else {
        return err(format!(
            "{name}: {} parameter(s) written by a `ktdp.store` — the extraction writes exactly one",
            outs.len()
        ));
    };
    let ins: Vec<Region> = r.iter().copied().filter(|x| !x.is_out).collect();
    let [src] = ins[..] else {
        return err(format!(
            "{name}: {} tensor input(s) — the extraction reads exactly one",
            ins.len()
        ));
    };
    if out.tid != LAST_HIDDEN_TID {
        return err(format!(
            "{name}: writes t{} — the last-row extraction's destination is the reserved \
             `LAST_HIDDEN_TID` synthetic, which is what the re-lowered lm-head tail reads",
            out.tid
        ));
    }
    // `mq` and the row come from the SOURCE's own view + window: the buffer is `[mq, hidden]` and the
    // program takes ONE row of it, at `r_start`.
    let (mq, hidden) = (src.v_rows, src.v_cols);
    let row = src.r_start;
    let stk = Fp16::ELEMS_PER_STICK;
    if hidden % stk != 0 {
        return err(format!(
            "{name}: hidden={hidden} is not a whole {stk}-fp16 stick, so the last prompt row is not \
             a run of whole stick-groups — the per-stick extraction cannot address it"
        ));
    }
    // The WINDOW is one SINGLE STICK — `regions` reports the first access tile, and the extraction's
    // tiles are `[1, stk]` by construction (that is the whole point: one stick-group per op). What must
    // be whole-row is the DESTINATION BUFFER, `[1, hidden]`.
    if src.r_len != 1 || src.c_len != stk || out.r_len != 1 || out.c_len != stk {
        return err(format!(
            "{name}: takes `[{}, {}]` into `[{}, {}]` — each copy is one `[1, {stk}]` stick-group, \
             which is the only form a row offset is representable in",
            src.r_len, src.c_len, out.r_len, out.c_len,
        ));
    }
    if out.v_rows != 1 || out.v_cols != hidden {
        return err(format!(
            "{name}: destination buffer is `[{}, {}]` — the last-row staging is `[1, hidden]` for the \
             `[{mq}, {hidden}]` source it is extracted from",
            out.v_rows, out.v_cols,
        ));
    }
    if row >= mq {
        return err(format!(
            "{name}: row {row} is outside the `[{mq}, {hidden}]` source — the row is \
             `selector_lastrow_col(mq)`, which is `mq - 1`"
        ));
    }
    if let Some(l) = layout {
        l.synth(PlaceId::Act(out.tid), &[1, hidden]);
    }
    // ⭐ TRUE EXTENTS, not the `[1, stk]`/`rbo` staging annotations: the source IS the `[mq, hidden]`
    // prompt buffer and the destination IS the `[1, hidden]` staging the synth above reserves, so the
    // handles state the buffers the corners below address — the mint's bound. The stick walk's last
    // window ends exactly at `mq*hidden` / `hidden` elements respectively.
    let src_name = rb(
        &src.name(),
        RowCount::of_token_rows(mq),
        BlockCols::of_feature_cols(hidden),
    );
    let dst = rb(
        &out.name(),
        RowCount::of_token_rows(1),
        BlockCols::of_feature_cols(hidden),
    );
    Ok((0..hidden / stk)
        .map(|j| {
            assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("lmlast{j}_o{owner}"),
                    op_func: "identity",
                    rows: crate::sdsc_abstract::RowCount::of_token_rows(1),
                    cols: crate::sdsc_abstract::BlockCols::of_one_stick(
                        crate::sdsc_abstract::Lanes::FP16,
                    ),
                    inputs: &[In::sliced(
                        &src_name,
                        // row `j*mq + row` of a `[.., stk]` staging — a corner, not a product.
                        crate::addr::rc_of((j + 1) * mq + row + 1, stk, j * mq + row, 0, Df::Fp16),
                    )
                    .ew()],
                    o: &dst,
                    // Column `j*stk` of the single-row `[1, hidden]` last-hidden staging — a corner on
                    // that tensor, not a product. At one row the stick plane term is 0, so this is the
                    // same address either way; saying it as a coordinate keeps the last hand-built offset
                    // in this file from being the one nobody notices.
                    out_offset: crate::addr::rc_of(1, hidden, 0, j * stk, Df::Fp16).into(),
                },
                sym_id_base,
                layout,
            )
        })
        .collect())
}

/// main's `lower_matmul_node` (main 7752-8362).
///
/// Its door: `m`/`k`/`n` came from a `TileOp` built off the node and now come from the program's own
/// tiles — `m`/`n` are the stored output tile's extents and `k` the activation tile's width, which is
/// exactly what `node_to_tile_ops`' MatmulTile arm read (`a.region.cols.len` / `output.region.*`).
///
/// ⭐ ARITY IS THE PRECISION, and the fp8 half of main's body is already ported: three tensor inputs
/// (activation, packed fp8 weight, the checkpoint's per-column `w_scale`) is W8A8 and goes to
/// [`super::ktir_matmul_fp8`], which IS main's arity-3 branch unchanged.
/// ⭐ UNCHANGED FOR EVERY EXISTING CALLER, deliberately. `crates/targets/spyre`'s own
/// `ktir_superdsc_door` is the caller that states the head-major fact; its weights are transpose-B
/// by construction (main's `[k, n]` SubtileIR region viewed as `[n, k]`) — which is exactly the
/// reading the guard inside hardcoded before the orientation became a parameter. So this delegates
/// at `TransposeB` and preserves today's behaviour byte-for-byte; only a producer that CAN prove
/// the orientation is asked to.
pub fn matmul(
    name: &str,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
    headmajor: Option<HeadmajorOproj>,
) -> Result<Vec<EmittedOp>, Error> {
    matmul_oriented(
        name,
        r,
        sym_id_base,
        layout,
        quantized,
        MatmulProof {
            b: super::whole_function::BOrient::TransposeB,
            origin: OperandOrigin::Staged,
            headmajor,
        },
    )
}

/// WHO THE OPERANDS OF THIS MATMUL CALL ARE — the typed discriminator the spurious-pad drop in
/// [`matmul_oriented`] reads (issue 201 item 7: a bool cannot say why).
pub enum OperandOrigin {
    /// Staged whole tensors: the per-`Program` door. The staged-buffer contract behind
    /// `DeviceWidth::for_output`'s bump holds — the worker's weight zero-pad makes the pad real.
    Staged,
    /// One WINDOW of the caller's parameters: the whole-function door. The padded weight columns
    /// would be read past the window's end, so [`DeviceWidth::for_matmul`] drops the util-floor
    /// bump when the logical width already meets the floor.
    Windowed,
}

/// ⭐ THE HEAD-MAJOR o_proj HANDOFF, for the matmul door — the caller-side twin of the spyre
/// door's `BundleAttnParams` head-major fact, carrying the ONE thing this side keys the B-operand
/// swap on: the o matmul's weight tid, named by the bundle facts (dataflow-found, see that
/// struct's own doc), never by a shape. `None` for every producer whose caller states no
/// handoff — those bake exactly what they baked before this existed.
#[derive(Clone, Copy)]
pub struct HeadmajorOproj {
    /// The o matmul's weight tid, as the bundle's canonical (layer-0) o_proj.
    pub oproj_wtid: u32,
}

/// THE CALLER-PROVEN FACTS a matmul emission keys on — everything the door knows that the
/// program's own tiles cannot state, bundled so the door states each fact exactly once: the weight
/// orientation ([`super::whole_function::BOrient`]), who the operands are ([`OperandOrigin`]), and
/// the head-major o_proj handoff ([`HeadmajorOproj`] — `None` wherever no bundle states one).
pub struct MatmulProof {
    pub b: super::whole_function::BOrient,
    pub origin: OperandOrigin,
    pub headmajor: Option<HeadmajorOproj>,
}

/// [`matmul`] with the weight orientation PROVEN by the caller instead of assumed — see
/// [`super::whole_function::matmul_b_orientation`], which reads it off the op's own `indexing_maps`.
pub fn matmul_oriented(
    name: &str,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
    proof: MatmulProof,
) -> Result<Vec<EmittedOp>, Error> {
    let outs: Vec<Region> = r.iter().copied().filter(|x| x.is_out).collect();
    let [out] = outs[..] else {
        return err(format!(
            "{name}: {} parameter(s) written by a `ktdp.store` — a matmul writes exactly one",
            outs.len()
        ));
    };
    let ins: Vec<Region> = r
        .iter()
        .copied()
        .filter(|x| !x.is_out && scale_idx_of(layout, x.tid).is_none())
        .collect();
    if ins.len() != 2 && ins.len() != 3 {
        return err(format!(
            "MatmulTile {name} expects 2 inputs (A, W) or 3 (A, W_fp8, w_scale for fp8 W8A8), \
             found {}",
            ins.len()
        ));
    }
    let a = ins[0];
    let w = ins[1];
    // ⛔ THE ACTIVATION IS READ AT THE BASE, so a stated row corner has nowhere to go. This is the
    // exact operand whose corner was discarded: `rb(&a.name(), m, k)` below addresses the tensor, and
    // the mq>1 lm-head tail used to arrive here with `r_start = mq-1`. It now arrives with the row
    // ALREADY MATERIALIZED (`lmlast`), so this refusal is the seal on that, for every arch.
    base_addressed(name, &a, "activation (A)")?;
    // ⛔ THE WEIGHT'S AND OUTPUT'S CORNERS ARE CARRIED, NOT DROPPED — the windowing arm below
    // states each as an `offset_elems` on the operand's own address law, so a corner this emitter
    // cannot carry is the one the LAW cannot address, and each is refused by name below (with the
    // corner computation, where the orientation has said which axis is which):
    //   * W's OUT corner (the kernel steps out in whole sticks — `(n/stk)·(k_phys·stk) + …`) must
    //     be stick-aligned; its IN corner steps `k·stk` and needs no alignment.
    //   * OUT: a row corner is refused exactly as A's (the output group stride derives from the
    //     swept `m`, which IS the tensor's rows); a column corner must be a whole stick.
    //   * A: a column corner must be a whole stick (its row corner is refused outright above).
    base_addressed(name, &out, "output")?;
    let rcol = |r: &Region, role: &str| -> Result<(), Error> {
        if r.c_start != 0 && !r.c_start.is_multiple_of(FP16_ELEMS_PER_STICK) {
            return err(format!(
                "{name}: {role} t{} states column corner {} of its `[{}, {}]` view, which is not \
                 a whole {FP16_ELEMS_PER_STICK}-column stick — the stick-blocked address law steps \
                 whole sticks (`(c/64)·(rows·64) + …`), so a sub-stick column corner has no \
                 address. Pad the blocking so every window corner is stick-aligned.",
                r.tid, r.c_start, r.v_rows, r.v_cols,
            ));
        }
        Ok(())
    };
    rcol(&a, "activation (A)")?;
    rcol(&out, "output")?;
    // ⭐ THE PRECISION IS THE WEIGHT VIEW'S ELEMENT TYPE, and arity agrees with it by construction:
    // `KtirFunc::matmul_fp8` was the only builder that viewed a weight through `view_fp8`, and it is
    // the only one that binds a third input (the checkpoint's per-column `w_scale`). Disagreement
    // between the two is a malformed program and says so rather than picking one.
    if w.is_fp8 != (ins.len() == 3) {
        return err(format!(
            "MatmulTile {name}: the weight view declares {} but the program binds {} input(s) — \
             a packed fp8 weight comes with its per-column `w_scale` and an fp16 one does not",
            if w.is_fp8 { "Fp8E4m3" } else { "f16" },
            ins.len()
        ));
    }
    // m/k/n as the program states them: the stored output tile is `[m, n]` and the activation tile
    // `[m, k]`.
    let (m, n) = (out.r_len, out.c_len);
    let k = a.c_len;
    // ⭐⭐⭐ THE HEAD-MAJOR o_proj SWAP — the matmul half of the restructure whose finalize half lives
    // in the attention door (`assemble_attn`'s `headmajor_handoff` form). Where the bundle's
    // attention finalize writes `out` SLAB-MAJOR (one op per feature slab of the head dim), the
    // contraction axis of THIS matmul arrives permuted — head-major `s·nqh·64 + h·64 + d` — so its
    // B operand must be the second, k-shuffled o_proj copy staged under `oproj_headmajor_tid`, the
    // copy whose retile walk enumerates exactly that order. The fact and the copy are minted
    // together in `compute_bundle_layout`; this door only NAMES the operand, which is why the key
    // is the WEIGHT TID the bundle facts stated by dataflow (see `HeadmajorOproj`) — never a shape:
    // at `k == n` (granite-3.1-8b's `[4096, 4096]` o_proj) the extents cannot tell the o matmul
    // from any other square projection, and a shape-keyed swap would re-point every square
    // projection at the permuted copy.
    //
    // ⛔ AND THE WIDTH AND PRECISION ARE THE HANDOFF'S OWN. The fact was minted from the bundle's
    // attention-output row count and the fp8 W8A8 o matmul the dataflow walk found; `m` here and
    // `w.is_fp8` here are that same pair stated by the program. A swap at any other width or
    // precision reads slab-major bytes against an unpermuted weight (or the reverse) — half a
    // restructure, fluent and wrong — so a disagreement is refused here rather than baked.
    let w_name = match proof.headmajor {
        Some(h) if h.oproj_wtid == w.tid => {
            if m != 1 || !w.is_fp8 {
                return err(format!(
                    "MatmulTile {name}: the bundle states the head-major o_proj handoff for weight \
                     t{}, but this program states m={m} over an {} weight — the handoff is minted \
                     for the single-row fp8 W8A8 o matmul (mq == 1 over a multi-slab head dim). \
                     Half a restructure is a slab-major activation read against an unpermuted \
                     weight: fix the bundle's statement, not the emission.",
                    w.tid,
                    if w.is_fp8 { "fp8" } else { "f16" },
                ));
            }
            crate::place::act_name(crate::reserved_tids::oproj_headmajor_tid(w.tid))
        }
        _ => w.name(),
    };
    // ── fp8 W8A8 (arity-3 [act(fp16 [m,k]), W(fp8 [k,n]), w_scale(f16 [n,1] ≡ [1,n])]): per-token quantize
    //    act → fp8, `matmulfp8` (fp8×fp8→fp16), dequant by `a_scale[m]·w_scale[n]`. fp8-ness is TYPED: the
    //    quantized act (`Df::Fp8` via `synth_df` + the qfp8ch convert's output) and the weight/act matmul
    //    operands (`set_df(Df::Fp8)`) carry SEN143_FP8 residency (½ f16) + drive the `matmulfp8` opFunc —
    //    no name suffix.
    if let [_, _, ws] = ins[..] {
        return Ok(super::ktir_matmul_fp8::matmul_fp8_descriptors(
            &super::ktir_matmul_fp8::Fp8Facts {
                a_tid: a.tid,
                a_name: a.name(),
                w_name,
                ws_name: ws.name(),
                out_tid: out.tid,
            },
            m,
            k,
            n,
            sym_id_base,
            layout,
            quantized,
        )?);
    }
    // Internal consistency guards — a mismatch is a build error, not garbage.
    //
    // ⛔ THE WEIGHT'S VIEW IS `[n, k]`, WHICH IS THE SAME STATEMENT AS main's `[k, n]` REGION. main
    // checked `w.region.rows.len != k` and `w.region.cols.len != n` against the SubtileIR region;
    // `KtirFunc::matmul` viewed the same buffer as its natural `[n, k]`, so the two checks swap sides
    // and nothing else about them changes.
    // ⛔ WHICH OF W'S TWO EXTENTS IS K IS A PROVEN FACT, NOT A CONVENTION. The check above reads the
    // weight as `[n, k]` (transpose-B). A plain-B weight is `[k, n]`, and at `k == n` — granite's
    // `[4096, 4096]` output projection — the extents alone cannot tell the two framings apart, so a
    // hardcoded reading is a silent wrong contraction on one of them. `proof.b` carries the
    // orientation proved from the op's own `indexing_maps` by `whole_function::matmul_b_orientation`.
    let (w_k, w_n, framing) = match proof.b {
        super::whole_function::BOrient::TransposeB => (w.c_len, w.r_len, "[n, k]"),
        super::whole_function::BOrient::PlainB => (w.r_len, w.c_len, "[k, n]"),
    };
    if w_k != k {
        return err(format!(
            "MatmulTile t{}: W's K extent is {w_k} but A's cols (K) are {k} — W is {} x {} read as \
             {framing}",
            out.tid, w.r_len, w.c_len
        ));
    }
    if w_n != n {
        return err(format!(
            "MatmulTile t{}: W's N extent is {w_n} but out cols (N) are {n} — W is {} x {} read as \
             {framing}",
            out.tid, w.r_len, w.c_len
        ));
    }
    if a.r_len != m {
        return err(format!(
            "MatmulTile t{}: A rows {} != out rows (M) {m}",
            out.tid, a.r_len
        ));
    }
    // ── Build-time GUARDS (priority-1 rule: a contract violation is a `cargo
    //    build` error, never on-card garbage). assemble_matmul emits NO
    //    coordinateMasking_, so the stick dims (K on INPUT, N on OUTPUT) MUST be
    //    64-fp16-stick-aligned; an unaligned dim would silently corrupt the tile
    //    layout on-card (guard #3). ──
    if !k.is_multiple_of(FP16_ELEMS_PER_STICK) {
        return err(format!(
            "MatmulTile t{}: K={k} not a multiple of the {FP16_ELEMS_PER_STICK}-fp16 stick \
             (assemble_matmul emits no coordinateMasking_) — emit masking or pad K",
            out.tid
        ));
    }
    // The OUTPUT stick dim N must be a whole 64-fp16 stick on-device. Round the LOGICAL width up to a
    // stick (`n64`); then — if this matmul's MACs cross the util floor AND `n64`'s stick count is
    // prime/awkward (would strand the gemm on <8 cores, e.g. granite lm_head 49216/64=769) — bump the
    // stick count to a multiple of 8 (`bump_sticks_to_splittable`, SHARED with the worker's weight
    // zero-pad so the staged buffer matches the emitted device width). The kernel's extra (n_dev − n)
    // columns are ZERO; the SubtileIR/manifest LOGICAL shape stays `n`; the host reads the leading
    // `vocab = n` (contiguous, m=1). ONLY the on-device layout uses `n_dev` (a whole stick by construction).
    // TYPE-SAFE device width (the padding/alignment invariant): `DeviceWidth::for_matmul` is the
    // SOLE rule for this door, SHARED with the intermediate's reservation in the whole-function
    // walk — the same `m, n, k, windowed` facts, so the buffer the mint reserves and the width the
    // emitter writes cannot disagree (issue 201 item 4: they previously over-reserved 8× on
    // granite tiled_k). `windowed_program` carries whether the padded weight columns would be read
    // from the caller's parameter windows — the one fact the staged-buffer contract of
    // `for_output` does not hold here.
    let n_dev =
        DeviceWidth::for_matmul(m, n, k, matches!(proof.origin, OperandOrigin::Windowed)).get();
    let macs = m as u64 * n_dev as u64 * k as u64;
    // ── GUARD #11 (util floor) computed on the PROVEN partition `CoreSplit::plan` (Kani: disjoint +
    //    covering, #50-free) — the SAME split the emit uses (`matmul_split_map` defers to CoreSplit for
    //    batch=1). A FLOP-heavy matmul left on <8 cores is the 1.45× sengraph regression this emitter
    //    exists to fix; the padding above is what fills them for a prime-stick output. ──
    let sp = CoreSplit::plan(m, n_dev);
    let cores = sp.ncores();
    if macs >= (1 << 20) && cores < UTIL_FLOOR_CORES {
        return err(format!(
            "MatmulTile t{}: {m}×{n_dev}×{k} ({macs} MACs) CoreSplit-divided onto only {cores} \
             core(s) — below the util floor; the OUTPUT stick count is not splittable to ≥8 even \
             after padding.",
            out.tid
        ));
    }
    // ── GUARD (priority-1, from OBSERVED on-card crash 2026-06-25): the per-core STICK extent (N on
    //    OUTPUT/KERNEL) MUST stay a whole 64-fp16 stick — a sub-stick split DtException's the dxp
    //    scheduler (L3DlOpsScheduler.cpp:1040). `CoreSplit` splits `out` by STICK COUNT (whole sticks by
    //    construction) and never splits K, so this holds; kept as a build-time seal. ──
    if !(n_dev / sp.stick_cores).is_multiple_of(FP16_ELEMS_PER_STICK) {
        return err(format!(
            "MatmulTile t{}: CoreSplit stick_cores={} leaves a SUB-STICK per-core N extent \
             (N/core={}) — must be a multiple of {FP16_ELEMS_PER_STICK}.",
            out.tid,
            sp.stick_cores,
            n_dev / sp.stick_cores
        ));
    }
    let op_name = format!("matmul_o{}", out.tid);
    // ⭐⭐⭐⭐⭐ THE WINDOWED EMISSION — every operand's corner and the kernel's physical `in`
    // extent, stated to the one assembler that can carry them.
    //
    // A Triton kernel's K/N-tiled matmul sweeps a WINDOW of each parameter: `[m, k_blk]` of an
    // `[m, k_full]` activation, `[k_blk, n_blk]` of the weight, `[m, n_blk]` of the output — the
    // regions this door is handed carry each load's OWN tile (whole_function::region_for_operand),
    // and this arm turns each corner into the operand's `offset_elems` exactly the way the
    // GQA-group bmm does. MEASURED on the unblocked SwiGLU small twin: without this arm every
    // matmul of every trip read the parameter's FIRST tile, and the card output matched
    // `2·[silu(2·x₀Wg)·(2·x₀Wu)]Wd` to corr 0.999996 — the ×2s being the duplicated no-offset
    // trips summed by the accumulator `arith.addf`s.
    //
    // ⛔ AND THE OFFSETS ALONE ARE NOT THE FIX: the kernel's stick-group stride is derived from the
    // TILE's k (this op's swept `k`), but a windowed weight lives in an allocation whose `in`
    // extent is the TENSOR's own. Reading column-block `k_blk` of a `[k_full, n]` weight with the
    // tile's stride lands every stick group after the first `k_blk`-invented rows away — the same
    // corr 0.983 ceiling the offsets-alone run hit. `w_in_phys` (the tensor's own in-extent) is
    // declared through `TensorArg::device_extent` — the torch-spyre `arg.device_size` law the
    // batched attention already relies on for the same reason.
    //
    // ⛔ THE ACTIVATION AND OUTPUT GROUP STRIDES ARE ALREADY RIGHT: both are RowBlocked sticked on
    // their column axis (a group stride is `rows · stick`), and `m` is not tiled by this door
    // (`base_addressed` above still refuses A's row corner), so the swept `m` IS the tensor's row
    // count and the handle extents the opspec derives are the tensor's own.
    //
    // A FULL-TENSOR program — every region the whole window of its view, corners 0, `k` the
    // tensor's whole extent — emits exactly the offsets-zero, no-device-extent form, which is
    // byte-identical to `assemble_matmul_seeded` (`matmul_opspec_off` at zero offsets IS
    // `matmul_opspec`; `set_device_extent` at `w_in_phys == k` states what the stride derivation
    // already concluded).
    //
    // The corners, per orientation (the kernel's coords are `[in=k, out=n]`, so which of W's two
    // view axes is k_start and which is n_start is decided by the PROVEN `proof.b`, not by
    // convention — at `k == n` the extents cannot tell the two framings apart):
    //   * A (RowBlocked `[rows, cols]`):      `rc_of(a.v_rows, a.v_cols, a.r_start, a.c_start)`
    //   * W (Kernel `[in, out]`):             `rc_of(k_full, n_full, k_start, n_start)`
    //   * O (RowBlocked `[rows, cols]`):      `col_of(m, out_full_cols, out.c_start)`
    // with `out_full_cols = out.v_cols`.
    let a_off = crate::addr::rc_of(a.v_rows, a.v_cols, a.r_start, a.c_start, Df::Fp16);
    let (k_start, n_start, k_full, n_full) = match proof.b {
        super::whole_function::BOrient::TransposeB => {
            // W's view is `[n, k]`: the ROW axis is out (n), the COLUMN axis is in (k).
            (w.c_start, w.r_start, w.v_cols, w.v_rows)
        }
        super::whole_function::BOrient::PlainB => {
            // W's view is `[k, n]`: the ROW axis is in (k), the COLUMN axis is out (n).
            (w.r_start, w.c_start, w.v_rows, w.v_cols)
        }
    };
    // ⛔ THE KERNEL'S OUT CORNER STEPS WHOLE STICKS (`(n/stk)·(k_phys·stk) + k·stk + (n%stk)`), so
    // a sub-stick `n_start` has no address on that law — refused by name rather than emitted as a
    // mis-strided read. (The IN corner steps `k·stk` and needs no alignment.)
    if n_start != 0 && !n_start.is_multiple_of(FP16_ELEMS_PER_STICK) {
        return err(format!(
            "{name}: weight (W) t{} states out-corner {} ({} framing), which is not a whole \
             {FP16_ELEMS_PER_STICK}-column stick — the kernel's stick-group stride steps whole \
             sticks, so a sub-stick out corner has no address. Pad the blocking so every window \
             corner is stick-aligned.",
            w.tid, n_start, framing,
        ));
    }
    let w_off = crate::addr::rc_of(k_full, n_full, k_start, n_start, Df::Fp16);
    let o_off = crate::addr::col_of(m, out.v_cols, out.c_start, Df::Fp16);
    // ⛔⛔⛔ A WHOLE-TENSOR PROGRAM KEEPS THE BASE EMISSION, BYTE-IDENTICAL. Nine fixtures are
    // verified on-card through `assemble_matmul_seeded`, and that call is also the one that
    // applies `out_width_the_weight_holds` — the pad-drop a KTIR producer's UNPADDED caller-owned
    // weight needs (MEASURED as `resolve_seg_base`'s footprint refusal at both SwiGLU blockings).
    // `assemble_matmul_windowed` states offsets and a device extent instead, which is a different
    // (correct) contract for a WINDOWED program and an unneeded risk for a whole-tensor one: a
    // window that is its view entire, corners 0, is exactly the program the base form was built
    // for. Anything else — any corner, any partial window — is the tiled case this arm exists for.
    let whole =
        |r: &Region| r.r_start == 0 && r.c_start == 0 && r.r_len == r.v_rows && r.c_len == r.v_cols;
    let op = if whole(&a) && whole(&w) && whole(&out) {
        assemble_matmul_seeded(
            &op_name,
            m,
            n_dev,
            k,
            1,
            &rb(
                &a.name(),
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(k),
            ),
            &Stk::<KernelTag>::kernel(k as usize, n_dev as usize, &w_name),
            &rb(
                &out.name(),
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(n_dev),
            ),
            sym_id_base,
            layout,
        )
    } else {
        // ⛔ A PARTIAL WINDOW MAY NOT TAKE THE UTIL-FLOOR PAD. `n_dev > n` invents output columns
        // the program never stated; on a whole-tensor program their write lands in the placement's
        // own pad (the footprint guard's business), but on a WINDOWED program it lands in the
        // NEIGHBOUR WINDOW's columns — silent corruption of a trip this program never names. The
        // weight's pad columns are read through the same neighbour. Refused by name; the blocking
        // must keep every window's stick count splittable on its own.
        if n_dev > n {
            // ⭐ BUT A WINDOW THAT SPANS THE OUTPUT'S WHOLE COLUMN WIDTH AT CORNER 0 MAY TAKE THE
            // PAD. The pad columns land in the row-partial window's OWN placement rows — the same
            // contract the base emission's pad has always carried (the footprint guard's business),
            // not a neighbour's. The prefill lm-head tail is exactly this: the output view is
            // `[rung_rows, vocab]` and the matmul's window is `[1, vocab]` (the tail extraction
            // materializes the last row), full-width, corner 0.
            let cols_entire = |r: &Region| r.c_start == 0 && r.c_len == r.v_cols;
            if !cols_entire(&out) {
                return err(format!(
                    "MatmulTile t{}: the output window is partial (n={n} of {}) and the util floor \
                     would pad the device width to {n_dev} — the padded columns would write into the \
                     neighbouring window's output and read its weight columns. Pad the blocking so \
                     each window's stick count is splittable to ≥8 on its own.",
                    out.tid, out.v_cols,
                ));
            }
        }
        assemble_matmul_windowed(
            &op_name,
            MatM::of_token_rows(m),
            MatN::of_out_features(n_dev),
            MatK::of_in_features(k),
            &rb(
                &a.name(),
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(k),
            ),
            a_off,
            &Stk::<KernelTag>::kernel(k as usize, n_dev as usize, &w_name),
            w_off,
            k_full,
            &rb(
                &out.name(),
                RowCount::of_token_rows(m),
                BlockCols::of_feature_cols(n_dev),
            ),
            o_off,
            sym_id_base,
            layout,
        )
    };
    // Default path: one f16 matmul.
    Ok(vec![op])
}

/// A WHOLE-TENSOR 2-D TRANSPOSE — `[mb, out]` → `[out, mb]`, one `interslicetranspose_fp16` on the PT
/// unit, via [`super::assemble_transpose`].
///
/// ⭐ THE ASSEMBLER WAS ALREADY WRITTEN; WHAT WAS MISSING WAS A DOOR. `OpFunc::Transpose` is a real
/// device primitive (`superdsc_opspec.rs`, wire name `interslicetranspose_fp16`) and
/// [`super::assemble_transpose`] has carried its 8×8 output-stick override since the SubtileIR path —
/// with no `Program` variant and no entry point, so it had ZERO callers and a KTIR producer could not
/// ask for a transpose at all. This is that entry point, and it is a per-kind body like every other
/// one here: it reads its two operands out of the program's own regions ([`split_out`]) and calls the
/// assembler. Unlike its neighbours it ports no `lower_*_node` body, because `ibm/main` has none —
/// main's Kᵀ is a whole-page re-transpose inside `lower_attn_node`, never a node of its own. So the
/// descriptor name follows THIS file's own `<func>_o<tid>` convention rather than matching a main
/// spelling; there is no main spelling to match.
///
/// ⭐ WHY A PRODUCER NEEDS IT. A `tt.trans` of a value the program LOADED folds into that load's
/// access-tile order and never becomes an op. A `tt.trans` of a value the program COMPUTED — a RoPE'd
/// K — has no access tile to fold into, so the reorder has to be a descriptor.
///
/// ⛔⛔⛔ AND IT IS THE STICK-ALIGNED CASE ONLY, WHICH IS THE ONE CASE THE HISTORY HERE DECIDES. This
/// file's module header records why the generic op walk was deleted: "a `linalg.transpose` of a
/// per-step `[1, 64]` key became a restickify whose stick extent is 1, which is not a whole multiple of
/// the 64-element SEN169_FP16 stick". That is not a builder limitation to route around — the dxp
/// scheduler refuses a sub-stick tile (`L3DlOpsScheduler:1040`), and the on-card ReStickify that was
/// asked for that shape anyway wrote the SECOND Kᵀ stick wrong, which is what garbled decode past 64
/// tokens (`ir/bridge/tiled_op_sdsc_op/attn.rs`'s "ONE restickify PER SUB-BLOCK" note, and the
/// reverted "LX-tile block" commit behind it).
///
/// So a mis-aligned extent is a LOUD REFUSAL naming the extent, the 64 rule and the alternative the
/// shipped attention actually uses — pad the row count to a stick multiple and loop whole 64×64 tiles
/// — and NOT an attempt at the unaligned shape. The unaligned transpose is a shape law of its own: it
/// needs a PADDED destination (so the trailing partial stick has somewhere legal to land) and a typed
/// tile-extent door to carry the padded-vs-logical extents apart, neither of which exists here. It is
/// out of scope of an entry point, deliberately.
///
/// ⛔ THE REFUSAL IS HERE AND NOT ONLY IN THE ASSEMBLER. [`super::assemble_transpose`] `panic!`s on a
/// bad extent (every `assemble_*` in that file does — they are called from a `#[forward]` expansion,
/// where a panic IS the build error). A producer crossing this door gets an `Error` it can report
/// against its own op instead: the stick law is re-stated below so the refusal can name the padding
/// alternative, and the builder's remaining refusals arrive as `Error`s through
/// [`super::try_assemble_transpose`]. The panicking form is never called from here.
///
/// ✅ THE CORE DIVISION IS NO LONGER A NARROWING. This door used to refuse shapes the stick law
/// accepts, because the builder split `mb` alone via `distribute_cores` — only a row extent that is
/// a multiple of `8 · 32 = 256` gave whole 8-blocks, and the vendor golden's own shape (`mb 384`)
/// was refused. That defect is fixed at the division: [`super::transpose_core_split`] cuts BOTH axes
/// at the block, measured against that golden (it reproduces the fixture's
/// `numWkSlicesPerDim_ {"out": 8, "mb": 4}` verbatim), so every pair of whole 64-stick extents now
/// bakes. The stick law below is therefore the ONLY shape this door refuses, deliberately.
pub fn transpose(
    name: &str,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let (ins, out) = split_out(name, r, layout, 1)?;
    let a = ins[0];
    // The INPUT's view IS the op's `[rows, cols]`: the builder iterates the input shape and the
    // output is its transpose. Read from the input so the two cannot be read from the same place and
    // agree vacuously.
    let (mb, cols) = (a.v_rows, a.v_cols);
    // ⛔⛔⛔ THE STICK LAW, FIRST, so the historically fatal shape can never fall past it into a
    // shape complaint about something else.
    let stk = Fp16::ELEMS_PER_STICK;
    if mb == 0 || cols == 0 || !mb.is_multiple_of(stk) || !cols.is_multiple_of(stk) {
        return err(format!(
            "{name}: transposing t{} `[{mb}, {cols}]` into t{} `[{cols}, {mb}]` — BOTH extents must \
             be whole multiples of the {stk}-element SEN169_FP16 stick and {} is not. This relayout \
             sticks its INPUT on the column extent and its OUTPUT on the row extent — the two axes \
             swap, which is the transposition — so each extent is a stick extent of one side; \
             a sub-stick tile is refused by the dxp scheduler (L3DlOpsScheduler:1040), and the on-card \
             ReStickify that was asked for the per-step `[1, {stk}]` key anyway wrote the SECOND \
             transposed stick wrong and garbled decode past {stk} tokens. Pad the offending extent up \
             to a multiple of {stk} and loop whole {stk}x{stk} tiles, which is what the shipped \
             attention does (it pads its row count and emits one single-stick transpose per \
             sub-block). A padded destination for the unaligned extent is a separate shape law with \
             its own typed tile extents; this entry point does not guess one.",
            a.tid,
            out.tid,
            match (
                mb.is_multiple_of(stk) && mb != 0,
                cols.is_multiple_of(stk) && cols != 0
            ) {
                (false, false) => format!("neither ({mb} rows, {cols} columns)"),
                (false, true) => format!("the row extent {mb}"),
                _ => format!("the column extent {cols}"),
            },
        ));
    }
    // The OUTPUT's own view must BE the transpose. Otherwise the descriptor would write a shape the
    // program never declared — and the two extents would still be stick-aligned, so nothing above
    // catches it.
    if (out.v_rows, out.v_cols) != (cols, mb) {
        return err(format!(
            "{name}: reads t{} `[{mb}, {cols}]` and writes t{} `[{}, {}]` — a transpose's output view \
             is its input's, swapped (`[{cols}, {mb}]`). The descriptor iterates the INPUT shape and \
             carries the reorder on the output's own stick order, so it cannot reshape as well.",
            a.tid, out.tid, out.v_rows, out.v_cols,
        ));
    }
    // WHOLE VIEWS, BOTH SIDES: `assemble_transpose` takes two NAMES and two extents, no corner and no
    // window — the same base-addressing fact [`base_addressed`] states for rows, in the column
    // direction too. A windowed transpose is a different descriptor, so a stated window is refused
    // rather than silently widened to the buffer.
    for (x, role) in [(a, "input"), (out, "output")] {
        if x.c_start != 0 || x.c_len != x.v_cols {
            return err(format!(
                "{name}: the {role} t{} takes columns {}..{} of its `[{}, {}]` view — \
                 the relayout builder names a tensor and its two extents, with no column corner and \
                 no window, so it transposes the WHOLE buffer. Materialize the window first.",
                x.tid,
                x.c_start,
                x.c_start + x.c_len,
                x.v_rows,
                x.v_cols,
            ));
        }
    }
    // The store windows must TILE the output, exactly as they must for every pointwise body: one
    // descriptor spans all of it, so a program whose windows do not cover it would leave rows holding
    // whatever the buffer held. `node_rows` is the shared guard; it answers the output's row extent,
    // which is the input's column extent by the check above.
    if node_rows(name, &out)? != cols {
        return err(format!(
            "{name}: the output's store windows do not cover its `[{cols}, {mb}]` view"
        ));
    }
    // The input's access tiles must likewise cover it — `Region::r_len` is block 0's height alone.
    if a.r_cover != (0, mb) {
        return err(format!(
            "{name}: the program's access tiles over its input t{} cover rows {}..{} of a {mb}-row \
             view — one descriptor transposes all of it, so the windows must tile `0..{mb}`.",
            a.tid, a.r_cover.0, a.r_cover.1,
        ));
    }
    // ⛔⛔⛔ THE RELAYOUT GOES OUT ON THE **RESTICKIFY** DOOR, NOT ON `OpFunc::Transpose`.
    //
    // `interslicetranspose_fp16` is the op named for this job and it CANNOT BE TRANSLATED at the
    // default RCUDD1A arch, at any shape. Its output stick is the 8×8 inter-slice block over
    // (`out`, `mb`) — TWO dims — and `Ddc::transformForInterSliceRestickify` turns a relayout's output
    // stick into a dynamic mask over the innermost loop carrying `outputStickDimOrder[0]`, which for a
    // 2-D tile is the FUSED `loop_dsX_dsY_out_mb`; `SNComputeLowering::constructDynamicMasking` accepts
    // one loop carrying ONE dim (`SNComputeLowering.cpp:74`). Measured through `dxp_standalone`: every
    // shape refuses there, including shapes whose core division splits both axes like the golden
    // `sdsc_interslicetranspose.json` — which is what ruled the core division out as the cause.
    // `OpFunc::Restickify` does the SAME transposition with a one-dim output stick and is accepted.
    // See `super::restickify_transpose_opspec_2d`, which states the addressing argument in full, and
    // `tests/zz_the_transpose_door_carries_a_two_dim_output_stick.rs`, which pins both doors' output sticks.
    //
    // ⛔ THE FALLIBLE FORM, because the builder makes a refusal this body cannot pre-check: the
    // per-core tile must fit LX and this door cannot time-tile, and what the per-core tile IS depends
    // on `distribute_cores`' answer for this shape. Restating that here would mean duplicating the
    // divider; a panicking assembler would abort the build with a bare panic where a producer needs an
    // error against its own op.
    // ⭐ THE HANDLES STATE THE VIEWS THE CHECKS ABOVE ALREADY PROVED: `[mb, cols]` in, `[cols, mb]`
    // out — whole buffers, both sides, by this body's own whole-view law. The mint's new fact is the
    // PLACEMENT half: a view can be whole while the reservation behind its name is smaller, which
    // bakes clean and overruns into the next tensor's bytes.
    super::try_assemble_restickify_transpose_2d(
        &format!("transpose_o{}", out.tid),
        mb,
        cols,
        &rb(
            &a.name(),
            RowCount::of_token_rows(mb),
            BlockCols::of_feature_cols(cols),
        ),
        &rb(
            &out.name(),
            RowCount::of_token_rows(cols),
            BlockCols::of_feature_cols(mb),
        ),
        sym_id_base,
        layout,
    )
    .map(|op| vec![op])
    .map_err(|message| Error {
        message: format!("{name}: {message}"),
    })
}

/// A program's TENSOR inputs (in parameter order) and its output.
///
/// ⛔ THE OUTPUT IS THE ONE A `ktdp.store` WRITES, and the model constants are the parameters at a
/// reserved ScalarMul-scale tid. Neither is a position: parameters are minted in first-use order, so
/// a constant first read after the output's view sits behind it.
pub fn split_out(
    name: &str,
    r: &[Region],
    layout: Option<&BundleLayout>,
    arity: usize,
) -> Result<(Vec<Region>, Region), Error> {
    split_out_excluding(name, r, layout, arity, &[])
}

/// [`split_out`] WITH THE INDEX PARAMETERS LEFT OUT — the door a gathering program needs.
///
/// ⛔ IT IS NOT A LOOSENED ARITY. A gather's index vector is a parameter and a tensor, and the counting
/// here treats every non-output parameter as one of the op's compute inputs — which is exact for every
/// affinely-addressed program and WRONG for a gathering one by precisely one operand. Triton's
/// `embedding_fwd` is the measured case: `desc_ids`, `desc_table`, `desc_o`, one `arith.mulf`, and so
/// "2 tensor input(s) for an op that reads 1" — a true statement about the parameter list and a false
/// one about the op. The index is an ADDRESSING operand (`computeOp_.indirectAccessIndexLabeledDs`,
/// never `inputLabeledDs`), so it is excluded HERE and re-attached by
/// [`OpSpec::attach_indirect_index`](crate::superdsc_opspec::OpSpec::attach_indirect_index) at the
/// position the descriptor needs. The arity a body states is unchanged, which is the point: the count
/// still has to be right.
///
/// `skip_tids` comes from [`gather_of`], i.e. from the program, never from a caller's assumption about
/// which parameter is an index.
///
/// ⛔ EVERY OPERAND THESE BODIES EMIT IS BASE-ADDRESSED, so a stated row corner cannot ride one
/// ([`base_addressed`]'s own argument). The windowed pointwise arm is the one emitter that CAN place a
/// row corner, and it splits through [`split_regions`] directly — this guard is what keeps every other
/// body from silently reading row 0 of a cornered operand.
pub fn split_out_excluding(
    name: &str,
    r: &[Region],
    layout: Option<&BundleLayout>,
    arity: usize,
    skip_tids: &[u32],
) -> Result<(Vec<Region>, Region), Error> {
    let (ins, out) = split_regions(name, r, layout, arity, skip_tids)?;
    base_addressed(name, &out, "output")?;
    for (i, x) in ins.iter().enumerate() {
        base_addressed(name, x, &format!("input {i}"))?;
    }
    Ok((ins, out))
}

/// [`split_out_excluding`]'s SPLIT — the is_out/scale/skip filter and the arity check, with NO
/// addressing guard. The door the windowed pointwise arm needs: its operands carry row corners the
/// per-operand window emission places, so which operands are inputs is a fact about the parameter
/// list, not about how any one body addresses them.
fn split_regions(
    name: &str,
    r: &[Region],
    layout: Option<&BundleLayout>,
    arity: usize,
    skip_tids: &[u32],
) -> Result<(Vec<Region>, Region), Error> {
    let outs: Vec<Region> = r.iter().copied().filter(|x| x.is_out).collect();
    let [out] = outs[..] else {
        return err(format!(
            "{name}: {} parameter(s) written by a `ktdp.store` — a node writes exactly one",
            outs.len()
        ));
    };
    let ins: Vec<Region> = r
        .iter()
        .copied()
        .filter(|x| {
            !x.is_out && scale_idx_of(layout, x.tid).is_none() && !skip_tids.contains(&x.tid)
        })
        .collect();
    if ins.len() != arity {
        return err(format!(
            "{name}: {} tensor input(s) for an op that reads {arity}",
            ins.len()
        ));
    }
    Ok((ins, out))
}

/// The consumer door for `Program::Elementwise` — the op table, and the extent guard that used to be
/// absent. Every operand shape below is a SHAPE THE PRODUCER ACTUALLY BUILDS, cited to it.
#[cfg(test)]
mod elementwise_tests {
    use super::*;

    /// One whole-buffer operand: the view IS the window, and the store windows tile it — the shape
    /// `regions()` reports for a parameter whose `construct_access_tile` takes all of its view.
    fn reg(tid: u32, rows: u32, cols: u32, is_out: bool) -> Region {
        Region {
            tid,
            v_rows: rows,
            v_cols: cols,
            r_start: 0,
            c_start: 0,
            r_len: rows,
            c_len: cols,
            r_cover: (0, rows),
            c_cover: (0, cols),
            is_out,
            is_fp8: false,
            df: Df::Fp16,
        }
    }

    /// A binary node's parameter list in producer order: `[in0, in1, out]`.
    fn node(a: Region, b: Region, o: Region) -> Vec<Region> {
        vec![a, b, o]
    }

    // ── (ii) REFUSAL + A CONTROL THAT STILL LOWERS ────────────────────────────────────────────

    /// ⛔ THE SHAPE THAT MOTIVATED THE GUARD, AND IT IS LIVE. `SubOp::Elementwise(EwKind::BiasAdd)`
    /// lowers to this `add`, and its bias is rank-1 `[D]` (`crates/compiler/macros/src/shape.rs`,
    /// `sig_bias_add`); `BiasAdd` is ABSENT from `lower_region`'s `elementwise` column-tiling list
    /// (`crates/compiler/subtile/src/subtile_ir.rs`), so its operands are taken WHOLE — a
    /// `[1, D]` region against a `[m, D]` output. The emitter gave it the output's dims and said
    /// nothing.
    #[test]
    fn a_row_broadcast_operand_is_refused_by_name() {
        let r = node(
            reg(5, 4, 128, false),
            reg(6, 1, 128, false),
            reg(7, 4, 128, true),
        );
        let mut sym = 0i64;
        // `EmittedOp` is not `Debug`, so `expect_err` is unavailable — bind the refusal directly.
        let Err(e) = elementwise("add_s3", Elementwise::Add, &r, &[], &mut sym, None) else {
            panic!("a [1,128] operand under a [4,128] output cannot be addressed here");
        };
        assert!(e.message.contains("t6"), "names the operand: {}", e.message);
        assert!(
            e.message.contains("`[1, 128]`") && e.message.contains("`[4, 128]`"),
            "states BOTH extents: {}",
            e.message
        );
        assert!(
            e.message.contains("device_dims"),
            "states WHY — one device_dims for every operand: {}",
            e.message
        );
    }

    /// THE CONTROL: identical extents on every operand, and it still lowers to exactly one op. If
    /// this ever fails, the guard has stopped being a guard and become a refusal of the ordinary
    /// case.
    #[test]
    fn matching_extents_still_lower_to_one_op() {
        let r = node(
            reg(5, 4, 128, false),
            reg(6, 4, 128, false),
            reg(7, 4, 128, true),
        );
        let mut sym = 0i64;
        let ops = elementwise("add_s3", Elementwise::Add, &r, &[], &mut sym, None)
            .expect("agreeing extents are the ordinary case");
        assert_eq!(ops.len(), 1, "one pointwise op, not a decomposition");
    }

    /// NEGATIVE CONTROL: a DIFFERENT mismatch — the per-row `[m, 1]` scalar, which is the LayerNorm
    /// mean-centering operand (`EwKind::Sub`'s own doc in `subtile_ir.rs`) — STILL refuses. Admitting
    /// `Sub` to the op table did not admit its broadcast: the reference evaluator genuinely
    /// broadcasts `bc == 1`, and this seeded path still cannot state it.
    #[test]
    fn a_per_row_scalar_operand_is_still_refused() {
        let r = node(
            reg(5, 4, 128, false),
            reg(6, 4, 1, false),
            reg(7, 4, 128, true),
        );
        let mut sym = 0i64;
        let Err(e) = elementwise("sub_s3", Elementwise::Sub, &r, &[], &mut sym, None) else {
            panic!("a [4,1] per-row scalar is a broadcast this path cannot state");
        };
        assert!(e.message.contains("t6"), "names the operand: {}", e.message);
        assert!(
            e.message.contains("In::col"),
            "points at the builder that CAN state it: {}",
            e.message
        );
    }

    /// A UNARY's single input is guarded too — the builder hands it the output's dims just the same.
    #[test]
    fn a_unary_input_of_the_wrong_extent_is_refused() {
        let r = vec![reg(5, 1, 128, false), reg(7, 4, 128, true)];
        let mut sym = 0i64;
        let Err(e) = elementwise("silu_s3", Elementwise::Silu, &r, &[], &mut sym, None) else {
            panic!("a unary's input extent must match its output too");
        };
        assert!(e.message.contains("t5"), "names the operand: {}", e.message);
    }

    /// ⭐ THE UNLOCK: a NON-broadcasting subtract — both operands at the output's extent — now
    /// lowers, where the blanket `Elementwise::Sub` refusal used to reject it. Its descriptor name
    /// is `sub_o*`, so `OpFunc::Subtract` is what the bake sees.
    #[test]
    fn a_non_broadcasting_sub_now_lowers() {
        let r = node(
            reg(5, 4, 128, false),
            reg(6, 4, 128, false),
            reg(7, 4, 128, true),
        );
        let mut sym = 0i64;
        let ops = elementwise("sub_s3", Elementwise::Sub, &r, &[], &mut sym, None)
            .expect("a same-extent subtract is structurally an add");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_name, "sub_o7");
    }

    // ── THE BROADCAST ARM, AND EVERY REFUSAL AROUND IT ────────────────────────────────────────
    //
    // ⭐⭐⭐ THE SOFTMAX IS WHY THIS ARM EXISTS. `decoder_layer_one_flat`'s `Elementwise(Sub)` takes the
    // per-row max as a `[64, 1]` operand and its `Elementwise(RealDiv)` takes the per-row sum the same
    // way; both were REFUSED by `pointwise_extents_agree`, correctly, because the seeded whole-tensor
    // builder addresses every operand at the OUTPUT's extent. `In::col` states the broadcast instead,
    // which is the builder that refusal has always named. MEASURED on card once this landed (with the
    // transpose rerouted and the ScalarMul scales bound): `decoder_layer_one_flat` max|err| 0.0306 =
    // 0.783 % of full scale over a 52-op chain, within_2pct_strict (floor 1e-3) 0.8937;
    // `decoder_two_layers_flat` 0.0343 = 0.800 % FS over 104 ops, 0.8458 strict. The STRICT figure is
    // quoted because it is the one that reproduces: the max errors reproduce exactly, a looser
    // denominator did not.

    /// ⭐ THE PER-ROW SCALAR NOW EMITS — one op, still named `sub_o7`, when the caller STATES the axis.
    /// Its twin above (`a_per_row_scalar_operand_is_still_refused`) is the control: the SAME regions
    /// with NO flag still refuse, so the flag is what admits it and not a loosened extent law.
    #[test]
    fn a_stated_col_broadcast_operand_emits_one_op() {
        let r = node(
            reg(5, 4, 128, false),
            reg(6, 4, 1, false),
            reg(7, 4, 128, true),
        );
        let mut sym = 0i64;
        let ops = elementwise(
            "sub_s3",
            Elementwise::Sub,
            &r,
            &[None, Some(BcastAxis::Col)],
            &mut sym,
            None,
        )
        .expect("a [m, 1] per-row scalar IS expressible once the axis is stated");
        assert_eq!(ops.len(), 1, "an operand mode is not a decomposition");
        assert_eq!(
            ops[0].op_name, "sub_o7",
            "the descriptor NAME must not change with the operand mode — it feeds the bundle \
             fingerprint, and the broadcast arm is the same op",
        );
    }

    /// ⭐ AND THE ROW VECTOR TOO, through the OTHER builder. `Mb` is why [`BcastAxis`] is a variant
    /// rather than a bool: an rmsnorm gain is `[1, n]` and reading it as `[m, 1]` would take one column
    /// and spray it.
    #[test]
    fn a_stated_mb_broadcast_operand_emits_one_op() {
        let r = node(
            reg(5, 4, 128, false),
            reg(6, 1, 128, false),
            reg(7, 4, 128, true),
        );
        let mut sym = 0i64;
        let ops = elementwise(
            "mul_s3",
            Elementwise::Mul,
            &r,
            &[None, Some(BcastAxis::Mb)],
            &mut sym,
            None,
        )
        .expect("a [1, n] row vector is the mb-broadcast operand mode");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_name, "multiply_o7");
    }

    // ── THE Mb SOURCE AT A WHOLE-STICK COLUMN CORNER — the per-block slice law ────────────────
    //
    // ⭐⭐⭐ THE SHAPE IS THE WALK'S OWN FLAT Mb FRAME. A spliced program stages a per-trip row
    // (a causal mask block) as `[PAGES·W4·MQ, 64]`, and the walk presents that parameter to the
    // door as the flat `[1, 256]` Mb frame with the trip's block as the window
    // `[w·64, (w+1)·64)` — byte-faithful, because row `w` of the stick-blocked `[4, 64]`
    // parent IS flat cols `[w·64, (w+1)·64)` (one plane, stride 64). The operand mode that
    // places exactly this shape is `In::mb_at` — the card-proven prefix-mask slice, one
    // stick-wide block at a time — so the Mb arm carries the corner instead of refusing it.

    /// ⭐ THE CORNERED SOURCE LOWERS, and the corner MOVES THE OPERAND'S ADDRESS. The same
    /// buffer, the same window width, the same flags — only `c_start` differs (64 of a
    /// `[1, 256]` view, one whole stick in): the Mb operand's emitted base must move by exactly
    /// the corner (64 fp16 elements = 128 B) while the dense input and the whole-tensor output
    /// stay at their bases, and the op stays ONE descriptor under the SAME name — the corner is
    /// addressing, not a decomposition.
    #[test]
    fn an_mb_source_at_a_whole_stick_column_corner_moves_the_operands_address() {
        let rows = 32u32;
        let layout = BundleLayout::default();
        layout.synth(PlaceId::Act(5), &[rows, 64]);
        layout.synth(PlaceId::Act(6), &[1, 256]);
        layout.synth(PlaceId::Act(7), &[rows, 64]);
        let bases = |c_start: u32| {
            let r = node(
                reg(5, rows, 64, false),
                chunked(6, 1, c_start, 64, 256, false),
                reg(7, rows, 64, true),
            );
            let mut sym = 0i64;
            let ops = elementwise(
                "add_m0",
                Elementwise::Add,
                &r,
                &[None, Some(BcastAxis::Mb)],
                &mut sym,
                Some(&layout),
            )
            .unwrap_or_else(|e| panic!("the cornered Mb source lowers: {}", e.message));
            assert_eq!(ops.len(), 1, "an operand mode is not a decomposition");
            assert_eq!(ops[0].op_name, "add_o7");
            operand_bases(&ops)
        };
        let at_zero = bases(0);
        let at_corner = bases(64);
        let [d0, m0, o0] = at_zero[..] else {
            panic!("[dense, mb, out] allocates, got {at_zero:?}")
        };
        let [d1, m1, o1] = at_corner[..] else {
            panic!("[dense, mb, out] allocates, got {at_corner:?}")
        };
        assert_eq!(d0, d1, "the dense operand is untouched by the corner");
        assert_eq!(o0, o1, "the output stays at the buffer base — whole-tensor");
        assert_eq!(
            m1 - m0,
            64 * 2,
            "the Mb operand moves by the corner: 64 fp16 elements = 128 B (got {m0} → {m1})"
        );
    }

    /// ⛔ A SUB-STICK CORNER IS REFUSED BY NAME — the one constructor's own law (#622): a
    /// mid-plane window is not a stick-major tensor of its own, so no descriptor reaches its
    /// bytes. The refusal is `col_window_of`'s, carried with the operand's name.
    #[test]
    fn a_sub_stick_mb_corner_is_refused_by_name() {
        let r = node(
            reg(5, 32, 64, false),
            chunked(6, 1, 32, 64, 256, false),
            reg(7, 32, 64, true),
        );
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "add_m1",
            Elementwise::Add,
            &r,
            &[None, Some(BcastAxis::Mb)],
            &mut sym,
            None,
        ) else {
            panic!("a half-stick Mb corner must not emit — no stick-major presentation reaches it")
        };
        assert!(
            e.message.contains("t6") && e.message.contains("not a whole 64-lane stick"),
            "names the operand and the constructor's own law: {}",
            e.message
        );
    }

    /// ⛔ THE ONE LEGAL CORNER IS STILL COLUMN-ONLY — a ROW corner on the Mb source is the
    /// banded-window shape, and `In::mb_at` states a column window, not a row one. Refused by
    /// name, same family as every other corner these arms cannot place.
    #[test]
    fn a_row_corner_on_an_mb_source_is_still_refused() {
        let r = vec![
            reg(5, 32, 64, false),
            Region {
                tid: 6,
                v_rows: 2,
                v_cols: 256,
                r_start: 1,
                c_start: 0,
                r_len: 1,
                c_len: 64,
                r_cover: (1, 2),
                c_cover: (0, 64),
                is_out: false,
                is_fp8: false,
                df: Df::Fp16,
            },
            reg(7, 32, 64, true),
        ];
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "add_m2",
            Elementwise::Add,
            &r,
            &[None, Some(BcastAxis::Mb)],
            &mut sym,
            None,
        ) else {
            panic!("a row corner is the banded-window shape — no broadcast mode places it")
        };
        assert!(
            e.message.contains("t6") && e.message.contains("row corner"),
            "names the operand and the corner KIND: {}",
            e.message
        );
    }

    /// ⛔ A COLUMN CORNER ON A DENSE OPERAND IS STILL REFUSED — the corner the Mb arm carries is
    /// a fact of the Mb OPERAND MODE (one row of values at a corner); a dense operand at a
    /// column corner is the column-blocked chunk, and that is DENSE elementwise — the dense
    /// arm's own `ColWindow` presentation, not a broadcast mode. Refused by name.
    #[test]
    fn a_column_corner_on_a_dense_operand_is_still_refused() {
        let r = node(
            chunked(5, 32, 64, 64, 256, false),
            reg(6, 1, 64, false),
            reg(7, 32, 64, true),
        );
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "add_m3",
            Elementwise::Add,
            &r,
            &[None, Some(BcastAxis::Mb)],
            &mut sym,
            None,
        ) else {
            panic!("a cornered DENSE operand is the column-blocked chunk — the dense arm's shape")
        };
        assert!(
            e.message.contains("t5") && e.message.contains("column corner"),
            "names the operand and the corner KIND: {}",
            e.message
        );
    }

    /// ⛔ AND THE OUTPUT'S CORNER IS STILL REFUSED — the corner rides the OPERAND; a windowed
    /// output under a broadcast flag is a producer shape no arm here places (the dense arm
    /// carries it, this one does not). Refused by name.
    #[test]
    fn an_output_column_corner_is_still_refused() {
        let r = node(
            reg(5, 32, 64, false),
            reg(6, 1, 64, false),
            chunked(7, 32, 64, 64, 256, true),
        );
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "add_m4",
            Elementwise::Add,
            &r,
            &[None, Some(BcastAxis::Mb)],
            &mut sym,
            None,
        ) else {
            panic!("a windowed output under a broadcast flag is not a shape this arm places")
        };
        assert!(
            e.message.contains("the output t7") && e.message.contains("column corner"),
            "names the output and the corner KIND: {}",
            e.message
        );
    }

    /// ⭐⭐⭐ THE ROW-SPLIT (Y) MODE NOW EMITS — the J-slot merge's own op at its production
    /// geometry: the subtract whose operands are the `[256, 4]` score tile (dense) and the
    /// `[64, 4]` merged max (the J-reduce's output, one spray's worth of rows), under a
    /// `[256, 4]` output — the exact regions the batched paged-decode kernel's broadcast chain
    /// resolves to. One op, still `sub_o*`: the operand mode is addressing, not a
    /// decomposition, and the frame law (`RegionFrame::split_rows`) has already re-proven
    /// `rows == y*mb` and the single-stick source before any operand was built.
    #[test]
    fn a_stated_y_broadcast_operand_emits_one_op() {
        let r = node(
            reg(5, 256, 4, false),
            reg(6, 64, 4, false),
            reg(7, 256, 4, true),
        );
        let mut sym = 0i64;
        let ops = elementwise(
            "sub_s3",
            Elementwise::Sub,
            &r,
            &[None, Some(BcastAxis::Y(4))],
            &mut sym,
            None,
        )
        .expect("the J-slot merge's subtract is the row-split operand mode");
        assert_eq!(ops.len(), 1, "an operand mode is not a decomposition");
        assert_eq!(
            ops[0].op_name, "sub_o7",
            "the descriptor NAME must not change with the operand mode — it feeds the bundle \
             fingerprint, and the row-split arm is the same op",
        );
    }

    /// ⭐ AND THE DESCRIPTOR STATES THE Y-MAJOR FRAME — the same op built directly at the
    /// `OpSpec` door, pinning what the bake actually receives: the plan walks `mb=64` (one
    /// spray's worth of rows), `out=64` (the ONE stick the frame law allows), `y=4` (the spray
    /// count); every tensor's layout is `["y", "mb", "out"]`; and the merged source's scale is
    /// `RedNonStick` on the `y` slot — the vendor's reduced-dim law, the stride-0 read that
    /// addresses `source[b, c]` at `result[j, b, c]` byte for byte.
    #[test]
    fn the_row_split_op_states_the_y_major_frame() {
        use crate::superdsc_opspec::Scale;
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(256),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let merged_h = rb(
            "t6",
            RowCount::of_token_rows(64),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let dense = In::full(&dense_h).ew();
        let merged = In::y_broadcast(&merged_h, 4).ew();
        let op = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &pointwise_tile_op(64, 64, 4, 3, Df::Fp16),
            256,
            64,
            op_func_from_str("sub"),
            &[dense, merged],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::Law),
        )
        .expect("the row-split op emits");
        assert_eq!(op.iter.extent("mb"), 64, "mb is one spray's worth of rows");
        assert_eq!(
            op.iter.extent("out"),
            64,
            "out is the ONE stick the frame law allows"
        );
        assert_eq!(op.iter.extent("y"), 4, "y is the spray count");
        for a in &op.args {
            assert_eq!(
                a.view().layout,
                &["y", "mb", "out"] as &[&'static str],
                "every tensor presents y-major in the row-split mode: {}",
                a.view().name
            );
        }
        let merged_view = op
            .args
            .iter()
            .find(|a| a.view().name == "t6")
            .expect("the merged source's arg")
            .view();
        assert_eq!(
            merged_view.scale,
            &[Scale::RedNonStick, Scale::Active, Scale::Active] as &[Scale],
            "the spray dim is the stride-0 read — the vendor's reduced-dim law"
        );
        // ⭐⭐⭐ AND THE DIVISION IS THE SPRAY SPLIT — the law this op's own layout demands: the
        // cores go to the leading `y` first (`y:4`), and the cores that leaves to `mb` (`mb:8`) —
        // an `mb` split INSIDE one spray is contiguous, so a short spray count strands no cores.
        assert_eq!(
            op.iter.split_of("y"),
            4,
            "the spray axis takes its cores first"
        );
        assert_eq!(
            op.iter.split_of("mb"),
            8,
            "the remainder goes to the slab's own outer dim"
        );
        assert_eq!(
            op.iter.split_of("out"),
            1,
            "the one-stick `out` never splits"
        );
    }

    /// ⭐⭐⭐⭐⭐ THE PREFILL MASK ADD'S OWN GEOMETRY, PINNED — `[y=32, mb=21, out=64]` (32 query
    /// heads, 21 tokens, one stick of slots): the spray split takes all 32 cores, ONE SPRAY PER
    /// CORE, and `mb` stays whole. This is the exact op whose rows-first emission measured 620 of
    /// 672 sticks never written (every core's base advanced ONE STICK while its slab was 32
    /// sticks — see [`crate::work::distribute_cores_sprays_first`]'s doc), and the per-core slab
    /// here is the producer's own: the `[672, 64]` score tile's 32-core row partition is exactly
    /// 21 sticks per core, the same bytes at the same bases.
    #[test]
    fn the_row_split_op_at_the_prefill_mask_geometry_splits_one_spray_per_core() {
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(672),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let mask_h = rb(
            "t6",
            RowCount::of_token_rows(21),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let dense = In::full(&dense_h).ew();
        let mask = In::y_broadcast(&mask_h, 32).ew();
        let op = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &pointwise_tile_op(21, 64, 32, 3, Df::Fp16),
            672,
            64,
            op_func_from_str("add"),
            &[dense, mask],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::Law),
        )
        .expect("the mask add emits under the spray split");
        assert_eq!(op.iter.extent("y"), 32, "y is the 32 query heads");
        assert_eq!(op.iter.extent("mb"), 21, "mb is the 21 tokens");
        assert_eq!(op.iter.split_of("y"), 32, "ONE SPRAY PER CORE");
        assert_eq!(
            op.iter.split_of("mb"),
            1,
            "the whole token axis stays in one slab"
        );
    }

    /// ⭐⭐⭐⭐⭐ THE ROW-SPLIT OP SERIALIZES ITS WIRE ORDER THE WAY DXP READS IT — STICK
    /// INNERMOST, RECONSTRUCT WALK. dxp reads `layoutDimOrder_` idx 0 as INNERMOST
    /// (`perfdsc/perfLdsInfo.h:29`, `dsc/dataOpDsc.h:347`) while this crate's address model is
    /// OUTERMOST-first, so the y-major view's `["y","mb","out"]` is serialized REVERSED:
    /// `["out","mb","y"]`, the stick INNERMOST — the score matmul's own output spelling — with
    /// the RECONSTRUCT walk (`maxDimSizes_ [-1,-1,-1]`), because a rank-3 stick-on-last view
    /// classifies `Flat` and would otherwise PIN its actual extents `[32,21,64]`, which dxp
    /// reads at the opposite end first. MEASURED on the prefill mask add (`11_add_o1417` at the
    /// spray split): the verbatim spelling tore the walk — 310 of 672 output sticks never
    /// written, the mask operand read at a TWO-STICK row stride (token row `2t mod 21`
    /// applied) — while the same bytes' producer and every `mb`-split sibling (stick innermost
    /// under `-1`) walked correctly. The address model reads `layout` unchanged, so the
    /// per-core starts (one 21-stick spray per core) are untouched.
    #[test]
    fn the_row_split_op_serializes_stick_innermost_and_reconstructs() {
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(672),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let mask_h = rb(
            "t6",
            RowCount::of_token_rows(21),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let out_h = rb(
            "t7",
            RowCount::of_token_rows(672),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let dense = In::full(&dense_h).ew();
        let mask = In::y_broadcast(&mask_h, 32).ew();
        let mut sid = 0i64;
        let emitted =
            crate::ir::bridge::tiled_op_sdsc_op::assemble_pointwise_broadcast_off_from_tile(
                "add_o1417",
                &pointwise_tile_op(21, 64, 32, 3, Df::Fp16),
                "add",
                672,
                64,
                &[dense, mask],
                &out_h,
                crate::addr::OutCorner::ZERO,
                &mut sid,
                None,
            );
        let v = serde_json::to_value(emitted.op.as_ref().expect("a descriptor")).unwrap();
        // The spray split rides along untouched: one spray per core.
        assert_eq!(
            v["numWkSlicesPerDim_"],
            serde_json::json!({"mb": 1, "out": 1, "y": 32})
        );
        let key = v["dscs_"][0]
            .as_object()
            .and_then(|m| m.keys().next().cloned())
            .expect("a dsc");
        let tree = v["dscs_"][0][&key]["scheduleTree_"]
            .as_array()
            .expect("the tree");
        let allocs: Vec<_> = tree
            .iter()
            .filter(|n| n.get("nodeType_").and_then(|t| t.as_str()) == Some("allocate"))
            .collect();
        assert_eq!(allocs.len(), 3, "the three operands' allocate nodes");
        for n in &allocs {
            let name = n["name_"].as_str().unwrap_or("?");
            assert_eq!(
                n["layoutDimOrder_"],
                serde_json::json!(["out", "mb", "y"]),
                "{name}: the stick is INNERMOST on the wire, the matmul output's own spelling"
            );
            assert_eq!(
                n["maxDimSizes_"],
                serde_json::json!([-1, -1, -1]),
                "{name}: the reconstruct walk — a pinned array is read at the opposite end first"
            );
        }
    }

    /// ⛔ A STATED NON-SPRAY DIVISION ON A Y-MAJOR OP IS REFUSED — the row-split mode's tensors
    /// walk `["y", "mb", "out"]`, so a split over any inner dim hands every core a strided stick
    /// set the per-core addressing walks at the wrong stride. The refusal names the law and the
    /// measurement; the caller states [`EwDivision::SpraysFirst`] or the [`EwDivision::Law`].
    #[test]
    fn a_row_split_op_with_a_stated_inner_dim_split_is_refused() {
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(672),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let mask_h = rb(
            "t6",
            RowCount::of_token_rows(21),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let dense = In::full(&dense_h).ew();
        let mask = In::y_broadcast(&mask_h, 32).ew();
        let Err(m) = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &pointwise_tile_op(21, 64, 32, 3, Df::Fp16),
            672,
            64,
            op_func_from_str("add"),
            &[dense, mask],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::RowsFirst),
        ) else {
            panic!("a y-major op stated rows-first must not emit — that is the mask add's defect");
        };
        assert!(
            m.contains("row-split (y-major) op whose stated division is not the spray split"),
            "names the law: {m}"
        );
    }

    /// ⛔ THE TILE AND THE OPERANDS MUST AGREE ON THE SPRAY COUNT — a `TileOp` whose `y` dim
    /// states a different count than the y-broadcast operands spray is a caller that built its
    /// tile from a different shape than its operands, and the tiler's iteration space and the
    /// operands' addressing would read different bytes. Refused by name.
    #[test]
    fn a_tile_that_disagrees_with_the_spray_count_is_refused() {
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(256),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let merged_h = rb(
            "t6",
            RowCount::of_token_rows(64),
            BlockCols::of_one_stick(Lanes::FP16),
        );
        let dense = In::full(&dense_h).ew();
        let merged = In::y_broadcast(&merged_h, 4).ew();
        let Err(m) = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &pointwise_tile_op(64, 64, 2, 3, Df::Fp16),
            256,
            64,
            op_func_from_str("sub"),
            &[dense, merged],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::Law),
        ) else {
            panic!("a tile stating y=2 over operands that spray 4 must not emit");
        };
        assert!(
            m.contains("the TileOp's `y` dim states") && m.contains("spray 4"),
            "states both counts: {m}"
        );
    }

    /// ⛔ THE SINGLE-STICK LAW AT THE OPSPEC DOOR — a row-split op wider than one stick did not
    /// come through the Y admission (the spray's source is single-stick by the frame law), and
    /// emitting it would address a wide op in a frame built for one stick. Refused by name.
    #[test]
    fn a_row_split_op_wider_than_one_stick_is_refused() {
        let dense_h = rb(
            "t5",
            RowCount::of_token_rows(256),
            BlockCols::of_feature_cols(128),
        );
        let merged_h = rb(
            "t6",
            RowCount::of_token_rows(64),
            BlockCols::of_feature_cols(128),
        );
        let dense = In::full(&dense_h).ew();
        let merged = In::y_broadcast(&merged_h, 4).ew();
        let Err(m) = crate::ir::bridge::tiled_op_sdsc_op::pointwise_broadcast_opspec_from_tile(
            &pointwise_tile_op(64, 128, 4, 3, Df::Fp16),
            256,
            128,
            op_func_from_str("sub"),
            &[dense, merged],
            "t7",
            crate::addr::OutCorner::ZERO,
            false,
            None,
            crate::work::CoreDivision::Ew(EwDivision::Law),
        ) else {
            panic!("a row-split op at cols=128 must not emit");
        };
        assert!(
            m.contains("a row-split (y-broadcast) op at cols=128"),
            "names the width and the law: {m}"
        );
    }

    /// ⛔ THE FRAME LAW AT THE CONSUMER DOOR — an output whose rows do not split into exactly
    /// `y` sprays of the source's rows (`250` into 4 sprays of 62 — 248, two short) is refused
    /// by the frame law BEFORE any want check or operand is built: the split the bytes do not
    /// support is a build failure, not a wrong-byte bake.
    #[test]
    fn an_output_whose_rows_do_not_split_is_refused_by_the_frame_law() {
        let r = node(
            reg(5, 250, 4, false),
            reg(6, 62, 4, false),
            reg(7, 250, 4, true),
        );
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "sub_s3",
            Elementwise::Sub,
            &r,
            &[None, Some(BcastAxis::Y(4))],
            &mut sym,
            None,
        ) else {
            panic!("250 rows do not split into 4 sprays of 62 — the frame law must refuse");
        };
        assert!(
            e.message.contains("rows == y*mb"),
            "names the frame law's own arithmetic: {}",
            e.message
        );
    }

    /// ⛔ AND THE MULTI-STICK OUTPUT — a `[256, 128]` output under a row-split flag: the
    /// source would have to be multi-stick, which the split law refuses (the split's `out` dim
    /// IS the one stick). Refused by name at the same door.
    #[test]
    fn a_multi_stick_output_under_a_row_split_flag_is_refused() {
        let r = node(
            reg(5, 256, 128, false),
            reg(6, 64, 128, false),
            reg(7, 256, 128, true),
        );
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "sub_s3",
            Elementwise::Sub,
            &r,
            &[None, Some(BcastAxis::Y(4))],
            &mut sym,
            None,
        ) else {
            panic!("a multi-stick row-split source must not emit");
        };
        assert!(
            e.message.contains("SINGLE stick column"),
            "names the single-stick law: {}",
            e.message
        );
    }

    /// ⛔⛔⛔ THE AXIS AND THE EXTENT MUST AGREE, and this is the refusal that makes the flag a
    /// STATEMENT rather than a licence. A `Col` flag over an operand that is really the whole tile
    /// would address one lane per row and spray it over 128 columns — a silently wrong answer with no
    /// shape error anywhere, which is exactly what the seeded path's own refusal exists to prevent.
    #[test]
    fn a_col_flag_over_a_full_tile_is_refused_by_name() {
        let r = node(
            reg(5, 4, 128, false),
            reg(6, 4, 128, false),
            reg(7, 4, 128, true),
        );
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "sub_s3",
            Elementwise::Sub,
            &r,
            &[None, Some(BcastAxis::Col)],
            &mut sym,
            None,
        ) else {
            panic!(
                "a Col flag over a [4, 128] operand was ACCEPTED — the descriptor now reads one lane \
                 per row and sprays it, and nothing downstream can tell"
            );
        };
        assert!(e.message.contains("t6"), "names the operand: {}", e.message);
        assert!(
            e.message.contains("Col") && e.message.contains("[4, 1]"),
            "states the mode AND the extent it requires: {}",
            e.message
        );
    }

    /// ⛔ AND THE FLAGS ARE POSITIONAL, so a length that does not match the inputs is refused rather
    /// than zip-truncated onto some other operand.
    #[test]
    fn a_flag_slice_of_the_wrong_length_is_refused() {
        let r = node(
            reg(5, 4, 128, false),
            reg(6, 4, 1, false),
            reg(7, 4, 128, true),
        );
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "sub_s3",
            Elementwise::Sub,
            &r,
            &[Some(BcastAxis::Col)],
            &mut sym,
            None,
        ) else {
            panic!("one flag for two inputs must refuse — zip would silently flag input 0 instead")
        };
        assert!(
            e.message.contains("POSITIONAL"),
            "says why the length matters: {}",
            e.message
        );
    }

    /// ⛔ A UNARY WHOSE ONLY INPUT IS THE BROADCAST is a materialization, not a pointwise op.
    #[test]
    fn a_unary_broadcast_input_is_refused_by_name() {
        let r = vec![reg(5, 4, 1, false), reg(7, 4, 128, true)];
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "silu_s3",
            Elementwise::Silu,
            &r,
            &[Some(BcastAxis::Col)],
            &mut sym,
            None,
        ) else {
            panic!("f(vector) -> tile is a spray, not a pointwise op")
        };
        assert!(
            e.message.contains("SPRAY") || e.message.contains("spray"),
            "names what it would be: {}",
            e.message
        );
    }

    // ── (iii) EXHAUSTIVE DECLARED-SET ASSERTIONS ──────────────────────────────────────────────

    /// ⛔ NO `_` ARM: a new [`Elementwise`] variant is an E0004 HERE, which is what makes
    /// [`Elementwise::ALL`] provably complete rather than a list someone forgot to extend — the
    /// sweep below then fails unless the new variant is in `ALL` too.
    fn declared_index(k: Elementwise) -> usize {
        match k {
            Elementwise::Silu => 0,
            Elementwise::Gelu => 1,
            Elementwise::Mul => 2,
            Elementwise::Add => 3,
            Elementwise::QuickGelu => 4,
            Elementwise::GeluErf => 5,
            Elementwise::Sub => 6,
            Elementwise::Exp => 7,
            Elementwise::Rsqrt => 8,
            Elementwise::Sqrt => 9,
            Elementwise::Abs => 10,
            Elementwise::Reciprocal => 11,
            Elementwise::Sigmoid => 12,
            Elementwise::Tanh => 13,
            Elementwise::Mish => 14,
            Elementwise::RealDiv => 15,
            Elementwise::Maximum => 16,
            Elementwise::Minimum => 17,
        }
    }

    const DECLARED: usize = 18;

    #[test]
    fn all_declares_every_variant_exactly_once() {
        assert_eq!(
            Elementwise::ALL.len(),
            DECLARED,
            "a variant is missing from ALL"
        );
        let mut seen = [false; DECLARED];
        for k in Elementwise::ALL {
            let i = declared_index(*k);
            assert!(!seen[i], "{k:?} is declared twice in ALL");
            seen[i] = true;
        }
        assert!(
            seen.iter().all(|s| *s),
            "a variant `declared_index` knows is absent from ALL"
        );
    }

    /// EVERY variant either maps to a key + arity or refuses BY NAME. No variant may fall through
    /// silently, and a refusal that does not name the kind is not a refusal anyone can act on.
    #[test]
    fn every_variant_maps_to_a_key_and_arity_or_refuses_by_name() {
        for k in Elementwise::ALL {
            match elementwise_op_func("probe", *k) {
                Ok((key, arity)) => {
                    assert!(!key.is_empty(), "{k:?} mapped to an empty op-func key");
                    assert!(
                        arity == 1 || arity == 2,
                        "{k:?} has arity {arity}; this emitter builds unary and binary pointwise \
                         ops only"
                    );
                }
                Err(e) => {
                    assert!(
                        e.message.contains(&format!("{k:?}")),
                        "the refusal for {k:?} does not name it: {}",
                        e.message
                    );
                }
            }
        }
    }

    /// Exactly the two kinds with NO `OpFunc` are refused — quick-gelu and exact-erf gelu. Any other
    /// refusal is a primitive we have and declined, which is what this pass removed.
    #[test]
    fn exactly_quickgelu_and_geluerf_are_refused() {
        let refused: Vec<Elementwise> = Elementwise::ALL
            .iter()
            .copied()
            .filter(|k| elementwise_op_func("probe", *k).is_err())
            .collect();
        assert_eq!(refused, vec![Elementwise::QuickGelu, Elementwise::GeluErf]);
    }

    // ── THE BYTE-IDENTITY GUARD ───────────────────────────────────────────────────────────────

    /// ⛔⛔⛔ THESE FOUR STRINGS ARE DESCRIPTOR NAMES AND THEY FEED THE BUNDLE FINGERPRINT.
    /// `elementwise` renders `format!("{op_func}_o{tid}")`, so changing `multiply` to `mul` or `gelu`
    /// to `gelufwd` — both of which are the `OpFunc::name()` dxp spellings, and therefore the
    /// plausible "fix" — renames every descriptor these four ever emitted. Nothing would fail to
    /// compile and the only visible trace would be the fingerprint.
    #[test]
    fn the_four_original_kinds_keep_their_op_func_keys() {
        assert_eq!(
            elementwise_op_func("p", Elementwise::Silu).unwrap(),
            ("silu", 1)
        );
        assert_eq!(
            elementwise_op_func("p", Elementwise::Gelu).unwrap(),
            ("gelu", 1)
        );
        assert_eq!(
            elementwise_op_func("p", Elementwise::Mul).unwrap(),
            ("multiply", 2)
        );
        assert_eq!(
            elementwise_op_func("p", Elementwise::Add).unwrap(),
            ("add", 2)
        );
    }

    /// Every key in the table is a key `op_func_from_str` ACCEPTS — it panics on an unknown name, so
    /// a typo here would be a panic at the first emission rather than a compile error. This walks the
    /// whole set so the panic is impossible.
    #[test]
    fn every_key_resolves_through_op_func_from_str() {
        for k in Elementwise::ALL {
            if let Ok((key, _)) = elementwise_op_func("probe", *k) {
                // Panics on an unrecognized key; reaching the assert means the key is in the table.
                let f = crate::emit::op_func_from_str(key);
                assert!(
                    !f.name().is_empty(),
                    "{k:?} → {key:?} has no dxp opFuncName"
                );
            }
        }
    }

    /// ⭐ THE KEY IS NOT THE dxp SPELLING, and the two that differ are pinned here so the distinction
    /// cannot be "tidied away". `multiply` → `OpFunc::Multiply` → dxp `"mul"`; `gelu` →
    /// `OpFunc::Gelu` → dxp `"gelufwd"`. The descriptor's `opFuncName` comes from `OpFunc::name()`;
    /// the key above is the INPUT to `op_func_from_str` and names the descriptor.
    #[test]
    fn the_two_keys_that_differ_from_their_dxp_spelling_still_differ() {
        use crate::emit::op_func_from_str;
        assert_eq!(op_func_from_str("multiply").name(), "mul");
        assert_eq!(op_func_from_str("gelu").name(), "gelufwd");
        // And the ones that agree, agree — including `sub`, the key this pass admitted.
        assert_eq!(op_func_from_str("sub").name(), "sub");
        assert_eq!(op_func_from_str("add").name(), "add");
        assert_eq!(op_func_from_str("silu").name(), "silu");
    }

    /// ⛔ THE GUARD AS A PROPERTY, SWEPT — because the Kani proofs below CANNOT RUN TODAY. Kani
    /// 0.67.0 ships rustc 1.93.0-nightly and `ktir-core` declares `rust-version = 1.94`, so
    /// `cargo kani` fails before it compiles anything ("Found 0 compilation errors") — which is true
    /// of this crate's PRE-EXISTING proofs too, not just these. So the property gets a concrete grid
    /// as well: over every `(rows, cols)` pair for the output and each input, the guard admits the
    /// set IFF every input matches the output exactly. 81 binary cases and 9 unary ones.
    #[test]
    fn the_extent_guard_admits_exactly_the_agreeing_operand_sets_over_a_grid() {
        const EXTENTS: [(u32, u32); 9] = [
            (1, 1),
            (1, 64),
            (1, 128),
            (2, 1),
            (2, 64),
            (2, 128),
            (4, 1),
            (4, 64),
            (4, 128),
        ];
        for o in EXTENTS {
            let out = reg(7, o.0, o.1, true);
            // UNARY: one input, so the quantifier is exercised at arity 1. The basis is the
            // output's own rows — `reg` mints a whole-view region, the shape `pointwise_rows`
            // reads `v_rows` off.
            for a in EXTENTS {
                let ins = [reg(5, a.0, a.1, false)];
                assert_eq!(
                    pointwise_extents_agree("p", Elementwise::Silu, &ins, &out, o.0).is_ok(),
                    a == o,
                    "unary out={o:?} in={a:?}"
                );
            }
            // BINARY: admitted only when BOTH inputs match — neither operand may be the broadcast.
            for a in EXTENTS {
                for b in EXTENTS {
                    let ins = [reg(5, a.0, a.1, false), reg(6, b.0, b.1, false)];
                    assert_eq!(
                        pointwise_extents_agree("p", Elementwise::Add, &ins, &out, o.0).is_ok(),
                        a == o && b == o,
                        "binary out={o:?} in0={a:?} in1={b:?}"
                    );
                }
            }
        }
    }

    /// ⭐ WALL 10's OWN SHAPE, as a unit pin: the FLUSH-COVER WINDOWED operand. The spliced
    /// batched paged-decode attention's runtime length mask states four chunk trips over one
    /// `[4096, 64]` view, tiling `[0, 4096)` flush — so the parameter walk's cover is the whole
    /// view — while each trip's add consumes its OWN `[1024, 64]` window (`r_len == rows ==
    /// 1024`, at row `1024·c`). The guard's basis is the operand's own form, never the cover,
    /// so every trip is admitted on its window; a window that does not hold the sweep, or one
    /// that runs past its view's end, is refused exactly as before.
    #[test]
    fn a_flush_cover_windowed_operand_is_admitted_on_its_own_window() {
        let out = reg(17, 1024, 64, true);
        let qk = reg(16, 1024, 64, false);
        let mask = |trip: u32| {
            let mut m = reg(9, 4096, 64, false);
            m.r_start = 1024 * trip;
            m.r_len = 1024;
            // The flush union of all four trips — the fact that used to be mistaken for "the
            // operand is addressed whole".
            m.r_cover = (0, 4096);
            m
        };
        for trip in 0..4u32 {
            assert!(
                pointwise_extents_agree("p", Elementwise::Add, &[qk, mask(trip)], &out, 1024)
                    .is_ok(),
                "trip {trip}'s [1024, 64] window at row {} holds the sweep at its own corner",
                1024 * trip
            );
        }
        // A window that does NOT hold the sweep is the broadcast shape — refused by both bases.
        let mut short = mask(0);
        short.r_len = 512;
        assert!(pointwise_extents_agree("p", Elementwise::Add, &[qk, short], &out, 1024).is_err());
        // A window past its view's end is an out-of-bounds read — refused.
        let mut over = mask(0);
        over.r_start = 4096 - 512;
        assert!(pointwise_extents_agree("p", Elementwise::Add, &[qk, over], &out, 1024).is_err());
    }

    /// The arity of every key, against the arithmetic the op IS. Unary functions take one operand and
    /// binary ones take two; getting this wrong makes `split_out` accept the wrong parameter count
    /// and `pointwise_tile_op` declare the wrong operand count to the LX budget.
    #[test]
    fn the_declared_arities_are_the_arithmetic() {
        for (k, want) in [
            (Elementwise::Exp, 1usize),
            (Elementwise::Rsqrt, 1),
            (Elementwise::Sqrt, 1),
            (Elementwise::Abs, 1),
            (Elementwise::Reciprocal, 1),
            (Elementwise::Sigmoid, 1),
            (Elementwise::Tanh, 1),
            (Elementwise::Mish, 1),
            (Elementwise::RealDiv, 2),
            (Elementwise::Maximum, 2),
            (Elementwise::Minimum, 2),
            (Elementwise::Sub, 2),
        ] {
            assert_eq!(elementwise_op_func("p", k).unwrap().1, want, "{k:?} arity");
        }
    }

    // ── THE COLUMN-BLOCKED CHUNK — the 12b gemma-4 garbage class ─────────────────────────────
    //
    // ⛔⛔⛔ THE MEASURED DEFECT. `subtile_ir.rs` column-tiles a wide `EwKind::Mul` (and `Add`/
    // `Silu`/`ScalarMul`) into nb-blocks sharing ONE output tensor; this door emitted every
    // chunk at offset 0, so chunk 1 re-computed chunk 0's columns at base 0 and the columns past
    // the first block were NEVER WRITTEN. On the 12b card run the gated-act product
    // `t1121[8192..15360]` was ALL ZERO at every one of 48 layers and the down-GEMM contracted a
    // half-zero activation into garbage. The tiny fixture never split (one block), which is why
    // the class was invisible at small geometry. `silumul` already carried its
    // corners; these tests pin the DENSE door doing the same.

    /// A region whose window is the column block `[c_start, c_start+c_len)` of a wider
    /// `[rows, v_cols]` view — the shape `regions()` reports for a column-chunked node's
    /// parameters (`load_region`/`store_region` put the corner on the access tile).
    fn chunked(tid: u32, rows: u32, c_start: u32, c_len: u32, v_cols: u32, is_out: bool) -> Region {
        Region {
            tid,
            v_rows: rows,
            v_cols,
            r_start: 0,
            c_start,
            r_len: rows,
            c_len,
            r_cover: (0, rows),
            c_cover: (c_start, c_start + c_len),
            is_out,
            is_fp8: false,
            df: Df::Fp16,
        }
    }

    /// The second chunk of a `[25, 15360]` MLP intermediate — THE measured node: the 12b gemma-4
    /// gated-act product, blocked at nb=8192 into `[0,8192)` + `[8192,15360)`. Descriptor 594
    /// (`594_multiply_o1121`, out=7168) was field-identical to descriptor 593 except width — no
    /// offset anywhere — and `t1121[:, 8192:15360]` read back zero.
    fn mlp_chunks() -> (Vec<Region>, u32, u32, u32) {
        let rows = 25u32;
        let c0 = 8192u32;
        let c1 = 7168u32;
        let r = vec![
            chunked(5, rows, c0, c1, 15360, false),
            chunked(6, rows, c0, c1, 15360, false),
            chunked(7, rows, c0, c1, 15360, true),
        ];
        (r, rows, c0, c1)
    }

    /// ⛔ THE OUTPUT'S PER-CORE START SITS AT THE CHUNK'S STICK-BLOCK CORNER, NOT AT BASE 0.
    /// Column `c` of a `[rows, full_cols]` stick-blocked tensor starts at
    /// `(c/64)·(rows·64)` elements — NOT `c` (the flat index), which is equal only at `rows==1`.
    /// This is the number the card's `startAddressCoreCorelet_` carries, so it is the number the
    /// 12b dump measured zero against.
    #[test]
    fn a_column_blocked_chunk_writes_at_its_stick_block_corner() {
        let (r, rows, c0, c1) = mlp_chunks();
        let mut sym = 0i64;
        let ops = elementwise("mul_s23", Elementwise::Mul, &r, &[], &mut sym, None)
            .expect("a dense column-blocked chunk is exactly the arm that exists for it");
        assert_eq!(ops.len(), 1);
        let dsc = ops[0]
            .op
            .as_ref()
            .expect("a dense elementwise op carries a descriptor")
            .dscs_[0]
            .values()
            .next()
            .expect("one dsc");
        // The output operand is the LAST allocate node (inputs first, output last).
        let out_node = dsc
            .scheduleTree_
            .iter()
            .rfind(|n| n.nodeType_ == "allocate")
            .expect("an allocate node per operand, the output last");
        let start: u64 = out_node
            .startAddressCoreCorelet_
            .data_
            .values()
            .next()
            .expect("a per-core address")
            .parse()
            .expect("a decimal address");
        // The intra-tensor element offset: `(8192/64)·(25·64)` = 128·1600 = 204800 elements = 409600 B.
        // The per-core start folds the (single-core) work-slice offset onto the segment base, so the
        // DELTA from the segment base is the corner — computed here as the difference against the
        // chunk-0 twin's start, which cancels every segment/arg term both share.
        let (r0, ..) = mlp_chunks_at_zero();
        let mut sym0 = 0i64;
        let ops0 = elementwise("mul_s22", Elementwise::Mul, &r0, &[], &mut sym0, None)
            .expect("chunk 0 lowers identically");
        let dsc0 = ops0[0].op.as_ref().expect("descriptor").dscs_[0]
            .values()
            .next()
            .expect("one dsc");
        let out0 = dsc0
            .scheduleTree_
            .iter()
            .rfind(|n| n.nodeType_ == "allocate")
            .expect("output allocate");
        let start0: u64 = out0
            .startAddressCoreCorelet_
            .data_
            .values()
            .next()
            .expect("a per-core address")
            .parse()
            .expect("a decimal address");
        let expect = ((c0 / FP16_ELEMS_PER_STICK) as u64)
            * (rows as u64)
            * (FP16_ELEMS_PER_STICK as u64)
            * 2;
        assert_eq!(
            start - start0,
            expect,
            "chunk 1's output start must sit {expect} B past chunk 0's — the stick-block corner \
             `(c/64)·(rows·64)` of the [25, 15360] intermediate, not the flat {c0}·2 = {} B",
            c0 * 2
        );
        let _ = c1;
    }

    /// Chunk 0 of the same pair — whole-tensor-shaped windows at corner 0 (the control the
    /// delta above cancels against).
    fn mlp_chunks_at_zero() -> (Vec<Region>, u32, u32, u32) {
        let rows = 25u32;
        let r = vec![
            chunked(5, rows, 0, 8192, 15360, false),
            chunked(6, rows, 0, 8192, 15360, false),
            chunked(7, rows, 0, 8192, 15360, true),
        ];
        (r, rows, 0, 8192)
    }

    /// ⛔ AND THE INPUTS READ AT THE SAME CORNER — a multiply whose operands sit at base 0 while
    /// its output writes the corner would recompute chunk 0's values into chunk 1's columns.
    #[test]
    fn a_column_blocked_chunk_reads_its_inputs_at_the_same_corner() {
        let (r, rows, c0, _c1) = mlp_chunks();
        let mut sym = 0i64;
        let ops = elementwise("mul_s23", Elementwise::Mul, &r, &[], &mut sym, None)
            .expect("the chunk lowers");
        let (r0, ..) = mlp_chunks_at_zero();
        let mut sym0 = 0i64;
        let ops0 = elementwise("mul_s22", Elementwise::Mul, &r0, &[], &mut sym0, None)
            .expect("chunk 0 lowers");
        let starts = |ops: &Vec<EmittedOp>| -> Vec<u64> {
            let dsc = ops[0].op.as_ref().expect("descriptor").dscs_[0]
                .values()
                .next()
                .expect("one dsc");
            dsc.scheduleTree_
                .iter()
                .filter(|n| n.nodeType_ == "allocate")
                .map(|n| {
                    n.startAddressCoreCorelet_
                        .data_
                        .values()
                        .next()
                        .expect("a per-core address")
                        .parse::<u64>()
                        .expect("decimal")
                })
                .collect()
        };
        let s1 = starts(&ops);
        let s0 = starts(&ops0);
        assert_eq!(s1.len(), s0.len(), "same operand count");
        let expect = ((c0 / FP16_ELEMS_PER_STICK) as u64)
            * (rows as u64)
            * (FP16_ELEMS_PER_STICK as u64)
            * 2;
        for (a, b) in s1.iter().zip(s0.iter()) {
            assert_eq!(
                a - b,
                expect,
                "every operand — inputs and output alike — moves by the corner"
            );
        }
    }

    /// ⭐ UNEQUAL CORNERS ARE CARRIED PER-OPERAND, not refused: a rotate-half rope pairs the
    /// halves' windows over ONE view (`x1` at columns `[0, 64)` against `x2` at `[64, 128)` of the
    /// same tensor), and each product's operands sit at their OWN corners — no single shared
    /// corner states that. The law here used to refuse every unequal pair, because the tiling
    /// front end mints one window per operand set and nothing else was known to reach the arm;
    /// the per-operand window emission makes the corner an operand fact, and the class law
    /// ([`column_windows_are_honoured`]) is what holds each binding to its own region's corner.
    #[test]
    fn unequal_input_corners_are_carried_per_operand() {
        let rows = 25u32;
        let r = vec![
            chunked(5, rows, 8192, 7168, 15360, false),
            chunked(6, rows, 0, 7168, 15360, false),
            chunked(7, rows, 8192, 7168, 15360, true),
        ];
        let mut sym = 0i64;
        let ops = elementwise("mul_s23", Elementwise::Mul, &r, &[], &mut sym, None)
            .expect("operands at different corners lower, each at its own");
        // The class law holds every binding to its OWN region's corner — t6 at 0, t5 and the
        // output at the stick-block corner of column 8192.
        column_windows_are_honoured("mul_s23", &r, &ops)
            .expect("each operand is addressed at its own corner");
        let off = |ops: &Vec<EmittedOp>, buf: String| {
            ops[0]
                .arg_bindings
                .iter()
                .find(|b| b.buffer == buf)
                .unwrap_or_else(|| panic!("a binding for {buf}"))
                .offset_elems
        };
        assert_eq!(off(&ops, r[1].name()), 0, "t6 reads at its own corner 0");
        let corner = (8192 / FP16_ELEMS_PER_STICK) * rows * FP16_ELEMS_PER_STICK;
        assert_eq!(off(&ops, r[0].name()), corner, "t5 reads at its own corner");
        assert_eq!(
            off(&ops, r[2].name()),
            corner,
            "the output writes its own corner"
        );
    }

    /// ⛔ THE BROADCAST/PER-ROW ARMS CANNOT CARRY A CORNER — refused by name, the same contract
    /// `base_addressed` states for a row corner.
    #[test]
    fn a_column_corner_on_a_broadcast_or_per_row_operand_is_refused() {
        let rows = 25u32;
        // A broadcast operand at a corner (the Col flag shape, input 1 `[25,1]`).
        let rb = vec![
            chunked(5, rows, 8192, 7168, 15360, false),
            chunked(6, rows, 8192, 1, 15360, false),
            chunked(7, rows, 8192, 7168, 15360, true),
        ];
        let mut sym = 0i64;
        let Err(e) = elementwise(
            "sub_s3",
            Elementwise::Sub,
            &rb,
            &[None, Some(BcastAxis::Col)],
            &mut sym,
            None,
        ) else {
            panic!("a broadcast arm at a column corner must refuse")
        };
        assert!(
            e.message.contains("column corner"),
            "names the corner and the arm: {}",
            e.message
        );
        // The per-row chain: a `[rows, 1]` output at a corner (a binary Sub, the softmax's
        // running-subtract shape).
        let pr = vec![
            chunked(5, rows, 8192, 8192, 15360, false),
            chunked(6, rows, 8192, 8192, 15360, false),
            chunked(7, rows, 8192, 1, 15360, true),
        ];
        let Err(e2) = elementwise("sub_s3", Elementwise::Sub, &pr, &[], &mut sym, None) else {
            panic!("the per-row arm at a column corner must refuse")
        };
        assert!(
            e2.message.contains("column corner"),
            "names the corner and the arm: {}",
            e2.message
        );
    }

    /// The lowest per-core start address of each operand's allocate node, in operand order.
    fn operand_bases(ops: &[EmittedOp]) -> Vec<u64> {
        let dsc = ops[0].op.as_ref().expect("descriptor").dscs_[0]
            .values()
            .next()
            .expect("one dsc");
        dsc.scheduleTree_
            .iter()
            .filter(|n| n.nodeType_ == "allocate")
            .map(|n| {
                n.startAddressCoreCorelet_
                    .data_
                    .values()
                    .map(|a| a.parse::<u64>().expect("decimal"))
                    .min()
                    .expect("a per-core address")
            })
            .collect()
    }

    /// ⛔⛔⛔ A COLUMN-CHUNKED SCALARMUL's PIECES TILE ITS ROW EXACTLY ONCE — the granite logits.
    ///
    /// `subtile_ir.rs` column-tiles granite's `logits_scaling` multiply over the lm_head's 49155
    /// columns into `n_blocks(49155, 8192)`: six 8192-wide pieces and a 3-wide tail, sharing one
    /// input and one output tensor. MEASURED on the card's baked suffix (granite-3.1-2b, every
    /// granite suffix dump on the pod): all seven pieces are the SAME descriptor, field for field,
    /// reading and writing columns `[0, 8192)` at one base. So logits `8192..49155` were never
    /// written, and the card never emitted a token id at or above 8192 that a stale byte did not
    /// happen to favour (0.60% of 130,572 emitted ids, against 25.9% of the fp32 references').
    ///
    /// Each piece's input and output must move by its own stick-block corner `(c/64)·(rows·64)`
    /// from piece 0's, the scale must not move, and the corners must cover `[0, 49155)` once, at
    /// the decode row count and at a prompt chunk's.
    #[test]
    fn a_column_chunked_scalarmul_pieces_tile_the_logits_row_once() {
        const VOCAB: u32 = 49155;
        const NB: u32 = 8192;
        for rows in [1u32, 4] {
            let mut pieces = Vec::new();
            let mut c = 0;
            while c < VOCAB {
                let len = NB.min(VOCAB - c);
                let r = vec![
                    chunked(5, rows, c, len, VOCAB, false),
                    chunked(7, rows, c, len, VOCAB, true),
                ];
                let mut sym = 0i64;
                let ops = scalarmul_scaled(
                    "scalarmul_s1",
                    &ScaleSource::Bound { tid: 99 },
                    ScalarOp::Multiply,
                    None,
                    &r,
                    &mut sym,
                    None,
                )
                .unwrap_or_else(|e| panic!("piece at column {c} lowers: {}", e.message));
                assert_eq!(ops.len(), 1, "one descriptor per piece");
                pieces.push((c, len, operand_bases(&ops)));
                c += len;
            }
            assert_eq!(pieces.len(), 7, "six 8192-wide pieces and the 3-wide tail");
            let (_, _, base0) = &pieces[0];
            let bytes_per_col = u64::from(rows) * 2;
            let mut covered = 0u32;
            for (c, len, bases) in &pieces {
                let [x, scale, out] = bases[..] else {
                    panic!("operands [x, scale, out], got {bases:?}")
                };
                let corner = u64::from(*c) * bytes_per_col;
                assert_eq!(
                    x - base0[0],
                    corner,
                    "rows {rows}: the piece at column {c} reads its own columns"
                );
                assert_eq!(scale, base0[1], "rows {rows}: the scale does not move");
                assert_eq!(
                    out - base0[2],
                    corner,
                    "rows {rows}: the piece at column {c} writes its own columns"
                );
                assert_eq!(
                    ((out - base0[2]) / bytes_per_col) as u32,
                    covered,
                    "rows {rows}: each piece starts where the previous one ended"
                );
                covered += len;
            }
            assert_eq!(covered, VOCAB, "rows {rows}: the pieces cover the row once");
        }
    }

    /// ⛔ THE CLASS GUARD SEES THE EMISSION. Every column-tiled body that honours its corner passes
    /// `column_windows_are_honoured` (elementwise, silumul and scalarmul at the MLP chunk, scalarmul at
    /// a logits piece), and the SAME ops with one operand moved back to the buffer base — the shape both
    /// measured defects emitted — are refused naming the op and the tensor.
    #[test]
    fn the_column_window_guard_passes_honoured_pieces_and_refuses_a_piece_at_the_base() {
        let (r, ..) = mlp_chunks();
        let mut sym = 0i64;
        let ew = elementwise("mul_s23", Elementwise::Mul, &r, &[], &mut sym, None).unwrap();
        column_windows_are_honoured("mul_s23", &r, &ew).expect("elementwise honours its window");
        let sm =
            silumul("silumul_s23", &r, &mut sym, None).unwrap_or_else(|e| panic!("{}", e.message));
        column_windows_are_honoured("silumul_s23", &r, &sm).expect("silumul honours its window");
        let logits = vec![
            chunked(5, 1, 8192, 8192, 49155, false),
            chunked(7, 1, 8192, 8192, 49155, true),
        ];
        let mut ops = scalarmul_scaled(
            "scalarmul_s1",
            &ScaleSource::Bound { tid: 99 },
            ScalarOp::Multiply,
            None,
            &logits,
            &mut sym,
            None,
        )
        .unwrap_or_else(|e| panic!("{}", e.message));
        column_windows_are_honoured("scalarmul_s1", &logits, &ops)
            .expect("the logits piece honours its window");
        // The old emission: the output written at the base.
        let out = ops[0]
            .arg_bindings
            .iter_mut()
            .find(|b| b.buffer == logits[1].name())
            .expect("the output's binding");
        out.offset_elems = 0;
        let Err(e) = column_windows_are_honoured("scalarmul_s1", &logits, &ops) else {
            panic!("a piece written at the buffer base must be refused")
        };
        assert!(
            e.message.contains("t7") && e.message.contains("columns 8192..16384"),
            "names the tensor and its window: {}",
            e.message
        );
    }

    /// ⛔⛔⛔ THE TWO SIDES OF A SUBTRACTION ARE DIFFERENT OPERAND ORDERS IN THE DESCRIPTOR —
    /// `C − tile` and `tile − C` are different functions, so the emission must carry the
    /// program's stated order into the operand layout, and this pins that it does: the multiply
    /// (commutative, the card-proven tile-first layout) anchors which base is the tile's and
    /// which the scale's, and the two subtraction orders must come out as exact swaps of that
    /// pair with the output LAST. A body that assumed one order — the doc-comment defect this
    /// wall deleted — would emit one of these two rows wrong and fail here.
    #[test]
    fn the_two_sides_of_a_subtraction_emit_different_operand_orders() {
        let c = -65504.0f32;
        let rows = 4u32;
        let r = vec![
            chunked(5, rows, 0, 64, 64, false),
            chunked(7, rows, 0, 64, 64, true),
        ];
        let mut layout = BundleLayout::default();
        layout.scalarmul_scales.push(c);
        // The operands are graph tensors, so the layout knows them the way the bake does: an
        // Activation placement each (the `graph_layout` law the placement-pass tests state).
        layout.synth(PlaceId::Act(5), &[rows, 64]);
        layout.synth(PlaceId::Act(7), &[rows, 64]);
        // The registry's slot 0 is a device tid like any other — the bake declares its `[1,1]`
        // placement (`scalarmul_scale_tid(i)`), and so does this layout.
        layout.synth(PlaceId::Act(scalarmul_scale_tid(0)), &[1, 1]);
        let mut sym = 0i64;
        let mul = operand_bases(
            &scalarmul_at("mul_o", c, None, &r, &mut sym, Some(&layout))
                .unwrap_or_else(|e| panic!("{}", e.message)),
        );
        let c_first = operand_bases(
            &scalarsub_at("sub_c", c, true, None, &r, &mut sym, Some(&layout))
                .unwrap_or_else(|e| panic!("{}", e.message)),
        );
        let t_first = operand_bases(
            &scalarsub_at("sub_t", c, false, None, &r, &mut sym, Some(&layout))
                .unwrap_or_else(|e| panic!("{}", e.message)),
        );
        let [m0, m1, m2] = mul[..] else {
            panic!("the multiply emits [tile, scale, out], got {mul:?}")
        };
        assert_ne!(
            m0, m1,
            "the tile and the scale must not share a base — the swap would be invisible"
        );
        // `tile − C`: the tile stays first, exactly the multiply's layout.
        assert_eq!(
            &t_first,
            &[m0, m1, m2],
            "tile − C keeps the tile-first order"
        );
        // `C − tile`: the scalar and the tile SWAP; the output stays last.
        assert_eq!(c_first, [m1, m0, m2], "C − tile puts the scalar first");
    }

    /// ⭐ THE CONTROL: chunk 0 — corner 0 on every operand — still takes the SEEDED emission,
    /// byte-identical to every whole-tensor program this crate has ever baked. The offset is the
    /// only thing the chunked arm adds, so a corner-free program must not change form: the seeded
    /// path's `rows > 1` op takes the rank-2 stick-major form (no `y` in any allocate's
    /// `layoutDimOrder_`).
    ///
    /// ⭐ AND SINCE #622 THE CHUNKED PIECE IS RANK-2 TOO. Its corner (column 8192 of 15360) is not a
    /// whole-BLOCK shift, which used to present it rank-3 flat (`y` present) and leave the
    /// work-division law out of reach. It is a whole-STICK column window of a same-rows stick-major
    /// tensor — a stick-major tensor of its own (`tests/zz_a_whole_stick_column_window_is_stick_major.rs`)
    /// — and `elementwise` states it as a `ColWindow`, so it is presented rank-2 as well. What still
    /// tells the two pieces apart is the corner every binding carries, pinned by
    /// `a_column_blocked_chunk_reads_its_inputs_at_the_same_corner`.
    #[test]
    fn a_chunk_at_corner_zero_keeps_the_seeded_emission() {
        let orders = |ops: &Vec<EmittedOp>| -> Vec<Vec<&'static str>> {
            let dsc = ops[0].op.as_ref().expect("descriptor").dscs_[0]
                .values()
                .next()
                .expect("one dsc");
            dsc.scheduleTree_
                .iter()
                .filter(|n| n.nodeType_ == "allocate")
                .map(|n| n.layoutDimOrder_.clone())
                .collect()
        };
        let (r0, ..) = mlp_chunks_at_zero();
        let mut sym = 0i64;
        let ops = elementwise("mul_s22", Elementwise::Mul, &r0, &[], &mut sym, None)
            .expect("chunk 0 lowers");
        assert_eq!(ops.len(), 1);
        for o in orders(&ops) {
            assert!(
                !o.contains(&"y"),
                "a corner-free chunk keeps the seeded rank-2 stick-major form (got {o:?})"
            );
        }
        // And the chunked op — the SAME regions at corner 8192 — is a whole-stick column window, so it
        // is presented rank-2 stick-major too (#622), at its own corner.
        let (r1, ..) = mlp_chunks();
        let mut sym1 = 0i64;
        let ops1 = elementwise("mul_s23", Elementwise::Mul, &r1, &[], &mut sym1, None)
            .expect("chunk 1 lowers");
        for o in orders(&ops1) {
            assert_eq!(
                o,
                vec!["mb", "out"],
                "a whole-stick column window presents rank-2 stick-major (got {o:?})"
            );
        }
        assert!(
            ops1[0].arg_bindings.iter().all(|b| b.offset_elems != 0),
            "every binding of the chunked piece sits at its window's corner"
        );
    }
}

/// ⛔ THE EXTENT GUARD AS A PROPERTY, over SYMBOLIC extents rather than the four examples the test
/// module pins — the guard is the ONE thing standing between a narrow operand and a descriptor that
/// addresses it at the output's width, so "admits exactly the agreeing operand sets" is proved, not
/// sampled.
#[cfg(kani)]
mod proofs {
    use super::*;

    fn reg(tid: u32, rows: u32, cols: u32, is_out: bool) -> Region {
        Region {
            tid,
            v_rows: rows,
            v_cols: cols,
            r_start: 0,
            c_start: 0,
            r_len: rows,
            c_len: cols,
            r_cover: (0, rows),
            c_cover: (0, cols),
            is_out,
            is_fp8: false,
            df: Df::Fp16,
        }
    }

    #[kani::proof]
    fn the_extent_guard_admits_exactly_the_agreeing_operand_sets() {
        let orow: u32 = kani::any();
        let ocol: u32 = kani::any();
        let arow: u32 = kani::any();
        let acol: u32 = kani::any();
        let brow: u32 = kani::any();
        let bcol: u32 = kani::any();
        kani::assume(orow >= 1 && orow <= 64 && arow >= 1 && arow <= 64 && brow >= 1 && brow <= 64);
        kani::assume(ocol >= 1 && ocol <= 256);
        kani::assume(acol >= 1 && acol <= 256 && bcol >= 1 && bcol <= 256);

        let out = reg(7, orow, ocol, true);
        let ins = [reg(5, arow, acol, false), reg(6, brow, bcol, false)];
        // `rows` is the op's sweep basis — for a whole-view output that IS the output's own row
        // count (the `pointwise_rows` read). The guard's two-bases law then reduces, at these
        // whole-view operands, to the property this proof pins: admitted iff every input IS the
        // output's extent.
        let admitted = pointwise_extents_agree("p", Elementwise::Add, &ins, &out, orow).is_ok();
        assert_eq!(
            admitted,
            (arow, acol) == (orow, ocol) && (brow, bcol) == (orow, ocol)
        );
    }

    /// A unary set, so the "every input" quantifier is proved at arity 1 as well as 2.
    #[kani::proof]
    fn the_extent_guard_admits_exactly_the_agreeing_unary_input() {
        let orow: u32 = kani::any();
        let ocol: u32 = kani::any();
        let arow: u32 = kani::any();
        let acol: u32 = kani::any();
        kani::assume(orow >= 1 && orow <= 64 && arow >= 1 && arow <= 64);
        kani::assume(ocol >= 1 && ocol <= 256 && acol >= 1 && acol <= 256);

        let out = reg(7, orow, ocol, true);
        let ins = [reg(5, arow, acol, false)];
        let admitted = pointwise_extents_agree("p", Elementwise::Silu, &ins, &out, orow).is_ok();
        assert_eq!(admitted, (arow, acol) == (orow, ocol));
    }

    /// ⭐ WALL 10's LAW, over the PURE PREDICATE: an operand holds the op's row sweep by BEING
    /// the whole view (`r_start == 0`, `v_rows == rows`) or by its OWN window (`r_len == rows`,
    /// fitting the view) — and by NOTHING else, in particular not the cover. The cover is set to
    /// ARBITRARY symbolic values here (not pinned flush): the wall-10 defect was a flush cover
    /// being read as "the operand is addressed whole", so the regression pin is that NO cover
    /// value moves the verdict. The predicate is pure comparisons — no `format!` in reach, no
    /// maps, no products — so this is the z3 recipe's class (seconds, not minutes).
    ///
    /// ⛔ THE WIRING HALF IS RUNTIME, BY RULING (the law-(a) split): the guard's grid test and
    /// the wall-10 unit test drive the REAL `pointwise_extents_agree` — message selection
    /// included — over the real mask Regions, because the refusal path's `format!` is a
    /// measured CBMC grind (36 min, killed) and the law does not need it to be proved.
    #[kani::proof]
    fn the_two_basis_sweep_law_is_cover_independent() {
        let rows: u32 = kani::any();
        let v_rows: u32 = kani::any();
        let r_start: u32 = kani::any();
        let r_len: u32 = kani::any();
        let cover_lo: u32 = kani::any();
        let cover_hi: u32 = kani::any();
        kani::assume(rows >= 1 && rows <= 64);
        kani::assume(v_rows >= 1 && v_rows <= 256);
        kani::assume(r_len >= 1 && r_len <= 256);
        kani::assume(r_start <= 256);

        let mut x = reg(5, v_rows, 64, false);
        x.r_start = r_start;
        x.r_len = r_len;
        x.r_cover = (cover_lo, cover_hi);
        let verdict = pointwise_operand_holds_sweep(&x, rows);
        // The two bases, stated as the disjunction the law names — and NOTHING else.
        let whole_view = r_start == 0 && v_rows == rows;
        let windowed = r_len == rows && r_start + r_len <= v_rows;
        assert_eq!(verdict, whole_view || windowed);
    }

    /// ⭐ WALL 11's LAW, over the PURE PREDICATE: the reduce family's respelled frame fires
    /// EXACTLY on the family facts — the reduce's own data sweep at matching numel, or a
    /// one-column input at matching rows — and on NOTHING else, in particular not on a WIDE
    /// input's row count (the gather-copy trap: a `[32, 64]` copy result over a `[2048, 64]`
    /// table has numel == the table's rows and must keep its stated shape). The value-count
    /// invariance — a respelled frame holds exactly the stated numel — is the `Some` arm of
    /// the same equivalence: both fire conditions require the match. The predicate is pure
    /// comparisons over `r_len`/`c_len`/`is_out` — no `format!` in reach, no maps, no products
    /// — so this is the z3 recipe's class (seconds, not minutes).
    ///
    /// ⛔ THE WIRING HALF IS RUNTIME, BY RULING (the law-(a) split):
    /// `zz_the_reduce_family_respells_its_frame` drives the REAL mint through `lower_function`
    /// — the reduce, the family members, the placement, and the door's own one-value-per-row
    /// check included — over the real softmax-slice shapes and both controls.
    #[kani::proof]
    fn the_respelled_frame_fires_exactly_on_the_family_facts() {
        let numel: u32 = kani::any();
        let a_len: u32 = kani::any();
        let a_cols: u32 = kani::any();
        let b_len: u32 = kani::any();
        let b_cols: u32 = kani::any();
        let b_out: bool = kani::any();
        let reduce_rows: u32 = kani::any();
        kani::assume(numel >= 1 && numel <= 4096);
        kani::assume(a_len >= 1 && a_len <= 4096);
        kani::assume(b_len >= 1 && b_len <= 4096);
        kani::assume(a_cols >= 1 && a_cols <= 256);
        kani::assume(b_cols >= 1 && b_cols <= 256);
        kani::assume(reduce_rows <= 4096);

        // Two whole-view inputs (the predicate reads `r_len`/`c_len`/`is_out` only) and the
        // reduce's own data sweep; `reduce_rows == 0` models the `None` — a non-reduce op.
        let a = reg(5, a_len, a_cols, false);
        let b = reg(6, b_len, b_cols, b_out);
        let verdict = crate::emit::whole_function::respelled_frame_rows(
            &[a, b],
            numel as u64,
            (reduce_rows >= 1).then_some(reduce_rows),
        );
        // The law, stated as the spec mirror: the reduce's matching sweep first, then the
        // FIRST one-column input at matching rows, and nothing else.
        let expect = if reduce_rows >= 1 && reduce_rows == numel {
            Some(reduce_rows)
        } else if a_cols == 1 && a_len == numel {
            Some(a_len)
        } else if !b_out && b_cols == 1 && b_len == numel {
            Some(b_len)
        } else {
            None
        };
        assert_eq!(verdict, expect);
    }

    /// ⭐ WALL 12's AXIS LAW, over the PURE PREDICATE: the block a reduce's `Dimensions` state
    /// fires EXACTLY on the stated block facts — a leading prefix (and not the whole rank), a
    /// trailing block, or neither — and on NOTHING else. The tie (a block that is both, the
    /// whole rank) is Trailing by the no-drift law, and a malformed statement (no dims, more
    /// dims than the rank) is Middle.
    ///
    /// ⛔ SECONDS-CLASS OR NOTHING (user law, 2026-10-22: "10 minute kani proofs are of no
    /// value" — the user killed this harness's first encoding themselves). The killed encoding
    /// mirrored the predicate through `to_vec` + `sort_unstable`: introsort's swap loops unwind
    /// over a SYMBOLIC length and CBMC produced no verdict in 10 minutes. This encoding is the
    /// wall-10/11 recipe: the predicate states the set fact by RANGE MEMBERSHIP + PAIRWISE
    /// DISTINCTNESS, the mirror below states the SAME fact by PER-VALUE COUNTING — two genuinely
    /// different encodings of one theorem, both plain bounded loops over n ≤ 4, both unwound at
    /// 8 (n ≤ 4 + headroom), no Vec, no sort, no closures, no enum forests.
    #[kani::proof]
    #[kani::unwind(8)]
    fn the_nonstick_axes_fire_exactly_on_the_block_facts() {
        let d0: i64 = kani::any();
        let d1: i64 = kani::any();
        let d2: i64 = kani::any();
        let d3: i64 = kani::any();
        let n: u32 = kani::any();
        let rank: u32 = kani::any();
        kani::assume(d0 >= 0 && d0 <= 8);
        kani::assume(d1 >= 0 && d1 <= 8);
        kani::assume(d2 >= 0 && d2 <= 8);
        kani::assume(d3 >= 0 && d3 <= 8);
        kani::assume(n <= 4);
        kani::assume(rank <= 8);

        let dims = [d0, d1, d2, d3];
        let dims = &dims[..n as usize];
        let verdict = reduce_axis_block(dims, rank);

        // The spec mirror, as COUNTING: the block is the prefix `[0, n)` resp. the trailing
        // `[rank−n, rank)`, and it is stated iff EVERY value of the range appears among the dims
        // EXACTLY ONCE — n values, n range members, each once, so nothing lies outside the range
        // either. A different encoding than the predicate's range+distinctness form.
        let expect = if n == 0 || n > rank {
            ReduceAxisBlock::Middle
        } else {
            let mut prefix_ok = true;
            let mut trailing_ok = true;
            for v in 0..n as i64 {
                let mut c = 0u32;
                let mut ct = 0u32;
                for i in 0..n as usize {
                    if dims[i] == v {
                        c += 1;
                    }
                    if dims[i] == (rank as i64 - n as i64) + v {
                        ct += 1;
                    }
                }
                if c != 1 {
                    prefix_ok = false;
                }
                if ct != 1 {
                    trailing_ok = false;
                }
            }
            if prefix_ok && !trailing_ok {
                ReduceAxisBlock::LeadingPrefix
            } else if trailing_ok {
                ReduceAxisBlock::Trailing
            } else {
                ReduceAxisBlock::Middle
            }
        };
        assert_eq!(verdict, expect);
        // DISPATCH DISJOINTNESS, stated on its own: the nonstick arm fires only on a block that
        // is NOT the trailing block — a dim set that is both (the whole rank, reduced to a
        // scalar) is the row-reduce path's, by the no-drift tie law — and never on an empty or
        // oversized statement.
        if matches!(verdict, ReduceAxisBlock::LeadingPrefix) {
            assert!(!trailing_stated(dims, rank));
            assert!(n >= 1 && n <= rank);
        }
    }

    /// The mirror's trailing fact, factored out so the disjointness assert states it once: every
    /// value of `[rank−n, rank)` appears among `dims` exactly once (the counting form).
    fn trailing_stated(dims: &[i64], rank: u32) -> bool {
        let n = dims.len();
        if n == 0 || n as u32 > rank {
            return false;
        }
        for v in 0..n as i64 {
            let mut ct = 0u32;
            for i in 0..n {
                if dims[i] == (rank as i64 - n as i64) + v {
                    ct += 1;
                }
            }
            if ct != 1 {
                return false;
            }
        }
        true
    }

    /// ⭐ WALL 12's GEOMETRY LAW (WALL 13: df-parameterized), over the PURE PREDICATE: the
    /// nonstick arm fires EXACTLY on `out_numel` dividing `in_numel`, a reduced extent of at
    /// least 2, and a whole-stick survivor width AT THE DATA'S OWN `df` — and the firing
    /// geometry carries the BYTE LAW as arithmetic: the product is `in_numel` EXACTLY (no
    /// padding hides in it), every `(reduced, survivor)` pair maps to a distinct flat offset
    /// inside the tensor, and the map is `reduced · survivor_cols + survivor` — survivors
    /// stride 1, the nonstick data law. The `df` is SYMBOLIC over the two bases the reduce
    /// family's DDLs bind (`Fp16` 64 lanes, `Fp32` 32 lanes — `summeanmaxexx2.ddl` /
    /// `summeanmaxexx2_fp32.ddl`), so the proof pins that the basis is the data's own and not
    /// a knob: the SAME numels fire at one df and refuse sub-stick at the other, exactly as
    /// the boundary pin's f32/f16 pair states at runtime.
    ///
    /// ⛔ SECONDS-CLASS OR NOTHING (the same user law as the axes harness): the universe is
    /// bounded at 4096 so the two symbolic-operand `%`/`/` stay in the solver's cheap range —
    /// every wall-12 firing geometry (the epilogue merges and their rungs) lives far inside it,
    /// and the law is scale-free arithmetic (divisibility, quotient, product), not a bound
    /// checked at the frontier.
    #[kani::proof]
    fn the_nonstick_geometry_fires_exactly_on_the_divisible_stick_facts() {
        let in_numel: u64 = kani::any();
        let out_numel: u64 = kani::any();
        let is_fp32: bool = kani::any();
        kani::assume(in_numel >= 1 && in_numel <= 4096);
        kani::assume(out_numel <= 4096);
        let df = if is_fp32 { Df::Fp32 } else { Df::Fp16 };

        let verdict = nonstick_geometry(in_numel, out_numel, df);

        // The spec mirror, stated as the law names it — the stick basis is the data's own df.
        let divisible = out_numel >= 1 && in_numel % out_numel == 0;
        let expect = if !divisible {
            NonstickGeometry::Indivisible
        } else if in_numel == out_numel {
            NonstickGeometry::RemovesNothing
        } else if out_numel % df.elems_per_stick() as u64 != 0 {
            NonstickGeometry::SubStick(out_numel as u32)
        } else {
            NonstickGeometry::Lowers((in_numel / out_numel) as u32, out_numel as u32)
        };
        assert_eq!(verdict, expect);

        // THE BYTE LAW, as arithmetic, over symbolic in-bounds coordinates.
        if let NonstickGeometry::Lowers(reduced_rows, survivor_cols) = verdict {
            // The product is exact — the window the descriptor declares IS the tensor.
            assert_eq!(reduced_rows as u64 * survivor_cols as u64, in_numel);
            let r_i: u32 = kani::any();
            let s_i: u32 = kani::any();
            let r_j: u32 = kani::any();
            let s_j: u32 = kani::any();
            kani::assume(r_i < reduced_rows && s_i < survivor_cols);
            kani::assume(r_j < reduced_rows && s_j < survivor_cols);
            let flat_i = r_i as u64 * survivor_cols as u64 + s_i as u64;
            let flat_j = r_j as u64 * survivor_cols as u64 + s_j as u64;
            // In bounds...
            assert!(flat_i < in_numel);
            // ...and injective — distinct coordinates are distinct bytes, so the descriptor
            // reads and writes every element exactly once.
            if (r_i, s_i) != (r_j, s_j) {
                assert_ne!(flat_i, flat_j);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PER-OP ENTRY POINTS THE WHOLE-FUNCTION WALK DISPATCHES TO.
//
// Each is the `_at` form of an existing program: same lowering, but reading its own
// constants off the op it is given rather than off a node that states one program for
// the whole function. `reduce` is the bare `linalg.reduce` case, which had no door.
// ─────────────────────────────────────────────────────────────────────────────

/// [`rmsnorm`] with the epsilon SUPPLIED rather than read off the whole function.
///
/// ⭐ WHY THE SPLIT EXISTS — the same reason as [`scalarmul_at`]'s. [`program_rmsnorm_eps`] enforces
/// ONE root `math.sqrt`/`math.rsqrt` in the function, which is exact for a function that IS one
/// rmsnorm node and returns `None` for any whole kernel with two of them. `decoder_layer_fwd` has
/// exactly two rmsnorms, so its epsilons can only be read PER CHAIN — which is what
/// [`super::whole_function::program_rmsnorm_chains`] does, at the `arith.addf` each chain owns.
///
/// The body below is shared, so the two doors cannot drift about what an rmsnorm EMITS.
pub fn rmsnorm_at(
    name: &str,
    eps: f32,
    ones: Option<f32>,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    // x, gamma and the output — the parameters the rmsnorm program mints. Its constants are
    // immediates, so none of them is a parameter.
    let (tensors, out) = split_out(name, r, layout, 2)?;
    let eps_idx = scale_slot(layout, eps).ok_or_else(|| Error {
        message: format!(
            "RmsNorm {name}: epsilon {eps}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor adds it as a bound `[1,1]` const, so \
             the value the program uses must have a registry slot (registry desync)"
        ),
    })?;
    check_pointwise_cols(out.c_len, out.df, "RmsNorm", out.tid)?;
    let rows = node_rows(name, &out)?;
    let cols = out.c_len;
    // rms_norm_eps (config) flows via the scalarmul registry (compute_bundle_layout collected it) —
    // the same reserved `[1,1]` const `subtile→superdsc` adds to the mean-of-squares.
    let eps_const = crate::place::act_name(scalarmul_scale_tid(eps_idx));
    // ⭐ THE GAIN OFFSET, THREADED FROM THE CHAIN READER — the arm this door's own comment once
    // demanded: `program_rmsnorm_chains` now reads each chain's gamma leg the way it reads its
    // epsilon, so a whole kernel with a OnePlusScale rmsnorm hands this door the offset. The
    // value is the chain reader's (already refused everything but the convention's own 1.0), the
    // slot is resolved BY THAT VALUE exactly as the epsilon's is, and the emission is the SAME
    // `assemble_rmsnorm` one-plus row the per-`Program` door emits — ONE `add` ahead of the gain
    // multiply, the formula the program runs, op for op.
    let ones_const = match ones {
        None => None,
        Some(1.0) => {
            let ones_idx = scale_slot(layout, 1.0).ok_or_else(|| Error {
                message: format!(
                    "RmsNorm {name}: the program states gain offset 1.0 (the OnePlusScale \
                     convention), which is absent from `BundleLayout::scalarmul_scales` — the \
                     descriptor adds it as a bound `[1,1]` const ahead of the gain multiply, so \
                     the value the program uses must have a registry slot (registry desync)"
                ),
            })?;
            Some(crate::place::act_name(scalarmul_scale_tid(ones_idx)))
        }
        Some(other) => {
            return Err(Error {
                message: format!(
                    "RmsNorm {name}: the program states gain offset {other}, which is not a \
                     gain convention — `GainConvention::from_offset` admits 0.0 (Scale) and 1.0 \
                     (OnePlusScale) only, so a program stating any other offset is a producer bug"
                ),
            });
        }
    };
    let x = tensors[0].name();
    let gamma = tensors[1].name();
    let t = out.tid;
    Ok(assemble_rmsnorm(
        RmsnormFacts {
            prefix: &format!("o{t}"),
            rows,
            cols,
            x: &x,
            gamma: &gamma,
            out_id: PlaceId::Act(t),
            eps_const: &eps_const,
            ones_const: ones_const.as_deref(),
        },
        sym_id_base,
        layout,
    ))
}

/// The per-`Program` door for [`crate::ktir_node::Program::RmsNormUnit`] —
/// `out = x · rsqrt(mean(x²) + eps)`, no gain.
///
/// Same shape as [`rmsnorm`]: the epsilon is read off the program (the value the
/// emulator adds), resolved to its registry slot by that value, and the body is
/// [`assemble_rmsnorm_unit`] — the first five steps of the gained form with the
/// normalising multiply writing `out` directly. ONE tensor parameter where
/// `rmsnorm` splits two, which is the only structural difference.
pub fn rmsnorm_unit(
    name: &str,
    k: &KtirNode,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    // x and the output — the parameters `KtirFunc::rmsnorm_unit` mints.
    let (tensors, out) = split_out(name, r, layout, 1)?;
    let eps = program_rmsnorm_eps(&k.func).ok_or_else(|| Error {
        message: format!(
            "RmsNormUnit {name}: the program states no epsilon. `KtirFunc::rmsnorm_unit` splats \
             it into the `arith.addf` that feeds its one root op, and the descriptor's `[1,1]` \
             const is resolved from that value, so a program without it cannot be lowered."
        ),
    })?;
    let eps_idx = scale_slot(layout, eps).ok_or_else(|| Error {
        message: format!(
            "RmsNormUnit {name}: epsilon {eps}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor adds it as a bound `[1,1]` const, so \
             the value the program uses must have a registry slot (registry desync)"
        )
    })?;
    check_pointwise_cols(out.c_len, out.df, "RmsNormUnit", out.tid)?;
    let rows = node_rows(name, &out)?;
    let cols = out.c_len;
    let eps_const = crate::place::act_name(scalarmul_scale_tid(eps_idx));
    let x = tensors[0].name();
    let t = out.tid;
    Ok(assemble_rmsnorm_unit(
        RmsnormUnitFacts {
            prefix: &format!("o{t}"),
            rows,
            cols,
            x: &x,
            out_id: PlaceId::Act(t),
            eps_const: &eps_const,
        },
        sym_id_base,
        layout,
    ))
}

/// The per-`Program` door for [`crate::ktir_node::Program::WindowedRmsNorm`] — the
/// FUSED per-head norm sandwich (`Reshape{Times(H)} → norm → Reshape{Times(1)}`).
///
/// `assemble_rmsnorm`'s chain applied per head: for each head `h`, the SAME six ops
/// over the `[m, D]` window at base offset `h·D·m` of the original `[m, H·D]` tensor —
/// which the head-window/block-alignment law (see the `Program` variant's docs) places
/// at exactly the elements the deleted split+flatten computed. The door derives every
/// extent from the program's own views, the way the attention door derives its geometry:
/// `m` and `H·D` off the output's view, `D` off the output's FIRST ACCESS TILE (one
/// head's window — the tile `KtirFunc::windowed_rmsnorm` states), `H = (H·D)/D`.
///
/// ⭐ THE SYNTHETICS ARE DECLARED ONCE, AT THE FULL RE-LAID FOOTPRINTS — `Sq16`/`Xn` at
/// `[H·m, D]`, `Mean`/`Meps`/`Rinv` at `[H·m, stick]` — and addressed per head at
/// `h·m·D` / `h·m·stick` base offsets: the writer and the reader of each synthetic use
/// the SAME per-head offset, so the layout they agree on is H consecutive per-head
/// blocks (the same nesting the un-fused form's one `[H·m, D]` norm used, which is why
/// the memory plan is comparable rather than larger).
///
/// ⭐ THE EPSILON comes from [`program_windowed_eps`] — the read-all-roots-agree form —
/// because this program states one root per head; the whole-region reader would refuse
/// it. Resolved to the registry slot by bits, exactly as every other norm constant.
///
/// ⭐ THE GAIN OFFSET RIDES THE EPSILON'S LAW: the OnePlusScale program states it
/// as ONE `arith.addf` over the loaded gamma row (`windowedrmsnorm_oneplus_fwd`'s
/// `w = load + GAIN_OFFSET`), the door resolves the SAME value's `[1,1]` registry
/// slot by bits, and ONE shared `add` ahead of the head loop fills the `[1, D]`
/// `Gw` row every per-head gain multiply reads — the gamma row is shared by all
/// heads, so the offset is applied once, not per head. Gemma-4's per-head q/k
/// norms are the `Scale` convention (measured: the bake's `rmsnorm_s8` states no
/// gain-offset addf), so they lower with no add at all — byte-identical to the
/// form before this row existed.
pub fn windowed_rmsnorm(
    name: &str,
    k: &KtirNode,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    use crate::ir::bridge::tiled_op_sdsc_op::{
        assemble_reduce_off, win_reduce_accum, win_reduce_data,
    };
    let unit = matches!(
        k.program,
        crate::ktir_node::Program::WindowedRmsNorm { unit: true }
    );
    // x (+ gamma in the gained form) and the output — the parameters
    // `KtirFunc::windowed_rmsnorm` mints, in first-use order.
    let arity = if unit { 1 } else { 2 };
    let (tensors, out) = split_out(name, r, layout, arity)?;
    let eps = program_windowed_eps(&k.func).ok_or_else(|| Error {
        message: format!(
            "WindowedRmsNorm {name}: the program states no epsilon. `KtirFunc::windowed_rmsnorm` \
             splats it into the `arith.addf` feeding each head's `math.sqrt` root — one root per \
             head, all agreeing — and the descriptor's `[1,1]` const is resolved from that value, \
             so a program without it cannot be lowered."
        ),
    })?;
    let eps_idx = scale_slot(layout, eps).ok_or_else(|| Error {
        message: format!(
            "WindowedRmsNorm {name}: epsilon {eps}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor adds it as a bound `[1,1]` const, so \
             the value the program uses must have a registry slot (registry desync)"
        ),
    })?;
    // The door's own geometry, off the program's views: the output view states `[m, H·D]`
    // and its first access tile is ONE head's `[m, D]` window.
    let m = node_rows(name, &out)?;
    let full = out.v_cols;
    let d = out.c_len;
    if d == 0 || full % d != 0 {
        return err(format!(
            "{name}: the output t{}'s view states [{m}, {full}] but its first access tile is \
             [{}, {d}] — a windowed norm's tile is one head's width and must divide the view's",
            out.tid, out.r_len,
        ));
    }
    let heads = full / d;
    // ⭐ THE GAIN OFFSET, OFF THE PROGRAM — the plain door's own match, stated here
    // so the two doors cannot drift about what an offset IS. `None` is the Scale
    // program (no addf), `Some(1.0)` is the OnePlusScale convention (the census
    // collected that exact value for this node), and any other value is refused by
    // name — not a convention `GainConvention::from_offset` admits.
    let gw_ones = if unit {
        None
    } else {
        match program_gain_offset(&k.func) {
            None => None,
            Some((1.0, _)) => {
                let ones_idx = scale_slot(layout, 1.0).ok_or_else(|| Error {
                    message: format!(
                        "WindowedRmsNorm {name}: the program states gain offset 1.0 (the \
                         OnePlusScale convention), which is absent from \
                         `BundleLayout::scalarmul_scales` — the descriptor adds it as a bound \
                         `[1,1]` const ahead of the gain multiply, so the value the program \
                         uses must have a registry slot (registry desync)"
                    ),
                })?;
                Some(crate::place::act_name(scalarmul_scale_tid(ones_idx)))
            }
            Some((other, _)) => {
                return err(format!(
                    "{name}: the program states gain offset {other}, which is not a gain \
                     convention — `GainConvention::from_offset` admits 0.0 (Scale) and 1.0 \
                     (OnePlusScale) only, so a program stating any other offset is a producer bug"
                ));
            }
        }
    };
    check_pointwise_cols(d, out.df, "WindowedRmsNorm", out.tid)?;
    let stk = crate::superdsc_opspec::Fp16::ELEMS_PER_STICK;
    let eps_const = crate::place::act_name(scalarmul_scale_tid(eps_idx));
    let x = tensors[0].name();
    let t = out.tid;
    let out_id = PlaceId::Act(t);
    use crate::place::SynthRole as R;
    let syn = |role: R| crate::placement::syn(layout, out_id.synth(role));
    // ── THE SYNTHETICS ──
    //
    // `Sq16` (and `Xn` in the gained form) are ONE `[m, H·D]` buffer each — the same
    // footprint the un-fused form's `[H·m, D]` re-laid intermediate had — addressed per
    // head at the window offset, which the block-alignment law keeps on the stick-major
    // rank-2 path at prefill (writer and reader of each use the SAME offset).
    //
    // ⭐ THE REDUCED SCALARS ARE PER-HEAD BUFFERS AT OFFSET ZERO (`HeadMean`/`HeadMeps`/
    // `HeadRinv`, the fp8 amax-partial precedent): a reduce's accum must sit at a
    // whole-block shift of its own slab for rank-2, and a shared `[H·m, stick]` buffer's
    // `h·m·stick` offsets are not block-aligned at the reduce's `[m, D]` block — the flat
    // rank-3 fallback would then read the stick-major `sq16` write scrambled. One buffer
    // per head makes every scalar-chain op offset-free.
    let sq16 = syn(R::Sq16);
    let xn = syn(R::Xn);
    let gw = syn(R::Gw); // γ+1   [1, d]  — OnePlusScale only
    if let Some(l) = layout {
        l.synth(out_id.synth(R::Sq16), &[m, full]);
        if !unit {
            l.synth(out_id.synth(R::Xn), &[m, full]);
        }
        if gw_ones.is_some() {
            l.synth(out_id.synth(R::Gw), &[1, d]);
        }
        for h in 0..heads {
            l.synth(out_id.synth(R::HeadMean(h)), &[m, stk]);
            l.synth(out_id.synth(R::HeadMeps(h)), &[m, stk]);
            l.synth(out_id.synth(R::HeadRinv(h)), &[m, stk]);
        }
    }
    // Handle kinds: `x`/`out`/`sq16`/`xn` are the residual stream's RowBlocked; the
    // reduced scalars are one-stick-wide. `assemble_rmsnorm`'s comment carries the law.
    let t_rows = RowCount::of_token_rows(m);
    let full_cols = BlockCols::of_feature_cols(full);
    let f_cols = BlockCols::of_feature_cols(d);
    let s_cols = BlockCols::of_one_stick(Lanes::FP16);
    let x_h = rb(&x, t_rows, full_cols);
    let out_h = rb(&out.name(), t_rows, full_cols);
    let sq16_h = rb(&sq16, t_rows, full_cols);
    let xn_h = rb(&xn, t_rows, full_cols);
    let eps_h = rb(&eps_const, RowCount::of_broadcast_row(), s_cols);
    let gamma_h = if unit {
        None
    } else {
        Some(rb(&tensors[1].name(), RowCount::of_broadcast_row(), f_cols))
    };
    // The OnePlusScale gain row: `gw = gamma + 1` filled ONCE, ahead of the head
    // loop — the gamma row is shared by every head, and so is its offset sum. The
    // add reads the `[1,1]`-broadcast offset (the eps operand mode) into the
    // `[1, d]` `Gw` row the per-head gain multiplies read.
    let gw_h = gw_ones
        .as_ref()
        .map(|_| rb(&gw, RowCount::of_broadcast_row(), f_cols));
    let ones_h = gw_ones
        .as_ref()
        .map(|c| rb(c, RowCount::of_broadcast_row(), s_cols));
    let mut ops: Vec<EmittedOp> = Vec::new();
    if let (Some(g), Some(ones), Some(gw)) = (&gamma_h, &ones_h, &gw_h) {
        ops.push(pw2(
            &format!("rmga_o{t}"),
            "add",
            crate::sdsc_abstract::RowCount::of_token_rows(1),
            f_cols,
            In::mb(g),
            In::scalar(ones),
            gw,
            sym_id_base,
            layout,
        )); // gw = γ + 1, shared by every head
    }
    for h in 0..heads {
        // The head window's base offset in the `[m, H·D]` buffers — a whole-block shift of
        // this op's own `[m, D]` block (the block-alignment law).
        let win = crate::addr::col_of(m, full, h * d, crate::superdsc_opspec::Df::Fp16);
        // The per-head reduced scalars — their OWN buffers, addressed at ZERO.
        let mean_h = rb(&syn(R::HeadMean(h)), t_rows, s_cols);
        let meps_h = rb(&syn(R::HeadMeps(h)), t_rows, s_cols);
        let rinv_h = rb(&syn(R::HeadRinv(h)), t_rows, s_cols);
        let prefix = format!("wh{h}_o{t}");
        // 1. sq16 = x² over the head window (out at the window — `pw2`'s ZERO would
        //    make every head overwrite head 0's block).
        ops.push(assemble_pointwise_broadcast_off(
            PointwiseFacts {
                op_name: &format!("rmsq_{prefix}"),
                op_func: "mul",
                rows: t_rows,
                cols: f_cols,
                inputs: &[In::sliced(&x_h, win).ew(), In::sliced(&x_h, win).ew()],
                o: &sq16_h,
                out_offset: win.into(),
            },
            sym_id_base,
            layout,
        ));
        // 2. mean(x²) — the per-head reduce: data at the window (block-aligned), accum at
        //    its own head's buffer at ZERO (the fp8 amax-partial shape).
        let rmmean_op = format!("rmmean_{prefix}");
        ops.push(assemble_reduce_off(
            &rmmean_op,
            "mean",
            t_rows,
            f_cols,
            // The data window is head `h`'s `[m, d]` slab of the `[m, full]` square — `col_of`'s
            // whole-block shift, minted against the buffer's own handle.
            win_reduce_data(
                layout,
                &sq16_h,
                win,
                t_rows,
                f_cols,
                &format!("{rmmean_op}: squares"),
            )
            .map_err(|e| Error { message: e })?,
            win_reduce_accum(
                layout,
                &mean_h,
                crate::addr::DevOff::ZERO,
                t_rows,
                &format!("{rmmean_op}: mean"),
            )
            .map_err(|e| Error { message: e })?,
            sym_id_base,
            layout,
        ));
        // 3. mean + eps.
        ops.push(pw2(
            &format!("rmeps_{prefix}"),
            "add",
            t_rows,
            s_cols,
            In::full(&mean_h),
            In::scalar(&eps_h),
            &meps_h,
            sym_id_base,
            layout,
        ));
        // 4. rinv = rsqrt(mean+eps).
        ops.push(pw1(
            &format!("rmrsqrt_{prefix}"),
            "rsqrt",
            t_rows,
            s_cols,
            In::full(&meps_h),
            &rinv_h,
            sym_id_base,
            layout,
        ));
        // 5. xn = x·rinv (r per-row of the head's block, out-broadcast over D) — the
        //    unit form's TERMINAL multiply writes `out` directly here instead.
        // 6. out = xn·gamma (the `[1, D]` row, mb-broadcast) — the Gw row (γ+1)
        //    when the convention states an offset, the stored row when it does not.
        if let Some(g) = &gamma_h {
            ops.push(assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("rmxn_{prefix}"),
                    op_func: "multiply",
                    rows: t_rows,
                    cols: f_cols,
                    inputs: &[In::sliced(&x_h, win).ew(), In::col(&rinv_h).ew()],
                    o: &xn_h,
                    out_offset: win.into(),
                },
                sym_id_base,
                layout,
            ));
            let gain = gw_h.as_ref().unwrap_or(g);
            ops.push(assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("rmg_{prefix}"),
                    op_func: "multiply",
                    rows: t_rows,
                    cols: f_cols,
                    inputs: &[In::sliced(&xn_h, win).ew(), In::mb(gain).ew()],
                    o: &out_h,
                    out_offset: win.into(),
                },
                sym_id_base,
                layout,
            ));
        } else {
            ops.push(assemble_pointwise_broadcast_off(
                PointwiseFacts {
                    op_name: &format!("rmxn_{prefix}"),
                    op_func: "multiply",
                    rows: t_rows,
                    cols: f_cols,
                    inputs: &[In::sliced(&x_h, win).ew(), In::col(&rinv_h).ew()],
                    o: &out_h,
                    out_offset: win.into(),
                },
                sym_id_base,
                layout,
            ));
        }
    }
    Ok(ops)
}

/// The per-`Program` door for [`crate::ktir_node::Program::TanhSoftCap`] —
/// `out = cap · tanh(x / cap)`, Gemma's final-logit soft cap.
///
/// Same shape as [`rmsnorm_unit`]: the cap is read off the program (the value
/// the emulator divides and multiplies by), resolved to its registry slot by
/// that value, and the body is [`assemble_tanhsoftcap`] — ONE `realdiv` by the
/// bound `[1,1]` cap const, ONE `tanh`, ONE `multiply` by the same const.
/// ONE tensor parameter, exactly like the unit norm.
pub fn tanhsoftcap(
    name: &str,
    k: &KtirNode,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    // x and the output — the parameters `KtirFunc::tanhsoftcap` mints.
    let (tensors, out) = split_out(name, r, layout, 1)?;
    let cap = program_tanhsoftcap_cap(&k.func).ok_or_else(|| Error {
        message: format!(
            "TanhSoftCap {name}: the program states no cap. `KtirFunc::tanhsoftcap` splats \
             it as the divisor of the `arith.divf` that feeds its one `math.tanh` root, and \
             the descriptor's `[1,1]` const is resolved from that value, so a program \
             without it cannot be lowered."
        ),
    })?;
    let cap_idx = scale_slot(layout, cap).ok_or_else(|| Error {
        message: format!(
            "TanhSoftCap {name}: cap {cap}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor divides and multiplies by a \
             bound `[1,1]` const, so the value the program uses must have a registry slot \
             (registry desync)"
        ),
    })?;
    check_pointwise_cols(out.c_len, out.df, "TanhSoftCap", out.tid)?;
    let rows = node_rows(name, &out)?;
    let cols = out.c_len;
    let cap_const = crate::place::act_name(scalarmul_scale_tid(cap_idx));
    let x = tensors[0].name();
    let t = out.tid;
    Ok(assemble_tanhsoftcap(
        TanhsoftcapFacts {
            prefix: &format!("o{t}"),
            rows,
            cols,
            x: &x,
            out_id: PlaceId::Act(t),
            cap_const: &cap_const,
        },
        sym_id_base,
        layout,
    ))
}

/// The per-`Program` door for [`crate::ktir_node::Program::ScalarWeightMul`] —
/// `out = x · w`, `w` a host-staged `[1]`-shaped weight (gemma4
/// `layer_scalar[layer]`).
///
/// ONE pointwise `multiply` whose second operand is the weight buffer read in
/// the SCALAR broadcast mode (`In::scalar`) — exactly the operand mode
/// [`scalarmul_at`] gives its registry const and `assemble_rmsnorm` gives its
/// epsilon, over a weight the WORKER stages from the checkpoint instead of a
/// const the bake binds. The weight's placement pads its `[1]` extent to a
/// whole stick (`nbytes` → `[1, 64]` f16 = 128 B), so the one-stick scalar
/// read is in bounds by construction.
pub fn scalarweightmul(
    name: &str,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    // x, the weight, and the output — the parameters `KtirFunc::scalar_weight_mul` mints.
    let (tensors, out) = split_out(name, r, layout, 2)?;
    let rows = node_rows(name, &out)?;
    pointwise_extents_agree(name, Elementwise::Mul, &tensors[0..1], &out, rows)?;
    check_pointwise_cols(out.c_len, out.df, "ScalarWeightMul", out.tid)?;
    let cols = out.c_len;
    let x = tensors[0].name();
    let w = tensors[1].name();
    let t = out.tid;
    // ⭐ TRUE EXTENTS, not `rbo`'s `1×64` annotation: the op sweeps `[rows, cols]` (the extents the
    // `RowCount`/`BlockCols` slots below state), so x and the output declare exactly that — the
    // mint's bound. The builder reads name + kind only.
    let x_h = rb(
        &x,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    // The weight is `[1]` on disk and `[1, stick]` in its placement — the same
    // one-stick shape the rmsnorm's `[1,1]` epsilon const takes, read in the
    // same `In::scalar` broadcast mode (lane 0 sprayed over rows and cols).
    let w_h = rb(
        &w,
        RowCount::of_broadcast_row(),
        BlockCols::of_one_stick(Lanes::FP16),
    );
    let out_h = rb(
        &out.name(),
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(cols),
    );
    let op_name = format!("scalarmul_o{t}");
    Ok(vec![pw2(
        &op_name,
        "multiply",
        crate::sdsc_abstract::RowCount::of_token_rows(rows),
        crate::sdsc_abstract::BlockCols::of_feature_cols(cols),
        In::full(&x_h),
        In::scalar(&w_h),
        &out_h,
        sym_id_base,
        layout,
    )])
}

/// The per-`Program` door for [`crate::ktir_node::Program::Reshape`] — a RE-LAYING COPY that
/// re-sticks a whole buffer from `[r_in, c_in]` to `[r_out, c_out]` preserving the flat element
/// sequence.
///
/// ⛔⛔⛔ THE VIEW CANNOT BE AN ALIAS, AND THE LAW IS ARITHMETIC, NOT A PREFERENCE. A device
/// tensor is stick-blocked: logical `(i, j)` sits at `dev_off([rows, cols], ·) =
/// (j/stk)·(rows·stk) + i·stk + (j%stk)` (see [`crate::sdsc_abstract`]), where `rows` is the
/// tensor's own ROW COUNT. Two views over one buffer with different extents therefore disagree
/// about where EVERY element lives — and `declare_arrangement` cannot catch the alias, because
/// it keys on tensor NAME and an alias gives the two views two names. So the reshape is a real
/// copy, and this door is the only spelling of it.
///
/// ⭐⭐ THE DECOMPOSITION IS SINGLE-STICK IDENTITY COPIES — the proven `lmlast` mechanism, and
/// for the same reason its own comment gives: a whole stick is the one rectangle whose address is
/// a corner a rank-2 view can name. Walk the flat sequence a stick at a time. Stick `q` covers
/// flat elements `64q..64q+64`; on the input those are `r = 64q / c_in`, `j = 64q % c_in`
/// (one or two adjacent column sticks — a stick never straddles three, because a column stick
/// IS 64 elements), and on the output `ρ = 64q / c_out`, `d = 64q % c_out`. Every address is an
/// [`rc_of`] corner of that side's OWN view, so the emitter never writes a hand-derived
/// multiplier: the nest law IS the address. At `m=1` (decode) the input is one flat row and
/// this degenerates to a row-blocked re-blocking — exactly the shape gemma4's per-head
/// `[1, H·D] → [H, D]` views and their flatten-backs take.
///
/// ⛔ THE STICK LAWS, BOTH SIDES, BECAUSE A PARTIAL STICK IS AN ADDRESS A VIEW CANNOT NAME.
/// `c_in` and `c_out` must each be whole 64-stick multiples (the producer's column widths are
/// feature widths, which are); the element counts must agree exactly (the SubOp's own contract,
/// re-stated here because a descriptor pair that moved a different number of elements than the
/// program claims would be silently-wrong on both ends); and the windows must be whole views
/// (`split_out` refuses a corner). A violated law is a build `Err` naming it.
pub fn reshape(
    name: &str,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    // The source and the destination — the parameters `KtirFunc::reshape` mints.
    let (tensors, out) = split_out(name, r, layout, 1)?;
    let src = &tensors[0];
    let stk = crate::work::FP16_ELEMS_PER_STICK;
    let (r_in, c_in) = (src.v_rows, src.v_cols);
    let (r_out, c_out) = (out.v_rows, out.v_cols);
    // WHOLE VIEWS ON BOTH AXES, STATED AS COVERS: a re-laying copy moves a whole buffer, and the
    // program may state that whole view as ONE tile (the builder's `load_region`/`store_region`
    // form, where the first tile IS the view) or as the stick walk this door itself emits (one
    // `[1, 64]` tile per stick, where the first tile understates the view on both axes). The
    // COVERS are the form-independent statement — `r_cover == (0, v_rows)` and
    // `c_cover == (0, v_cols)` — and they are STRICTER than the first-tile equality they replace:
    // a windowed program (a chunk at `8192..12800` of a wider view) answers a cover that is not
    // the whole view, exactly as it answered a first tile that was not. A windowed reshape is a
    // different op the producer must materialize first (the same law [`transpose`] states for its
    // two sides).
    for (x, role) in [(src, "input"), (&out, "output")] {
        if x.r_cover != (0, x.v_rows) || x.c_cover != (0, x.v_cols) {
            return err(format!(
                "{name}: the {role} t{}'s windows cover rows {}..{} and columns {}..{} of its \
                 `[{}, {}]` view — a reshape re-sticks a WHOLE buffer, and the relayout builder \
                 names a tensor and its two extents with no corner and no window. Materialize the \
                 window first.",
                x.tid, x.r_cover.0, x.r_cover.1, x.c_cover.0, x.c_cover.1, x.v_rows, x.v_cols,
            ));
        }
    }
    if !c_in.is_multiple_of(stk) {
        return err(format!(
            "{name}: the input t{}'s column extent {c_in} is not a whole {stk}-element fp16 stick. \
             The copies below read whole input column sticks, so a partial one is an address no \
             rank-2 view can name. Pad the producer's width to a stick multiple.",
            src.tid,
        ));
    }
    if !c_out.is_multiple_of(stk) {
        return err(format!(
            "{name}: the output t{}'s column extent {c_out} is not a whole {stk}-element fp16 \
             stick. The copies below write whole output column sticks, so a partial one is an \
             address no rank-2 view can name. Pad the destination's width to a stick multiple.",
            out.tid,
        ));
    }
    // THE COUNT LAW: a reshape preserves the element count (the SubOp's host oracle asserts the
    // same). Stated here because these descriptors move exactly that many — a mismatch would be a
    // build error in the producer's graph, not something this door can repair.
    if u64::from(r_in) * u64::from(c_in) != u64::from(r_out) * u64::from(c_out) {
        return err(format!(
            "{name}: re-sticking t{} `[{r_in}, {c_in}]` into t{} `[{r_out}, {c_out}]` does not \
             preserve the element count — a reshape moves the same elements to new coordinates, so \
             these extents do not describe one.",
            src.tid, out.tid,
        ));
    }
    let total = u64::from(r_out) * u64::from(c_out);
    // The store windows must tile the output (`node_rows` is that guard, and it returns the row
    // count the walk below uses). The input's coverage is the whole-view loop above — it states
    // `r_cover`/`c_cover` on BOTH sides once, which is what replaced this arm's own first-tile
    // equality AND the input row-cover check that stood here separately.
    node_rows(name, &out)?;
    // ⭐ ONE `identity` COPY PER STICK — the `lmlast` mechanism over a two-sided coordinate
    // walk. Each op is `[1, stk]` on the card (the one representable row offset, per `lmlast`'s
    // own addressing note) and the addresses below are corners of each side's OWN view, so the
    // nest law carries the stick-block arithmetic rather than a hand-derived multiplier.
    //
    // A STICK NEVER STRADDLES A ROW ON EITHER SIDE: `base`, `c_in` and `c_out` are all
    // multiples of `stk`, so `base % c_in` (and `base % c_out`) is a multiple of `stk` at most
    // `c − stk` — the 64-element run lies inside ONE row's column stick on both the input and
    // the output, one copy covers it, and no partial-stick address is ever formed.
    // ⭐ TRUE EXTENTS on both sides, not the `[1, stk]`/`rbo` staging annotations: the corners below
    // are `rc_of` corners of each side's OWN `[r, c]` view, so the handles declare those views —
    // the mint's bound. The stick walk's last window ends exactly at `total` elements on each side.
    let src_name = rb(
        &src.name(),
        RowCount::of_token_rows(r_in),
        BlockCols::of_feature_cols(c_in),
    );
    let dst = rb(
        &out.name(),
        RowCount::of_token_rows(r_out),
        BlockCols::of_feature_cols(c_out),
    );
    let n_sticks = total / u64::from(stk);
    let mut ops = Vec::with_capacity(n_sticks as usize);
    for q in 0..n_sticks {
        let base = q * u64::from(stk);
        ops.push(assemble_pointwise_broadcast_off(
            PointwiseFacts {
                op_name: &format!("reshape{q}_o{}", out.tid),
                op_func: "identity",
                rows: crate::sdsc_abstract::RowCount::of_token_rows(1),
                cols: crate::sdsc_abstract::BlockCols::of_one_stick(
                    crate::sdsc_abstract::Lanes::FP16,
                ),
                inputs: &[In::sliced(
                    &src_name,
                    crate::addr::rc_of(
                        r_in,
                        c_in,
                        (base / u64::from(c_in)) as u32,
                        (base % u64::from(c_in)) as u32,
                        Df::Fp16,
                    ),
                )
                .ew()],
                o: &dst,
                out_offset: crate::addr::rc_of(
                    r_out,
                    c_out,
                    (base / u64::from(c_out)) as u32,
                    (base % u64::from(c_out)) as u32,
                    Df::Fp16,
                )
                .into(),
            },
            sym_id_base,
            layout,
        ));
    }
    Ok(ops)
}

/// [`scalarmul`] with the multiplier SUPPLIED rather than read off the whole function.
///
/// ⭐ WHY THE SPLIT EXISTS. [`program_scalarmul_scale`] reads EVERY `arith.mulf` in the function and
/// requires them all to agree — exact for a function that IS one scalarmul node, which is what
/// the producer emits. A whole-kernel function is a different shape: one decoder layer holds 21
/// `arith.mulf`, most with no splat at all and the splatted ones carrying FOUR different constants
/// (`INV_D`, `QK_SCALE`, and `RM` twice). The whole-function reader necessarily returns `None` there,
/// so a per-op caller must state the scale it PROVED for THAT op. The proving stays with the caller
/// ([`super::whole_function::splat_scale_of`]), which is where the op is; this body's job is the
/// descriptor.
///
/// ⭐ AND THIS IS THE ONLY BODY. [`scalarmul`] DELEGATES here, so the two doors cannot drift about
/// what a scalarmul emits — only about where its number came from. It used to be a COPY with that
/// same sentence on it, and the copy drifted in the one arm nothing could reach; see the operand-order
/// note below for what that cost.
pub fn scalarmul_at(
    name: &str,
    scale: f32,
    // The gather this node's tile is read through, when it is read through one — [`gather_of`]'s
    // answer, so the whole-function door and the per-`Program` door cannot disagree about whether a
    // node gathers.
    gather: Option<Gather>,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let idx = scale_slot(layout, scale).ok_or_else(|| Error {
        message: format!(
            "ScalarMul {name}: multiplier {scale}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor multiplies by a bound `[1,1]` const, so \
             the value the program uses must have a registry slot (registry desync)"
        ),
    })?;
    scalarmul_scaled(
        name,
        &ScaleSource::Registry {
            slot: scalarmul_scale_tid(idx),
        },
        // A multiply is commutative, so the stated order cannot change what it computes; the
        // body keeps the card-proven layout either way.
        ScalarOp::Multiply,
        gather,
        r,
        sym_id_base,
        layout,
    )
}

/// ⭐⭐⭐ THE SUBTRACTION'S OWN SCALAR FORM — `tile − C` / `C − tile` with `C` a constant the
/// program states as a splat, the non-commutative member of the constant-operand pointwise
/// class, the same tiebreak [`scalarmul_at`] states for multiply and [`scalarextremum_at`] for
/// the extrema.
///
/// The 1:1 map lowers a two-tensor `arith.subf` as `Elementwise(Sub)`; what that arm cannot
/// take is a splat operand (no `Region`, the same refusal every splat meets). But a subtract
/// against a STATED CONSTANT is not a two-tensor op — it is the subtraction's scalar form, and
/// the online softmax's running-max rescale is exactly that shape: trip 0's `m − m_new` reads
/// the `tl.full` initial (−65504, the f16 floor) on one side of a subtract.
///
/// ⛔ AND THE ORDER IS THE FUNCTION. `C − tile` and `tile − C` differ by more than sign, so
/// the program's stated order (`const_first`, read by [`super::whole_function::splat_scale_of`])
/// travels INTO the emission: the ungathered body lays its operands out as stated, and the
/// gathered body refuses the one shape no order can serve (see [`scalarmul_scaled`]'s
/// gathered-arm law). The two sides of a subtraction are distinguishable all the way into the
/// descriptor.
///
/// The constant is resolved through the SAME value lookup ([`scale_slot`]: exact bits, then the
/// f16 image), so the registry's completion pass — the program-stated half, which collects
/// `arith.subf` exactly like the mulf and the extrema — must carry it, and a program whose
/// constant the registry lacks is refused as the desync it is.
pub fn scalarsub_at(
    name: &str,
    value: f32,
    const_first: bool,
    // The gather this node's tile is read through, when it is read through one — [`gather_of`]'s
    // answer, so the whole-function door and the per-`Program` door cannot disagree about whether a
    // node gathers.
    gather: Option<Gather>,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let idx = scale_slot(layout, value).ok_or_else(|| Error {
        message: format!(
            "Subtraction {name}: constant {value}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor reads a bound `[1,1]` const, so the \
             value the program uses must have a registry slot (registry desync)"
        ),
    })?;
    scalarmul_scaled(
        name,
        &ScaleSource::Registry {
            slot: scalarmul_scale_tid(idx),
        },
        ScalarOp::Subtract { const_first },
        gather,
        r,
        sym_id_base,
        layout,
    )
}

/// ⭐⭐⭐ THE EXTREMUM'S OWN SCALAR FORM — `maximum(tile, C)` / `minimum(tile, C)` with `C` a
/// constant the program states as a splat, the max/min analogue of [`scalarmul_at`].
///
/// The 1:1 map already lowers a two-tensor `arith.maxnumf`/`arith.minnumf` as
/// `Elementwise(Maximum/Minimum)` — the MoE kernel's fp8 clamps ride it, with both bounds as f16
/// TILES. What that arm cannot take is a splat operand: a splat has no `Region`, so the walk's
/// input collection refuses it by name rather than choosing a node kind on the producer's behalf.
/// But a clamp against a STATED CONSTANT is not a two-tensor op at all — it is the extremum's
/// scalar form, exactly the tiebreak `Program`'s own doc states for `Elementwise(Mul)` vs
/// `ScalarMul` ("`program_scalarmul_scale` tells them apart by whether an operand is a
/// `tensor.splat`"), applied to the two ops whose identity lives at the range's edge: the online
/// softmax's running-max initial (−65504, the f16 floor, stated by `tl.full`), any clamp against
/// a stated bound.
///
/// ⭐ ONE BODY WITH [`scalarmul_at`], NOT A COPY: the descriptor is the same shape (one tile
/// input, one `In::scalar` over the bound `[1,1]` const), so this delegates to
/// [`scalarmul_scaled`] with the op's own `opFuncName` — the drift class [`scalarmul_at`]'s
/// header records is closed by sharing, and `maximum`/`minimum` are commutative, so the gathered
/// arm's operand swap keeps its argument there.
///
/// The bound is resolved through the SAME value lookup ([`scale_slot`]: exact bits, then the f16
/// image), so the registry's completion pass — the program-stated half — must carry it, and a
/// program whose extremum constant the registry lacks is refused as the desync it is.
pub fn scalarextremum_at(
    name: &str,
    value: f32,
    max: bool,
    // The gather this node's tile is read through, when it is read through one — [`gather_of`]'s
    // answer, so the whole-function door and the per-`Program` door cannot disagree about whether a
    // node gathers.
    gather: Option<Gather>,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let idx = scale_slot(layout, value).ok_or_else(|| Error {
        message: format!(
            "{} {name}: bound {value}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor reads a bound `[1,1]` const, so the \
             value the program uses must have a registry slot (registry desync)",
            if max { "Maximum" } else { "Minimum" }
        ),
    })?;
    scalarmul_scaled(
        name,
        &ScaleSource::Registry {
            slot: scalarmul_scale_tid(idx),
        },
        // Both extrema are commutative, so the stated order cannot change what they compute.
        if max {
            ScalarOp::Maximum
        } else {
            ScalarOp::Minimum
        },
        gather,
        r,
        sym_id_base,
        layout,
    )
}

/// ⭐⭐⭐ THE FILL — a `ktdp.store` of a splatted constant, the store-side member of the
/// constant-operand pointwise class [`scalarmul_at`] states for multiply, [`scalarsub_at`] for
/// subtract and [`scalarextremum_at`] for the extrema. Those three read a TILE and combine it
/// with a bound `[1,1]` const; the fill has no tile — the constant IS the whole payload — so it
/// is lowered as the IDENTITY broadcast of the value's registry slot over the store's own
/// window: one pointwise, one `In::scalar` input, the same descriptor shape the scale ops give
/// their riding const with the tile dropped. The online-softmax carry's seed (max at the finite
/// −65504 sentinel, sum and accumulator at zero, three stores in one program) is the producing
/// shape; the statement is general — any program storing a splatted compile-time float fills
/// its window with it.
///
/// The constant is resolved through the SAME value lookup ([`scale_slot`]: exact bits, then the
/// f16 image), so the registry's completion pass — the program-stated half, which collects
/// these stores beside the mulf/subf/extrema splats — must carry it, and a program whose
/// constant the registry lacks is refused as the desync it is.
///
/// ⛔ THE WINDOW IS THE STORE'S OWN TILE, resolved by the walk before this door is reached (the
/// same `region_of_view_operand` + `windowed_region` the compute store arm takes), so the fill
/// writes exactly the columns the program states — swept at the STICK-PADDED width a pointwise
/// emission demands (`DeviceWidth::for_pointwise` of the window, floor-dropped to what the
/// placement holds by [`pointwise_width_the_output_holds`], the same pair the scalar bodies
/// derive). A carry placement that reserves only the view's own sub-stick bytes is refused
/// there by name — the pad is the fill's own law, and the placement pass reserves it.
pub fn scalarfill_at(
    name: &str,
    value: f32,
    out: &Region,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let idx = scale_slot(layout, value).ok_or_else(|| Error {
        message: format!(
            "Fill {name}: constant {value}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor broadcasts a bound `[1,1]` const, so \
             the value the program uses must have a registry slot (registry desync)"
        ),
    })?;
    let scale_tid = scalarmul_scale_tid(idx);
    let out_name = out.name();
    let rows = pointwise_rows(name, out)?;
    // The scalar bodies' own corner law: a column-blocked piece reads and writes ONE window.
    let corner = chunk_window(name, rows, out)?;
    let (cols, store_cols) = pointwise_width_the_output_holds(
        layout,
        &[&out_name],
        rows,
        out.v_cols,
        DeviceWidth::for_pointwise(out.c_len).get(),
    )?;
    let scale_h = rbo(&crate::place::act_name(scale_tid));
    let out_h = rb_df(
        &out_name,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(store_cols),
        out.df,
    );
    // `n_operands: 2` is the `_df` funnel's own count for a one-input op (the input plus the
    // output — what `assemble_pointwise_broadcast_off` states for lmlast's identity copies);
    // nothing rides in this op, so the count is derived from its shape, not hand-picked.
    let tile_op = pointwise_tile_op(rows, cols, 1, 2, out.df);
    let inputs = [In::scalar(&scale_h).ew()];
    let op_name = format!("fill_o{}", out.tid);
    Ok(vec![assemble_pointwise_broadcast_off_from_tile(
        &op_name,
        &tile_op,
        "identity",
        rows,
        cols,
        &inputs,
        &out_h,
        corner,
        sym_id_base,
        layout,
    )])
}

/// ⭐⭐⭐⭐⭐ THE WRITE-BACK COPY — a `ktdp.store` of a value this function MINTED. The store arm
/// binds a store to its producing op only when the result is the store's alone; a result other
/// compute ops still read (the online-softmax fold body's running max: stored at the trip's end,
/// read by `m − m_new` and `qk − m_new[.., .., None]` on the way there) is minted as an
/// intermediate like any other, and the STORE lowers at its own program position — the
/// indexed-store arm's own law — as this identity copy from the mint to the store's window. The
/// copy is what keeps the read ORDER honest: binding the store to the producer would emit the
/// producer writing the carry IN PLACE at its own position, and the subtraction afterwards would
/// read `m_new − m_new = 0`; minting keeps the buffer untouched until the copy, so the old value
/// is live for every consumer that states it.
///
/// Same descriptor shape as [`scalarfill_at`] and lmlast's identity copies — one input, one
/// output, `identity` — with the input a FULL window over the mint rather than an `In::scalar`
/// broadcast: the mint's own placement holds the swept width (both names go to
/// `pointwise_width_the_output_holds`, so a mint that held less than the window refuses by name
/// rather than reading past its footprint).
pub fn mintcopy_at(
    name: &str,
    src: &Region,
    out: &Region,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    if src.df != out.df {
        return Err(Error {
            message: format!(
                "Copy {name}: the minted t{} is {:?} and the store's window t{} is {:?} — a \
                 store that changes dtype on the way to memory is a new producer form this door \
                 must be re-derived for",
                src.tid, src.df, out.tid, out.df
            ),
        });
    }
    let src_name = src.name();
    let out_name = out.name();
    let rows = pointwise_rows(name, out)?;
    // The scalar bodies' own corner law: a column-blocked piece reads and writes ONE window.
    let corner = chunk_window(name, rows, out)?;
    let (cols, store_cols) = pointwise_width_the_output_holds(
        layout,
        &[&src_name, &out_name],
        rows,
        out.v_cols,
        DeviceWidth::for_pointwise(out.c_len).get(),
    )?;
    let src_h = rb_df(
        &src_name,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(store_cols),
        out.df,
    );
    let out_h = rb_df(
        &out_name,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(store_cols),
        out.df,
    );
    // `n_operands: 2` is the `_df` funnel's own count for a one-input op (the input plus the
    // output) — the count the fill and lmlast's identity copies already state.
    let tile_op = pointwise_tile_op(rows, cols, 1, 2, out.df);
    let inputs = [In::full(&src_h).ew()];
    let op_name = format!("copy_o{}", out.tid);
    Ok(vec![assemble_pointwise_broadcast_off_from_tile(
        &op_name,
        &tile_op,
        "identity",
        rows,
        cols,
        &inputs,
        &out_h,
        corner,
        sym_id_base,
        layout,
    )])
}

/// ⭐ THE RUNG-3 BODY: a scalar multiply whose multiplier is a LAUNCH BINDING, not a baked value.
///
/// Same emission as [`scalarmul_at`] — the descriptor is identical, `In::scalar` over the bound
/// `[1,1]` const — with the ONE difference that matters: the scale's operand NAME is the
/// parameter's own binding tid (`In::scalar(&rbo(&act_name(tid)))`), so the value the card
/// multiplies by is whatever the launch bound at `const:t<tid>`, read at RUN TIME.
///
/// ⛔ THE SLOT IS THE TID, NOT A REGISTRY INDEX. `scalarmul_at` looks its multiplier up in
/// `BundleLayout::scalarmul_scales` BY BITS (`scale_slot`), which is exact for a baked constant
/// and wrong for a bound one: nothing at bake time knows the launch's value, and the registry is
/// empty on this path (`scales_for_program_shape` registers only constant-backed splats). The
/// scale's ADDRESS is the parameter's placement — 2 B, `SegRole::Activation`, placed by the
/// `[1,1]`-region arm in `triton-ktir-superdsc`'s `for_regions` — and the runner's `const:t<id>`
/// bind fills it.
///
/// ⛔ AND THE `[1,1]` REGION MUST NOT COUNT AS ARITY. `split_out_excluding` drops scale regions
/// by RESERVED tid (`scale_idx_of`); a positional binding is not one, so the bound scale's tid
/// travels in the skip list explicitly — one filter, the same mechanism, keyed by the fact the
/// source owns.
pub fn scalarmul_bound(
    name: &str,
    bound: &crate::emit::whole_function::BoundScale,
    // The gather this node's tile is read through, when it is read through one — [`gather_of`]'s
    // answer, so the whole-function door and the per-`Program` door cannot disagree about whether a
    // node gathers.
    gather: Option<Gather>,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    scalarmul_scaled(
        name,
        &ScaleSource::Bound { tid: bound.tid },
        // A multiply is commutative, so the stated order cannot change what it computes.
        ScalarOp::Multiply,
        gather,
        r,
        sym_id_base,
        layout,
    )
}

/// WHERE A SCALARMUL'S MULTIPLIER COMES FROM — the one axis on which [`scalarmul_at`] (a baked
/// constant, in a registry slot) and [`scalarmul_bound`] (a launch binding, at its own parameter
/// placement) differ. Everything else about the emission is one body (issue 201 item 6): the skip
/// list, the split, the width rule, the tile op, the gathered arm, the guards.
enum ScaleSource {
    /// A baked constant: the operand name is the registry slot's scale tid, already excluded from
    /// arity by `split_out_excluding`'s reserved-tid filter — no extra skip needed.
    Registry { slot: u32 },
    /// A launch binding: the operand name is the parameter's own binding tid, which is NOT a
    /// reserved slot and so must travel in the skip list explicitly.
    Bound { tid: u32 },
}

/// ⭐ THE OP'S DDL `opFuncName` AND ITS COMMUTATIVITY, CARRIED AS THE ONE DIFFERENCE between the
/// constant-operand bodies: `"multiply"` for the scale bodies, `"maximum"`/`"minimum"` for the
/// extremum's own scalar form ([`scalarextremum_at`]), `"subtract"` for the subtraction's own
/// ([`scalarsub_at`]). The descriptor shape is the same for all four — one tile input, one
/// `In::scalar` over the bound `[1,1]` const — which is why this is one body and not copies
/// that could drift (the drift class [`scalarmul_at`]'s own doc records).
///
/// ⛔ AND THE COMMUTATIVITY IS A DERIVED FACT THE BODY READS, NOT A DOC-COMMENT ASSUMPTION. Every
/// operand-layout decision below — which input the scalar occupies ungathered, whether the
/// gathered arm may swap — is a `match` over `op` itself, whose [`ScalarOp::Subtract`] variant
/// carries the program's stated order and whose [`ScalarOp::commutative`] answers the swap
/// question. `subtract` is the one member whose two operand orders are DIFFERENT functions, so
/// it takes its layout from the stated order and refuses the one gathered shape no order can
/// serve; the other three swap freely because `commutative()` says so, not because a comment
/// promised it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ScalarOp {
    Multiply,
    Maximum,
    Minimum,
    /// `tile − C` / `C − tile` — NOT commutative: the two operand orders are different
    /// functions, so the program's stated order rides the discriminant itself
    /// (`const_first`: the splatted constant is the op's first input, `C − tile`).
    Subtract {
        const_first: bool,
    },
}

impl ScalarOp {
    fn op_func(self) -> &'static str {
        match self {
            ScalarOp::Multiply => "multiply",
            ScalarOp::Maximum => "maximum",
            ScalarOp::Minimum => "minimum",
            ScalarOp::Subtract { .. } => "subtract",
        }
    }

    /// The one fact every operand-layout decision below reads: whether exchanging the two
    /// inputs computes the same function. `true` for `multiply`/`maximum`/`minimum`; `false`
    /// for `subtract` — `C − tile` and `tile − C` differ by more than sign.
    fn commutative(self) -> bool {
        !matches!(self, ScalarOp::Subtract { .. })
    }
}

fn scalarmul_scaled(
    name: &str,
    scale: &ScaleSource,
    op: ScalarOp,
    gather: Option<Gather>,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let op_func = op.op_func();
    // The index parameter is an ADDRESSING operand, so it is not one of the op's tensor inputs — the
    // arity stated below is still 1 (the tile), which is the whole point of excluding it by tid rather
    // than by relaxing the count. See [`split_out_excluding`]. A BOUND scale joins the skip list for
    // the same reason (its `[1,1]` region is not tensor arity either); a registry slot needs no skip
    // because `split_out_excluding` already drops reserved scale tids.
    let mut skip: Vec<u32> = gather.iter().map(|g| g.index_tid).collect();
    if let ScaleSource::Bound { tid } = scale {
        skip.push(*tid);
    }
    let (ins, out) = split_out_excluding(name, r, layout, 1, &skip)?;
    // ⛔ WALL 13 — THE MIXED-df PIN, SCALAR-OP SEAT: the tile input states the output's own df
    // (the same law `elementwise` states for its operands). The BOUND `[1,1]` const rides the op
    // and is emitted at the op's own df (`pointwise_tile_op`'s), so a tile at a different df
    // would be the mixed op the DDL refuses at bake — refused here by name instead.
    if ins[0].df != out.df {
        return err(format!(
            "{name}: input t{} states df {:?} but the output t{} states {:?} — a MIXED-df \
             scalar op. The vocabulary requires the transition as an explicit \
             `arith.extf`/`arith.truncf` (lowered as a convert with each side at its own df), so \
             an uncast mixed op is a producer defect",
            ins[0].tid, ins[0].df, out.tid, out.df
        ));
    }
    let scale_tid = match scale {
        ScaleSource::Registry { slot } => *slot,
        ScaleSource::Bound { tid } => *tid,
    };
    let x = ins[0].name();
    let out_name = out.name();
    let rows = node_rows(name, &out)?;
    // ⛔⛔⛔ A COLUMN-BLOCKED PIECE READS AND WRITES ITS OWN COLUMNS — the granite logits. The front end
    // column-tiles a wide ScalarMul (`subtile_ir.rs`'s tiling set) into pieces that share one input and
    // one output tensor, and this body emitted every piece at the tensors' bases. MEASURED on the card's
    // granite suffix: `logits_scaling` over the 49155-wide lm_head output is seven pieces, all one
    // descriptor writing columns `[0, 8192)`, so logits `8192..` were never written and the card almost
    // never sampled an id above 8191. The corner is `elementwise`'s, through the same helper.
    if ins[0].c_start != out.c_start {
        return err(format!(
            "{name}: the output t{}'s window starts at column {} but its input t{} sits at {} — a \
             column-blocked piece reads and writes ONE window, so unequal corners would scale one \
             operand's columns into another's.",
            out.tid, out.c_start, ins[0].tid, ins[0].c_start,
        ));
    }
    // A [`crate::addr::ColWindow`], `elementwise`'s corner: a column-blocked piece is presented stick-major
    // (#622).
    let corner = chunk_window(name, rows, &out)?;
    // DEVICE width (the padding invariant): a ScalarMul on the padded logits `[.,49159]` must use the SAME
    // device width its producer matmul emitted (49664), not the logical 49159 (whose sub-stick 7 the dxp
    // scheduler rejects). `for_pointwise` == the producer's `for_output` for macs≥2^20 producers. A no-op
    // for 64-aligned tensors (residual/embedding [.,4096]).
    //
    // ⛔ AND THE PAD IS ONLY REAL IF SOMETHING RESERVED IT — the width pair: `cols` is what the op
    // sweeps (the floored width, floor-dropped to what the placement holds), `store_cols` is what
    // the placement holds per row (the whole-view storage the handles declare). A placement that
    // cannot hold the program's own view width is an `Err` here — never a narrower descriptor.
    // See [`pointwise_width_the_output_holds`].
    let (cols, store_cols) = pointwise_width_the_output_holds(
        layout,
        &[&out_name, &x],
        rows,
        out.v_cols,
        DeviceWidth::for_pointwise(out.c_len).get(),
    )?;
    let scale_name = crate::place::act_name(scale_tid);
    let op_name = format!("scalarmul_o{}", out.tid);
    // ⭐ TRUE SHAPES, not `rbo` lies: the tile and output are declared at the `[rows, store_cols]`
    // the placement actually holds per row (the whole-view storage — a chunked piece's corner sits
    // past a piece-width declaration), and the op sweeps `cols` within it. The `[1,1]` scale keeps
    // `rbo`: one RedStick IS its whole swept window.
    // ⭐ WALL 13 — the value handles at the op's own df (`rb` IS `rb_df(Fp16)`; an f32 scalar op
    // must not understate its window 2×). The `[1,1]` scale keeps `rbo` — the scale slot's own
    // word is the registry's business, not this handle's.
    let x_h = rb_df(
        &x,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(store_cols),
        out.df,
    );
    let out_h = rb_df(
        &out_name,
        RowCount::of_token_rows(rows),
        BlockCols::of_feature_cols(store_cols),
        out.df,
    );
    let scale_h = rbo(&scale_name);
    // `In::col_window` at corner 0 is field for field `In::full` in what it emits (corner 0 is a whole-block
    // shift, stick-major either way), so a whole-tensor ScalarMul (piece 0, and every node narrower than one
    // tiling block) emits the bytes it always did.
    //
    // ⭐ AND THE ORDER IS DERIVED FROM THE FACTS, NOT ASSUMED. A COMMUTATIVE op keeps the
    // card-proven tile-first layout — exchanging the operands computes the same function, which
    // is `commutative()`'s answer, not a comment's promise — so every multiply/extremum
    // emission is byte-identical to what it always was. The SUBTRACTION takes the program's
    // STATED order, because its two orders are different functions: the descriptor's
    // `subtract` computes input 0 − input 1 (the old attention path's `corrsub`:
    // `[run_m, newm]` → `run_m − newm`, card-proven), so `C − tile` puts the scalar first and
    // `tile − C` the tile first, exactly as stated.
    let (tile_in, scalar_in) = (In::col_window(&x_h, corner).ew(), In::scalar(&scale_h).ew());
    let inputs = if op.commutative() || matches!(op, ScalarOp::Subtract { const_first: false }) {
        [tile_in, scalar_in]
    } else {
        [scalar_in, tile_in]
    };
    // ⭐ THE SAME `TileOp` `node_to_tile_ops`' ScalarMul arm declares: `out` at the DEVICE width
    // computed above, `n_operands: 2` (the scalar rides in the op, not as a tiled operand).
    let mut tile_op = pointwise_tile_op(rows, cols, 1, 2, out.df);
    tile_op.kind = TileOpKind::PointwiseOrReduce { n_operands: 2 };
    // ⭐⭐⭐ THE GATHERED FORM IS THE SAME OP WITH AN INDEX OPERAND ON ITS FIRST INPUT — not a different
    // `opFuncName`. The shipped `dxp/test/test_gather_1core/sdsc_1.json` is an ordinary `identity`
    // carrying one extra `labeledDs_` and one extra `computeOp_` field, so nothing about the multiply
    // changes; `assemble_pointwise_broadcast_gather` adds the index and forces the single-core plan the
    // gather's base-address rule requires.
    //
    // ⛔ AND THE PROGRAM'S OWN ENTRY COUNT IS CHECKED AGAINST THE DESCRIPTOR'S ROW COUNT, because the
    // two are the same fact stated twice: one index per gathered row. The index vector is described
    // rank-1 over the op's `mb`, so a program whose indirect tile takes a different number of entries
    // than the node has rows would be described with an index vector of the WRONG LENGTH — and the
    // length is what the idx→address program iterates.
    if let Some(g) = gather {
        // A gathered operand is addressed through its index from the table's BASE, so a column window
        // has no place on it.
        if out.c_start != 0 {
            return err(format!(
                "{name}: a gathered ScalarMul at column corner {} of its output t{} — the gather \
                 addresses its table from the base, so a column-blocked piece cannot be gathered. \
                 Give the gathered node a whole-tensor region.",
                out.c_start, out.tid,
            ));
        }
        if g.source.tid() != ins[0].tid {
            return err(format!(
                "{name}: the program gathers t{} but this node's tile operand is t{} — the index \
                 operand must sit immediately after the tensor it indexes, so a gather of a tensor this \
                 op does not read has no position in the descriptor.",
                g.source.tid(),
                ins[0].tid,
            ));
        }
        // ⛔⛔⛔ THE GATHER'S ENTRIES MUST **TILE** THE NODE, AND THE INDEX BUFFER MUST HOLD ONE INDEX
        // PER NODE ROW. Two separate facts, and neither is the other.
        //
        // The program states ONE work item: `embedding.py` reads `start_m = tl.program_id(0)` and takes a
        // `[BLOCK_M, D_MODEL]` tile, so its indirect access tile says BLOCK_M entries. [`node_rows`]
        // meanwhile reports the OUTPUT VIEW's full row extent, because that is what the emitted
        // descriptor spans — the same fold every other body here performs (a row-blocked program states
        // `ceil(m/blk)` windows and one descriptor computes all of them). So the two numbers differ by
        // exactly the grid, and requiring them EQUAL would refuse the fold rather than check it.
        //
        // What has to hold instead is that the work items TILE the node (`rows % entries == 0`), exactly
        // as `node_rows` requires of the store windows — and, because the index operand is described
        // rank-1 over the op's `mb`, that the INDEX BUFFER really holds `rows` indices. That second one
        // is the load-bearing check: the descriptor makes the idx→address program iterate `mb`
        // addresses, so an id buffer shorter than the node's rows would convert past its own end and
        // gather from whatever follows it. It is read off the index parameter's OWN view, which is the
        // only statement of that buffer's length anywhere in the program.
        if g.entries == 0 || !rows.is_multiple_of(g.entries) {
            return err(format!(
                "{name}: the indirect access tile takes {} entries and the node writes {rows} row(s), \
                 which {} does not divide. The emitted descriptor spans the whole node, so the work \
                 items have to TILE it — the same obligation `node_rows` puts on the store windows.",
                g.entries, g.entries,
            ));
        }
        let idx_r = r.iter().find(|x| x.tid == g.index_tid).ok_or_else(|| Error {
            message: format!(
                "{name}: the program gathers through t{}, which is not one of this node's parameters — \
                 an index buffer with no binding has no placement and no stated length",
                g.index_tid
            ),
        })?;
        let idx_len = (idx_r.v_rows as u64) * (idx_r.v_cols as u64);
        if idx_len != rows as u64 {
            return err(format!(
                "{name}: the index buffer t{} states a `[{}, {}]` view — {idx_len} index(es) — while \
                 this descriptor gathers {rows} row(s). The index operand is described rank-1 over the \
                 op's `mb`, so dbo's idx→address program converts exactly {rows} entries: a shorter \
                 buffer is read past its end and the surplus rows gather from whatever is placed next. \
                 Emit one node per work item, or bind an index buffer covering the node.",
                g.index_tid, idx_r.v_rows, idx_r.v_cols,
            ));
        }
        // ⛔⛔⛔ THE TABLE GOES **LAST** — THE GATHERED LAYOUT IS FORCED, AND ITS CORRECTNESS IS
        // DERIVED FROM THE ORDER FACT.
        //
        // Two placement laws have to hold at once and they coincide at exactly one position. The index
        // is inserted immediately BEFORE THE OUTPUT (`attach_gather_index`, which is the vendor's
        // `[input, index, output]` order and what keeps the output last however many operands are
        // added); dbo separately needs the index to sit NEXT TO the tensor it indexes, or the gathered
        // operand's use precedes the idx→address program's definition and
        // `DSC2ToDataflowIR.cpp:51` reports `operand #1 does not dominate this use`. Both hold only
        // when the gathered tensor is the LAST input.
        //
        // The ungathered commutative order above is `[table, scalar]`, which would emit
        // `[table, scalar, idx, out]` — index at 2, table at 0, non-adjacent. Swapping to
        // `[scalar, table]` emits `[scalar, table, idx, out]`: adjacent AND output-last. That
        // swap is what the forced layout IS, and whether it computes this op's function is a
        // DERIVED predicate, never a doc-comment promise: a commutative op computes the same
        // function with its operands exchanged (`commutative()`'s own answer); a `subtract`
        // computes `C − tile` in this layout, which is the program's subtraction exactly when
        // the program stated the constant FIRST. The one shape left over — `tile − C` gathered
        // — satisfies no order at all (the placement laws force the table last while the stated
        // subtraction needs it first) and is refused below, by name, rather than emitted as the
        // silently different `C − tile`.
        //
        // ⛔⛔⛔ AND THIS ARM HELD `inputs` / `gathered_input: 0` UNTIL THE WHOLE-FUNCTION DOOR REACHED
        // IT. `scalarmul_at`'s header said "everything below is unchanged and shared, so the two doors
        // cannot drift about what a scalarmul EMITS" — but the body was COPIED, not shared, and the
        // swap above was added to ONE copy. The other was unreachable (`region_for_operand` refused an
        // indirect access tile before any op was built), so nothing ran it, and the first program that
        // did got `assemble_pointwise_broadcast_gather`'s own refusal: "input 0 of 2 is named as the
        // gathered operand ... those are the same slot only for the LAST input". MEASURED, the moment
        // that refusal was lifted: `KTIR_WHOLE=1 bake_py embedding_granite`. `scalarmul` now DELEGATES
        // here, so there is one body and the drift class is gone rather than repaired.
        if let ScalarOp::Subtract { const_first: false } = op {
            return err(format!(
                "{name}: a gathered `subtract` whose constant is the SECOND operand (`tile − C`) — \
                 the gather's placement laws force the table LAST with the index beside it \
                 (`[scalar, table, idx, out]`, the dbo dominance law), and that layout computes \
                 `C − tile`. No operand order satisfies both the index-adjacency law and the stated \
                 subtraction, so this is a refusal rather than a descriptor computing a different \
                 function than the program. State the constant first, or lower this op ungathered."
            ));
        }
        let gathered_inputs = [In::scalar(&scale_h).ew(), In::full(&x_h).ew()];
        // ⭐ THE RUN, BUILT WHERE THE NODE'S ROWS ARE KNOWN — the program's stick-checked anchor
        // plus the whole sticks this node's entries span ([`StickRun::covering`]). The constructor
        // re-states the anchor check `gathers_of` already made, which is the point: the run is
        // refused at the FIRST door that can see a mid-stick start, and a hand-built one cannot
        // get behind it.
        let run = crate::superdsc_opspec::StickRun::covering(g.first_entry.entries(), rows)
            .map_err(|msg| Error {
                message: format!("{name}: {msg}"),
            })?;
        return crate::emit::assemble_pointwise_broadcast_gather(
            crate::emit::PointwiseGather {
                op_name: &op_name,
                tile_op: &tile_op,
                op_func,
                rows,
                cols,
                inputs: &gathered_inputs,
                gathered_input: 1,
                index_name: &crate::place::act_name(g.index_tid),
                run,
                o: &out_h,
            },
            sym_id_base,
            layout,
        )
        // ⭐ MANY OPS, ONE PER INDEX STICK. The index reaches the IBR as ONE stick transfer, so a node
        // with more entries than that is a leg per stick — each writing its own row window of the
        // output in place. `assemble_pointwise_broadcast_gather` owns that cut, and the node's own
        // 32-entry ceiling with it, so nothing here counts entries.
        .map_err(Error::from);
    }
    Ok(vec![assemble_pointwise_broadcast_off_from_tile(
        &op_name,
        &tile_op,
        op_func,
        rows,
        cols,
        &inputs,
        &out_h,
        corner,
        sym_id_base,
        layout,
    )])
}

/// ⭐⭐⭐⭐⭐ WALL 12's LAW — which axis block a `linalg.reduce`'s own `Dimensions` state, against
/// the rank they are stated in. PURE over the op's own facts (`Dimensions` verbatim, the result
/// `Shape`'s rank plus the reduced count); nothing here reads a frame, a placement or a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReduceAxisBlock {
    /// The trailing block `[rank−n .. rank)` — the row-reduce form this door has always lowered:
    /// the walk's 2D frame merges every leading dim into rows, so a trailing reduce IS a row
    /// reduce in that frame, which is why the door could assume last-axis before this law
    /// existed. The TIE case — a block that is BOTH leading and trailing, i.e. the whole rank
    /// reduced to a scalar — is deliberately Trailing: that shape lowers through the row-reduce
    /// path today (a rank-1 `[N]→[]` reduce is a one-row reduce over `N` columns in the frame),
    /// and this arm must not move its bytes.
    Trailing,
    /// The leading prefix `[0 .. n)` that is not also the whole rank — the DDL's NONSTICK form
    /// (`summeanmaxexx2.ddl`'s `sumnonstick`/`maxnonstick`/`meannonstick`/`minnonstick`/
    /// `absmaxnonstick` binds, `:39-56`). A leading prefix is exactly what makes row-major flat
    /// read `reduced · survivor_cols + survivor` — survivors stride 1, the reduced axis slow —
    /// the nonstick data law (`%slice_layout_nonstick`/`%stick_layout_nonstick` key the lane
    /// roles on `%outer_dim#0`, a SURVIVOR) with ZERO data movement.
    LeadingPrefix,
    /// Neither — a middle block, or a malformed statement (no dims, or more dims than the rank).
    /// A middle block's survivors are strided on both sides (`[A,B,C]` reducing `B` leaves `A`
    /// at stride `B·C` and `C` at stride 1), so no dim decomposition states them stride-1 and
    /// the nonstick form cannot express it; the row-reduce body below owns the refusal, exactly
    /// as it did before this law existed (byte-identical behavior for every shape it lowers).
    Middle,
}

/// The block a reduce's `Dimensions` state. See [`ReduceAxisBlock`] for the law; the door alone
/// reads this, and the Kani proof below pins it to the stated block facts exactly.
pub(crate) fn reduce_axis_block(dims: &[i64], in_rank: u32) -> ReduceAxisBlock {
    let n = dims.len();
    // Malformed statements are Middle: no dims reduces nothing, and more dims than the rank is
    // not a tensor this door can read. Both keep the row-reduce body's own guards as the
    // refusal — the no-drift law. (The walk's own reader refuses the empty and oversized
    // statements BY NAME before the door ever sees them; this branch is the pure law's
    // defensive floor, Kani-pinned, never textually misnamed.)
    if n == 0 || n as u32 > in_rank {
        return ReduceAxisBlock::Middle;
    }
    // The set facts, stated WITHOUT sorting: n values that each lie inside a range of exactly n
    // integers and are pairwise distinct are precisely that range's permutation (pigeonhole) —
    // the same fact a `sort + compare` states, in plain bounded loops a solver unwinds in
    // seconds. (The sort-based form this replaced is the ENCODING the user killed on
    // 2026-10-22 — "10 minute kani proofs are of no value": `to_vec` + `sort_unstable` unwind
    // introsort's swap loops over a SYMBOLIC length and CBMC sat in "Checking harness..." with
    // no verdict for 10+ minutes. The theorem never changed; only the encoding did.)
    let mut in_prefix = true;
    let mut in_trailing = true;
    let mut distinct = true;
    for i in 0..n {
        if dims[i] < 0 || dims[i] >= n as i64 {
            in_prefix = false;
        }
        if dims[i] < (in_rank - n as u32) as i64 || dims[i] >= in_rank as i64 {
            in_trailing = false;
        }
        for j in (i + 1)..n {
            if dims[i] == dims[j] {
                distinct = false;
            }
        }
    }
    let prefix = in_prefix && distinct;
    let trailing = in_trailing && distinct;
    if prefix && !trailing {
        ReduceAxisBlock::LeadingPrefix
    } else if trailing {
        ReduceAxisBlock::Trailing
    } else {
        ReduceAxisBlock::Middle
    }
}

/// ⭐⭐⭐⭐⭐ WALL 12's GEOMETRY — what a leading-prefix reduce states, from the op's OWN numels
/// (the data's and the output's, both frame-invariant: a region's `r_len · c_len` is its true
/// element count whatever frame the walk threaded). PURE; the door derives everything below
/// from these two numbers and never from the threaded `[rows, 1]` family frame (wall 11's law).
/// WALL 13 added the third input: the DATA's OWN `df` — a stick is 128 bytes, so the survivor
/// count is measured against the data's own lanes-per-stick (`Fp16` 64, `Fp32` 32), never a
/// hardwired width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NonstickGeometry {
    /// The arm fires: `(reduced_rows, survivor_cols)` with `reduced_rows ≥ 2`,
    /// `survivor_cols` a whole stick AT THE DATA'S OWN `df`, and
    /// `reduced_rows · survivor_cols == in_numel` EXACTLY — the byte law the descriptor's
    /// window and the DDL's lane roles both state.
    Lowers(u32, u32),
    /// The survivors are SUB-STICK at the data's own `df` — fewer lanes than one stick. The
    /// physical stick would have to span `[a_lo × survivor]` (TWO dims), which no `ItDim`
    /// table states (one `is_stick` dim), and SAMV-padding the survivor dim up to a stick
    /// RE-LAYOUTS the bytes (`a·stick+s ≠ a·survivor+s`) — movement, which the kernel's
    /// contract forbids. ⭐ WALL 13's RESOLUTION: the online-softmax `m`/`l` merges
    /// (`[J,32]→[32]` at B=1) classified here only while the basis was the hardcoded fp16
    /// stick — at the data's own `Fp32` basis, 32 survivors are EXACTLY one whole stick and
    /// the arm fires (the merge input `m` is f32 in the kernel, `summeanmaxexx2_fp32.ddl`
    /// binds the full nonstick family at IEEE_FP32). The DDL's own governing lines are the
    /// subchunk datastage cap over the REDUCE dims (`summeanmaxexx2.ddl:94`, max=64) and the
    /// above/below-interleave constraint over the OUTER dims (`:95`, ∈ {1,2,4}).
    SubStick(u32),
    /// The reduced extent is 1 — the op removes nothing (`in_numel == out_numel`). Today's
    /// row-reduce path owns that shape wherever its guards pass it; the no-drift law keeps it
    /// there rather than re-routing it through a one-row nonstick descriptor.
    RemovesNothing,
    /// `out_numel` does not divide `in_numel` (or an extent overflows `u32`) — not a reduce any
    /// decomposition states. Refused by name.
    Indivisible,
}

/// The geometry a leading-prefix reduce states, or why the nonstick arm does not fire. See
/// [`NonstickGeometry`] for the law; the Kani proof below pins the firing conditions and the
/// product law exactly — df-parameterized, so the basis is the data's own, not a knob.
pub(crate) fn nonstick_geometry(in_numel: u64, out_numel: u64, df: Df) -> NonstickGeometry {
    if out_numel == 0
        || !in_numel.is_multiple_of(out_numel)
        || in_numel > u32::MAX as u64
        || out_numel > u32::MAX as u64
    {
        return NonstickGeometry::Indivisible;
    }
    let survivor_cols = out_numel;
    let reduced_rows = in_numel / out_numel;
    if reduced_rows == 1 {
        return NonstickGeometry::RemovesNothing;
    }
    // The stick basis is TYPED — the DATA's OWN `Df::elems_per_stick`, the same basis
    // `stick_basis` derives for the tiler (superdsc_opspec.rs:1259) — never a hardwired width:
    // an f32 tensor's 32 survivors are ONE whole stick of its own format, sub-stick only
    // against a foreign 64-lane basis (wall 13's law; f16 data keeps the fp16 basis and its
    // classification is byte-identical to the pre-wall-13 hardcode).
    if !survivor_cols.is_multiple_of(df.elems_per_stick() as u64) {
        return NonstickGeometry::SubStick(survivor_cols as u32);
    }
    NonstickGeometry::Lowers(reduced_rows as u32, survivor_cols as u32)
}

/// A BARE ROW REDUCTION — `[rows, cols]` → `[rows, 1]`, one `sfp` op along the stick axis.
///
/// ⭐ THE ASSEMBLER WAS ALREADY WRITTEN; WHAT WAS MISSING WAS A DOOR — the same story as
/// [`transpose`]. `assemble_reduce` / `assemble_reduce_seeded` have carried the `sum`/`max`/`mean`
/// reduce since the SubtileIR path, and `assemble_rmsnorm` calls them, but only from INSIDE a fused
/// kind. A producer whose program states a bare `linalg.reduce` — every Triton `tl.sum(x, 1)` or
/// `tl.max(x, 1)` that is not part of a recognised rmsnorm — had no entry point at all.
///
/// # ⭐ THE WIDE MULTI-ROW MAX LOWERS — THE REFUSAL THAT USED TO LIVE HERE MISREAD ITS ONE DIAGNOSTIC
///
/// This door used to REFUSE `Max` at `rows > 1 && cols > 64` by name, on the reading that the
/// on-card reduce-MAX "mis-combines the per-stick partial maxima and hands back the seed" inside
/// exactly that cell. The template's own DDL refutes both halves of that sentence:
///
/// * `summeanmaxexx2.ddl` seeds the identity ITSELF — `negInf`/`posInf`/`zero`/`one`
///   `ddl.define_constant`s (`:69-72`) that the PE accumulator LRFs are initialized from (negInf
///   for max at `:209`/`:235`, zero for sum), with the accumulator's HBM content read only on
///   second-and-later reduce-dim chunk trips (`:159-196`, `:583-585`). The seed is not the
///   accumulator's launch content, and a seed-win would read −65504, not 0.
/// * The one diagnostic that ever measured the defect — `mxp = 0` over `[nqh, cap]`, 2026-07-28 —
///   is the ZEROS-READ signature: `max(−65504, 0, x) = 0` for non-positive `x`, i.e. the reduced
///   width walked its stick PADDING and the padding read a fresh buffer's zeros. And it predates
///   the row-mixing fixes — `wire.rs`'s `group_stride` law records the same-era signature for a
///   SUM ("summed a single stick across 32 consecutive rows (+ zero padding) instead of one
///   row's 32 sticks → Σx²≈0 → amax≈0"), and that fix landed 2026-09-16 (`d878aceae`), with the
///   `head_major` fix in the same window. The measured cell carried OUR addressing defect and
///   OUR unmasked padding together; it could not have attributed itself to the reduce.
///
/// Both causes are closed in the emission: the addressing by the `group_stride`/`head_major`
/// fixes — sum at `[rows>1, cols>64]` has been card-proven since (the rmsnorm prefill parity
/// runs), and the template's sum and max paths differ only in compute type and internal seed —
/// and the pad lanes by the SAMV coordinate masking
/// ([`crate::superdsc_opspec::SamvCoordinateMask`], the vendor's own `_get_coordinate_mask`).
/// torch-spyre lowers the wide multi-row max as ONE descriptor in ordinary production (SDPA's
/// `torch.amax(scores, dim=-1)` over 512-wide = 8-stick tiles, pinned against CPU in their own
/// test suite), so the shape is template-supported. The door lowers it.
///
/// ⛔ THE ARBITER IS THE CARD, NOT THE COMPILE PATH: `dxp_standalone` executes no arithmetic, so
/// no gate here can see numerics. The wide multi-row max is validated by the card gate — `[4,
/// W>64]` max AND sum against the host reference, the `attn_bmax`-form and `fq_amax`-form
/// no-regression runs, and the GQA-shared paged-attention twin flipping green.
///
/// `attention_flash.py` and `swiglu_mlp.py` still pin `BLOCK_N = 64` — that is those kernels'
/// own work-division choice (and the fixtures' correctness constraint), not a door law.
/// ⭐⭐⭐⭐⭐ WALL 13's ISLAND — A DTYPE CAST AS A FIRST-CLASS CONVERT, over `[data, out]` regions.
///
/// The cast's whole purpose is to CHANGE df, so the (from, to) pair is a property of the op the
/// walk read off its own operand/result types ([`super::whole_function::cast_of`]) and is stated
/// HERE as a fact the door checks against — never re-derived, never defaulted. Only the closed
/// inventory's fp16↔fp32 pair is admitted: `dl16tofp32` (`quantization_double_pad.ddl:58`) and
/// `fp32todl16` (`:59`), the two opaques the DDL binds with input LX→SFP `vias=["pe"]`,
/// `disable_transfer_promotion`, and exactly-one-valid. Every other transition is refused BY
/// NAME — a convert that changes the wrong format is a silent byte re-interpretation, not a loud
/// failure.
///
/// ⛔⛔ THE MIXED-df PIN, CONVERT SEAT (the auditor's law): the regions' OWN dfs must equal the
/// pair the op stated — the input region at `from`, the output region at `to`. A mismatch is a
/// producer defect (a value whose buffer holds a different format than its type says) and is
/// refused by name. Together with the elementwise door's operand-df agreement this makes a
/// mixed-df TTIR op UNEMITTABLE as one op: every df transition is one of these converts, so the
/// DD2 mixed-op refusal ("Unsupported result precision conversion") is unreachable by our
/// emission.
///
/// WHOLE-TENSOR ONLY, and that is stated rather than assumed: the shared body
/// ([`super::assemble_convert`]) mints both operands at the buffer base, so a windowed or
/// cornered convert is refused here by name instead of silently reading row 0 — the same
/// contract [`base_addressed`] states for every other base-addressed body. (The column-window
/// form exists — [`super::assemble_convert_col`] — and a producer that needs it states its
/// corner through the nest the way the fp8 quant tail does.)
pub fn convert(
    name: &str,
    from: Df,
    to: Df,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let op_func = match (from, to) {
        (Df::Fp16, Df::Fp32) => "dl16tofp32",
        (Df::Fp32, Df::Fp16) => "fp32todl16",
        _ => {
            return err(format!(
                "convert {name}: the format transition {from:?} → {to:?} is outside the closed \
                 inventory — the vendor binds only the fp16↔fp32 pair (`dl16tofp32`/`fp32todl16`, \
                 quantization_double_pad.ddl:58/:59), and routing any other transition through a \
                 nearest-neighbor opfunc would silently re-interpret the value's bytes at the \
                 wrong width"
            ));
        }
    };
    let (ins, out) = split_out(name, r, layout, 1)?;
    let data = ins[0];
    // ⛔ THE MIXED-df PIN, checked on what the regions STATE — never on what the op meant.
    if data.df != from {
        return err(format!(
            "convert {name}: the input t{}'s region states df {:?} but the cast reads {:?} — the \
             value's buffer holds a different format than its own type says, which is a producer \
             defect, not a convert this door can emit. (The mixed-df law: a cast is the ONE op \
             whose sides may differ, and it emits as this convert with each side at its own df.)",
            data.tid, data.df, from
        ));
    }
    if out.df != to {
        return err(format!(
            "convert {name}: the output t{}'s region states df {:?} but the cast writes {:?} — \
             the value's buffer holds a different format than its own type says, which is a \
             producer defect, not a convert this door can emit",
            out.tid, out.df, to
        ));
    }
    // ⭐ THE SWEEP IS THE OUTPUT's OWN VIEW (the whole-tensor law): a convert is shape-preserving,
    // so the input's window must BE the output's — same rows, same columns, both corners at 0
    // (the shared body mints at the buffer base; a corner it cannot place is a named refusal,
    // never row 0 by default).
    let rows = node_rows(name, &out)?;
    if data.r_len != rows || data.c_len != out.c_len {
        return err(format!(
            "convert {name}: the input t{}'s window is `[{}, {}]` but the output t{}'s is \
             `[{rows}, {}]` — a convert is shape-preserving, so the two windows must be the same",
            data.tid, data.r_len, data.c_len, out.tid, out.c_len
        ));
    }
    if data.r_start != 0 || data.c_start != 0 || out.r_start != 0 || out.c_start != 0 {
        return err(format!(
            "convert {name}: input t{} at corner ({}, {}) and output t{} at ({}, {}) — the \
             shared convert body addresses both operands at the BUFFER BASE, so a window or \
             corner it cannot place is refused here rather than silently reading row 0. A \
             column-windowed convert states its corner through the nest the way the fp8 quant \
             tail does (`assemble_convert_col`)",
            data.tid, data.r_start, data.c_start, out.tid, out.r_start, out.c_start
        ));
    }
    let cols = out.c_len;
    if rows == 0 || cols == 0 {
        return err(format!(
            "convert {name}: t{} `[{rows}, {cols}]` — a convert needs a non-empty tile",
            out.tid
        ));
    }
    // The stick law at the WIDER format's basis is the shared body's own (`assert_df_stick_multiple`
    // over the pair's wider stick — fp16↔fp32 keeps 64), so a sub-stick width refuses there with
    // the typed message; nothing here re-states it.
    let op_name = format!("{op_func}_o{}", out.tid);
    // ⭐ EACH HANDLE AT ITS OWN df — the window mint prices bytes at the handle's word length, so
    // the fp32 side must not be annotated at 2 bytes ([`rb_df`]'s own law, the fp8 activation's).
    // The handles declare each region's OWN view (the storage the base-addressed body sweeps).
    let in_h = rb_df(
        &data.name(),
        RowCount::of_token_rows(data.v_rows),
        BlockCols::of_feature_cols(data.v_cols),
        from,
    );
    let out_h = rb_df(
        &out.name(),
        RowCount::of_token_rows(out.v_rows),
        BlockCols::of_feature_cols(out.v_cols),
        to,
    );
    Ok(vec![assemble_convert(
        &op_name,
        op_func,
        rows,
        cols,
        &in_h,
        &out_h,
        sym_id_base,
        layout,
    )])
}

pub fn reduce(
    name: &str,
    kind: crate::ktir_node::ReduceKind,
    axes: crate::ktir_node::ReduceAxes,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let (ins, out) = split_out(name, r, layout, 1)?;
    let data = ins[0];
    // The REDUCED extent is the data's column span and the result is one column per row. Read the
    // shape off the DATA rather than the output, so `rows` and `cols` cannot both come from the same
    // region and agree vacuously.
    let (rows, cols) = (data.r_len, data.c_len);
    if rows == 0 || cols == 0 {
        return err(format!(
            "{name}: reducing t{} `[{rows}, {cols}]` — a reduce needs a non-empty tile",
            data.tid
        ));
    }
    // ⭐⭐⭐⭐⭐ WALL 12 — THE LEADING-PREFIX ARM. The op's OWN `Dimensions` state which axis block
    // it reduces; a LEADING PREFIX (and not the whole rank) is the DDL's nonstick form, whose
    // survivors stride 1 in row-major flat (`reduced · survivor_cols + survivor`). The geometry
    // comes from the op's OWN numels — `rows·cols` and `out.r_len·out.c_len` are frame-invariant
    // (wall 11's law: never the threaded `[rows, 1]` family frame) — so the arm cannot mis-read
    // a respelled region. Everything that is NOT a leading prefix falls through to the row-reduce
    // body below BYTE-IDENTICALLY (the no-drift law): Middle (a middle block, or a degenerate
    // shape the row-reduce body already lowers — hard-refusing Middle outright would regress
    // shapes like `[A,1,1]` reducing dims `[1]` that lower today), Trailing (the tie included —
    // a full-rank reduce-to-scalar is deliberately the row-reduce path's), and a
    // `RemovesNothing` geometry (reduced extent 1 — the op removes nothing; today's row-reduce
    // body owns it).
    if matches!(
        reduce_axis_block(axes.dims(), axes.in_rank()),
        ReduceAxisBlock::LeadingPrefix
    ) {
        let in_numel = rows as u64 * cols as u64;
        let out_numel = out.r_len as u64 * out.c_len as u64;
        // WALL 13: the basis is the DATA's own df (Region::df's law) — never a hardwired width.
        match nonstick_geometry(in_numel, out_numel, data.df) {
            NonstickGeometry::Lowers(reduced_rows, survivor_cols) => {
                // The nonstick descriptor: survivors are the lanes, the accumulator is the
                // survivor vector. The handles carry the regions' OWN frame dims — the
                // assembler's agreement-assert re-derives the numels from them, so a frame that
                // disagrees with the geometry is a named build failure, never a mis-walk.
                let op_func = kind.nonstick_op_func();
                let op_name = format!("{op_func}_o{}", out.tid);
                return Ok(vec![
                    crate::ir::bridge::tiled_op_sdsc_op::reduce::assemble_reduce_nonstick(
                        &op_name,
                        op_func,
                        reduced_rows,
                        survivor_cols,
                        data.df,
                        // ⭐ WALL 13 — each handle at its region's OWN df: the window mint prices
                        // bytes at the handle's word length, so an f32 data tensor annotated at
                        // fp16 would UNDERSTATE its access 2× and let a past-the-end read pass
                        // the mint. `rb` IS `rb_df(Df::Fp16)`, so every f16 reduce is
                        // byte-identical.
                        &rb_df(
                            &data.name(),
                            RowCount::of_token_rows(rows),
                            BlockCols::of_feature_cols(cols),
                            data.df,
                        ),
                        &rb_df(
                            &out.name(),
                            RowCount::of_token_rows(out.r_len),
                            BlockCols::of_feature_cols(out.c_len),
                            out.df,
                        ),
                        sym_id_base,
                        layout,
                    ),
                ]);
            }
            NonstickGeometry::SubStick(survivor_cols) => {
                // The named refusal, at the DATA's OWN basis: this vocabulary states ONE
                // `is_stick` ItDim, and a physical stick spanning the survivors plus their row
                // groups would be TWO logical dims in one stick (`a·stick + s ≠ a·survivor + s`
                // re-layouts the bytes). WALL 13's resolution lives one fact earlier — if the
                // data is f32, its own stick is 32 lanes and 32 survivors are ONE WHOLE STICK
                // (`summeanmaxexx2_fp32.ddl` carries the full nonstick family at IEEE_FP32),
                // so a 32-survivor refusal here names an fp16 (or narrower-basis) tensor: the
                // fix is the data's true df in `Region::df`, not a wider vocabulary. The DDL's
                // own governing lines for any residual case: the subchunk datastage cap over
                // the REDUCE dims (`summeanmaxexx2.ddl:94`, max=64) and the above/below-
                // interleave constraint over the OUTER dims (`:95`, ∈ {1,2,4}).
                let eps = data.df.elems_per_stick() as u64;
                return err(format!(
                    "{name}: the reduce's output t{} is `[{}, {}]` — a leading-axis reduce with \
                     {survivor_cols} survivor lanes, SUB-STICK at the data's own format (a {} \
                     stick is {eps} lanes). A physical stick spanning the survivors and their \
                     row groups would be TWO logical dims in one stick, which this vocabulary's \
                     one `is_stick` dim cannot state, and padding the survivors up to a stick \
                     re-layouts the bytes (`a·{eps}+s ≠ a·{survivor_cols}+s`) — movement the \
                     op's contract forbids.",
                    out.tid,
                    out.r_len,
                    out.c_len,
                    data.df.name()
                ));
            }
            NonstickGeometry::Indivisible => {
                return err(format!(
                    "{name}: the reduce's output t{} states {out_numel} elements from t{}'s \
                     {in_numel} — a leading-axis reduce's survivor count must divide its input \
                     count exactly; this decomposition is not a reduce any dim decomposition states.",
                    out.tid, data.tid
                ));
            }
            // The op removes nothing — today's row-reduce body owns the shape (no-drift).
            NonstickGeometry::RemovesNothing => {}
        }
    }
    // ⭐ THE WIDE MULTI-ROW MAX PASSES THROUGH — the guard that used to refuse it here was built
    // on a misread diagnostic and is deleted; the corrected law (with the DDL citations and the
    // card gate that arbitrates) is this function's own doc above. The shape checks below are
    // unchanged and still refuse everything that is not a well-formed row reduction.
    // A `[rows, 1]` accumulator: the reduce writes one value per row.
    if out.c_len != 1 {
        return err(format!(
            "{name}: the reduce's output t{} is `[{}, {}]`, but a row reduction along the stick axis \
             writes ONE value per row — `[{rows}, 1]`. An output wider than one column would leave \
             every column but the first holding whatever the buffer held.",
            out.tid, out.r_len, out.c_len
        ));
    }
    if out.r_len != rows {
        return err(format!(
            "{name}: the reduce reads {rows} row(s) of t{} and writes {} row(s) of t{} — a row \
             reduction writes exactly one value per row it reads",
            data.tid, out.r_len, out.tid
        ));
    }
    let op_name = format!("{}_o{}", kind.op_func(), out.tid);
    Ok(vec![
        crate::ir::bridge::tiled_op_sdsc_op::reduce::assemble_reduce_seeded(
            &op_name,
            kind.op_func(),
            rows,
            cols,
            &rb(
                &data.name(),
                RowCount::of_token_rows(rows),
                BlockCols::of_feature_cols(cols),
            ),
            // The accumulator's own view frame — `[rows, 1]` by the two checks above (one
            // value per row the reduce reads), stated by the region the way the nonstick
            // branch's out handle states its own frame.
            &rb(
                &out.name(),
                RowCount::of_token_rows(out.r_len),
                BlockCols::of_feature_cols(out.c_len),
            ),
            sym_id_base,
            layout,
        ),
    ])
}
