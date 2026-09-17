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
    EmittedOp, In, assemble_pointwise_broadcast_off, assemble_restickify_kt_2d, bmm_site,
    check_pointwise_cols, emit_sdsc_tiled, fl, op_func_from_str, pointwise_broadcast_opspec,
    pointwise_chunk_out_offset, pw2, rb, rbo,
};
use crate::ir::bridge::tiled_op_sdsc_op::{
    assemble_attn, assemble_matmul_off, try_assemble_matmul_seeded,
};
use crate::ir::bridge::tiled_op_sdsc_op::{
    assemble_pointwise_broadcast_off_from_tile, assemble_pointwise_seeded_from_tile,
    assemble_rmsnorm,
};
use crate::ir::island::tile_op::{TileOp, TileOpKind};
use crate::ktir_node::{Elementwise, KtirNode};
use crate::place::{PlaceId, SynthRole};
use crate::placement::{BundleLayout, syn};
use crate::reserved_tids::{
    ATTN_CAUSAL_TID, ATTN_MASK_TID, ATTN_ZERO_TID, IDENTITY_TID, LAST_HIDDEN_TID, ROPE_P_TID,
    kct_resident_tid, scalarmul_scale_tid,
};
use crate::sdsc_abstract::{KernelTag, Stk};
use crate::superdsc_opspec::{DataFormat, Df, Fp16, ItDim, SdscFoldSet};
use crate::work::{CoreSplit, DeviceWidth, FP16_ELEMS_PER_STICK};
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
    /// ⭐ NOT ALWAYS THE TENSOR'S SHAPE, AND THAT IS THE POINT. `KtirFunc::rope` views `x` as
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
    /// prefill lm-head tail states its row in the KTIR — `KtirFunc::matmul` puts
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
    /// This parameter is the program's OUTPUT — a `ktdp.store` writes through its view.
    ///
    /// ⛔ NOT "THE LAST PARAMETER". Parameters are minted in FIRST-USE order, and a model constant
    /// first used after the output's view (a matmul's zero seed, read by `splat_zero` only once the
    /// contraction is built) lands after it. The store is what says which buffer is written.
    pub is_out: bool,
    /// This parameter's view declares `Fp8E4m3` elements — one byte each.
    ///
    /// ⭐ THE fp8-NESS IS A VIEW FACT, not an arity guess: `KtirFunc::matmul_fp8` builds its weight
    /// through `view_fp8`, which writes `Dtype = Fp8E4m3` on the `ktdp.construct_memory_view`.
    pub is_fp8: bool,
}

impl Region {
    /// The operand name — the spelling `ds_name` produced for the SubtileIR path.
    pub fn name(&self) -> String {
        crate::place::act_name(self.tid)
    }
}

/// ⛔⛔⛔ THE GUARD FOR THE FACT THIS FILE USED TO DROP: an operand this emitter addresses at the
/// BUFFER BASE may not carry a row corner.
///
/// `rb(name, rows, cols)` — every operand spelling below — names a tensor, not an offset into one, so
/// the only row a base-addressed descriptor can read is row 0. A KTIR program CAN state a row
/// (`KtirFunc::matmul` puts `selector_lastrow_col(mq)` on its activation access tile), and for the
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
/// `lower_scalarmul_node`) are each called ONCE per node with `node.output.region.rows.len` and emit
/// descriptors spanning all of it. The KTIR side states that same number on
/// [`KtirNode::out_shape`] — and it has to, because a producer arm may ROW-BLOCK its region so the
/// emulator's per-core LX holds the live tile set (`KtirFunc::silu_mul` always does; elementwise and
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
        // this crate nor `KtirFunc` mentions `scf.for` anywhere (`grep -rn ScfFor` is empty in both),
        // so the guard fires only for a program shape that has never reached here — a Triton front
        // end's K-blocked MLP, whose windows all live in the loop body. It changes nothing for a
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
        // arm that ROW-BLOCKS its region for the emulator's LX (`KtirFunc::silu_mul`,
        // `lower_elementwise_node_rows`, `lower_scalarmul_node`'s `by_row`) states one access tile per
        // block, and the field above keeps only block 0 — so a ported body reading its row count off
        // it emitted main's descriptors over the BLOCK HEIGHT and every later block's rows were never
        // written. MEASURED, granite-3.1-2b prefill at mq=31: `silu_o461`/`mulsilu_o461` carried
        // `mb_=16` where every other op in the layer carried `mb_=31`, so rows 16..30 of all 40
        // layers' MLP output kept whatever the buffer held. This states the span so
        // [`node_rows`] can prove the blocks tile the node before a body spans them with one
        // descriptor.
        //
        // ⛔ AND IT IS EVERY VIEW OF THE PARAMETER, NOT THE ONE FOUND ABOVE. `KtirFunc::view_of` mints
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
        let is_out = view.result.is_some_and(|v| stored.contains(&v));
        let is_fp8 = view.attributes.iter().any(|(key, v)| {
            matches!(
                (key, v),
                (
                    AttrKey::Dtype,
                    Attr::Dtype(ktir_core::dtypes::DType::Fp8E4m3)
                )
            )
        });
        out.push(Region {
            tid: tid.get(),
            v_rows: rows,
            v_cols: cols,
            r_start,
            c_start,
            r_len,
            c_len,
            r_cover,
            is_out,
            is_fp8,
        });
    }
    Ok(out)
}

/// Every view a `ktdp.store` writes through — the program's OUTPUT buffers. A store's second operand
/// is the access tile it writes, and that tile's first operand is the view.
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
fn index_constants(f: &IRFunction<'static>) -> std::collections::HashMap<Ssa, i64> {
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
    m
}

/// An op's `[.., rows, cols]` from its own `shape` attribute or result type — the same reading
/// `logical_2d` did, kept because a 1-D shape is a ROW (our per-channel vectors are `[hidden]`).
fn shape_2d(op: &Operation<'_>) -> Option<(u32, u32)> {
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
        [.., a, b] => (*a as u32, *b as u32),
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
/// `out` is the DEVICE width. Those are the three arms, and they are the three arms here.
fn pointwise_tile_op(rows: u32, cols: u32, n_operands: u32) -> TileOp {
    TileOp {
        kind: TileOpKind::PointwiseOrReduce { n_operands },
        dims: vec![
            ItDim {
                name: "mb",
                size: rows,
                is_reduction: false,
                is_stick: false,
                df: Df::Fp16,
            },
            ItDim {
                name: "out",
                size: cols.max(1),
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
        ],
        df: Df::Fp16,
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
        Elementwise::Exp => ("exp", 1),
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
fn pointwise_extents_agree(
    name: &str,
    kind: Elementwise,
    ins: &[Region],
    out: &Region,
) -> Result<(), Error> {
    for (i, x) in ins.iter().enumerate() {
        if (x.v_rows, x.c_len) != (out.v_rows, out.c_len) {
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
                out.v_rows,
                out.c_len,
                x.v_rows,
                x.c_len,
            ));
        }
    }
    Ok(())
}

/// main's `lower_elementwise_node` (main 8362-8432). Its door: the [`Elementwise`] kind is stated by
/// the caller, the operand names are the parameters, and the row count is the node's ([`node_rows`] —
/// the producer row-blocks a region wider than the LX holds, so the first window is not the node).
pub fn elementwise(
    name: &str,
    kind: Elementwise,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let (op_func, arity) = elementwise_op_func(name, kind)?;
    let (ins, out) = split_out(name, r, layout, arity)?;
    pointwise_extents_agree(name, kind, &ins, &out)?;
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
    check_pointwise_cols(cols, "Elementwise", out.tid)?;
    let in_names: Vec<String> = ins.iter().map(|x| x.name()).collect();
    let in_refs: Vec<&str> = in_names.iter().map(|s| s.as_str()).collect();
    let op_name = format!("{op_func}_o{}", out.tid);
    let tile_op = pointwise_tile_op(node_rows(name, &out)?, cols, arity as u32 + 1);
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
    check_pointwise_cols(o.c_len, "SiluMul", o.tid)?;
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
    // HANDLE-FLOW: bind gate/up/tmp/out once; `tmp` flows from the silu output INTO the mul input.
    let gate = rb(&gate, rows, cols);
    let up = rb(&up, rows, cols);
    let tmp = rb(&tmp, rows, cols);
    let out = rb(&out, rows, cols);
    Ok(vec![
        assemble_pointwise_broadcast_off(
            &silu_name,
            "silu",
            crate::sdsc_abstract::RowCount::of_token_rows(rows),
            crate::sdsc_abstract::BlockCols::of_feature_cols(cols),
            &[In::sliced(
                &gate,
                crate::addr::col_of(rows, full_cols, gate_off, Df::Fp16),
            )
            .ew()],
            &tmp,
            // The WRITE needs the same nest the READS above go through. A raw `cols_start` is the
            // flat element index; in the stick-blocked output, column `c` of a `[rows, cols]` tensor
            // starts at `(c/64)*(rows*64)`. Equal only at rows == 1, so decode was right and prefill
            // wrote every later column block on top of the first one.
            crate::addr::col_of(rows, full_cols, out_off, Df::Fp16),
            sym_id_base,
            layout,
        ),
        assemble_pointwise_broadcast_off(
            &mul_name,
            "multiply",
            crate::sdsc_abstract::RowCount::of_token_rows(rows),
            crate::sdsc_abstract::BlockCols::of_feature_cols(cols),
            &[
                In::sliced(
                    &tmp,
                    crate::addr::col_of(rows, full_cols, out_off, Df::Fp16),
                )
                .ew(),
                In::sliced(&up, crate::addr::col_of(rows, full_cols, up_off, Df::Fp16)).ew(),
            ],
            &out,
            crate::addr::col_of(rows, full_cols, out_off, Df::Fp16),
            sym_id_base,
            layout,
        ),
    ])
}

/// main's `lower_rmsnorm_node` (main 8604-8666): the 6-op sequence IBM's `torch_spyre` uses
/// (`decompositions.py:409 spyre_rms_norm`), assembled by `assemble_rmsnorm`.
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
    // x, gamma and the output — the parameters `KtirFunc::rmsnorm` mints. Its constants are
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
            "RmsNorm {name}: the program states no epsilon. `KtirFunc::rms_norm` splats it into the \
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
    check_pointwise_cols(out.c_len, "RmsNorm", out.tid)?;
    let rows = node_rows(name, &out)?;
    let cols = out.c_len;
    // rms_norm_eps (config) flows via the scalarmul registry (compute_bundle_layout collected it) —
    // the same reserved `[1,1]` const `subtile→superdsc` adds to the mean-of-squares.
    let eps_const = crate::place::act_name(scalarmul_scale_tid(eps_idx));
    let x = tensors[0].name();
    let gamma = tensors[1].name();
    let t = out.tid;
    Ok(assemble_rmsnorm(
        &format!("o{t}"),
        rows,
        cols,
        &x,
        &gamma,
        PlaceId::Act(t),
        &eps_const,
        sym_id_base,
        layout,
    ))
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
    let (ins, out) = split_out(name, r, layout, 1)?;
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
    let idx = scale_slot(layout, scale).ok_or_else(|| Error {
        message: format!(
            "ScalarMul {name}: multiplier {scale}, read off the program, is absent from \
             `BundleLayout::scalarmul_scales` — the descriptor multiplies by a bound `[1,1]` const, so \
             the value the program uses must have a registry slot (registry desync)"
        ),
    })?;
    let x = ins[0].name();
    let out_name = out.name();
    let rows = node_rows(name, &out)?;
    // DEVICE width (the padding invariant): a ScalarMul on the padded logits `[.,49159]` must use the SAME
    // device width its producer matmul emitted (49664), not the logical 49159 (whose sub-stick 7 the dxp
    // scheduler rejects). `for_pointwise` == the producer's `for_output` for macs≥2^20 producers. A no-op
    // for 64-aligned tensors (residual/embedding [.,4096]).
    let cols = DeviceWidth::for_pointwise(out.c_len).get();
    let scale_name = crate::place::act_name(scalarmul_scale_tid(idx));
    let op_name = format!("scalarmul_o{}", out.tid);
    let x_h = rbo(&x);
    let scale_h = rbo(&scale_name);
    let inputs = [In::full(&x_h).ew(), In::scalar(&scale_h).ew()];
    // ⭐ THE SAME `TileOp` `node_to_tile_ops`' ScalarMul arm declares: `out` at the DEVICE width
    // computed above, `n_operands: 2` (the scalar rides in the op, not as a tiled operand).
    let mut tile_op = pointwise_tile_op(rows, cols, 2);
    tile_op.kind = TileOpKind::PointwiseOrReduce { n_operands: 2 };
    Ok(vec![assemble_pointwise_broadcast_off_from_tile(
        &op_name,
        &tile_op,
        "multiply",
        rows,
        cols,
        &inputs,
        &rbo(&out_name),
        0,
        sym_id_base,
        layout,
    )])
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
    /// emulator does not model at all — the producer's `KtirFunc::attn`/`::rope` never receive it and
    /// emit the same ops either way.
    pub rows_are_requests: bool,
    pub sym_id_base: &'a mut i64,
    pub layout: Option<&'a BundleLayout>,
}

/// ⭐⭐⭐ THE ATTENTION NODE'S OPERANDS AND ITS RESIDENT CAPACITY, READ OUT OF THE PROGRAM THAT
/// STATES THEM — the seven fields `AttnFacts` used to carry from the producer.
///
/// Every one is a positional join on [`KtirNode::args`], which is the join [`regions`] already
/// performs: the producer's `KtirFunc::attn` mints its parameters in first-use order — `q`, `out`,
/// then (when this node emits the runtime length mask) the mask, then EVERY segment's K view followed
/// by EVERY segment's V view — and `crate::place::act_name(tid)` is the operand spelling every proven
/// builder agrees on.
///
/// ⛔ AND THE POSITIONS ARE CROSS-CHECKED AGAINST THE SHAPES THEY MUST HAVE, because a silent
/// positional slip REPOINTS A TENSOR: `q`/`out` are `[mq, nqh·hd]` and every K/V view is `[·, nkvh·hd]`,
/// so a q↔kv swap is caught; the two views of one SEGMENT share a row extent while a segment's K and
/// the NEXT segment's V do not, so an all-Ks-then-all-Vs order mistaken for an interleaved one is
/// caught. A mismatch is a build error naming what disagreed, never a descriptor built on a guess.
#[derive(Clone, Copy, Debug)]
pub struct AttnOperands {
    /// main's `inputs[0]` — the rotated Q.
    pub q_id: u32,
    /// main's `output`.
    pub out_id: u32,
    /// main's `kv.cache_tensor()` / `kv.v_cache_tensor()` — the resident prefix cache, segment 0.
    pub k_id: u32,
    pub v_id: u32,
    /// main's `inputs[3]` / `inputs[4]` — this step's new K and V, the last segment.
    pub new_k_id: u32,
    pub new_v_id: u32,
    /// The resident cache's structural capacity — the prefix K view's OWN row extent, which is
    /// `ir.tensors[kv.cache_tensor()].rows`, i.e. exactly what main's call site looked up.
    pub cap: u32,
    /// Which PARAMETER the resident K is, so the sweep check below reads the rows of that buffer
    /// without searching for it a second time.
    pub k_param: usize,
    /// The query row count, read off `q`'s own view — not a fact the caller states.
    pub mq: u32,
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
    // `q`, then `out`: `KtirFunc::attn`'s first two `arg_for` calls, in that order.
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
             program's parameters are not the order `KtirFunc::attn` mints them in",
            q.tid,
            if q.is_out { "" } else { "not " },
            out.tid,
            if out.is_out { "" } else { "not " },
        ));
    }
    // ⭐ THE HEAD DIM, FROM THE PROGRAM'S OWN PER-HEAD WINDOW. `KtirFunc::attn` reads head `h` as
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
        q_id: q.tid,
        out_id: out.tid,
        k_id: kc.tid,
        v_id: vc.tid,
        new_k_id: new_k.tid,
        new_v_id: new_v.tid,
        cap: kc.v_rows,
        k_param,
        mq,
    })
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
fn param_first_tile(f: &IRFunction<'static>, ptr: &Ssa) -> Option<(u32, u32)> {
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
/// `KtirFunc::attn` also states `-1e38` and `0.0`, and telling them apart by value would be a
/// convention rather than a reading.
/// The RMSNORM epsilon THE PROGRAM STATES, read structurally.
///
/// ⭐ RECOGNISED BY ITS SHAPE, NOT ITS VALUE. `KtirFunc::rms_norm` computes `1/sqrt(mean + eps)`, so
/// the epsilon is the splat operand of the `arith.addf` that feeds the one `math.sqrt`. The same body
/// also splats `0.0` (the reduce seed), `1.0` (the reciprocal's numerator) and the reciprocal row
/// count, so picking a float out by value would be a convention rather than a reading — the same
/// discipline [`program_score_scale`] follows.
///
/// `f32_splat` emits `arith.constant` (scalar) then `tensor.splat`, so the value is one hop behind the
/// operand.
///
/// ⭐ AND THE ROOT IS EITHER `math.sqrt` OR `math.rsqrt`, because a third-party producer states
/// `1/sqrt(x)` as ONE op. `KtirFunc::rms_norm` spells it `math.sqrt` then `arith.divf(1.0, ·)`; a
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
/// closely than `KtirFunc`'s `math.sqrt` + `arith.divf` does, and the reader was the only thing in
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
    // the whole discriminator, and it holds for any spelling of the mean.
    let mut splats = add.operands.iter().filter_map(|&s| splat_value(s));
    let eps = splats.next()?;
    if splats.next().is_some() {
        return None;
    }
    Some(eps as f32)
}

/// The SCALARMUL multiplier the program states, read structurally.
///
/// ⭐ EVERY `arith.mulf` IN THE PROGRAM, AND THEY MUST AGREE. `KtirFunc`'s scalarmul arm splats the
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
    layout.and_then(|l| {
        l.scalarmul_scales
            .iter()
            .position(|s| s.to_bits() == value.to_bits())
    })
}

/// ⭐ `pub` FOR THE SAME REASON AS [`program_rmsnorm_eps`]: the caller builds the registry this
/// value is looked up in, by bits, so it must be able to read the same value rather than restate the
/// reading.
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
    let is_matmul: std::collections::HashSet<Ssa> = f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::LinalgMatmul)
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
    if hd > Fp16::ELEMS_PER_STICK && !(hd / Fp16::ELEMS_PER_STICK).is_multiple_of(2) {
        return err(format!(
            "RoPE prefill (t{t}, mq={mq}): head_dim {hd} > stick {stk} with an ODD stick count \
             ({ns}); the rotate-half slab-swap needs half=hd/2 to be a whole number of sticks (hd a \
             multiple of 2·stick). Real head_dims (128/256/512) satisfy this.",
            stk = Fp16::ELEMS_PER_STICK,
            ns = hd / Fp16::ELEMS_PER_STICK,
        ));
    }
    let p = crate::place::act_name(ROPE_P_TID);
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
    let ew_ph = |ops: &mut Vec<EmittedOp>,
                 name: String,
                 f: &'static str,
                 a: &str,
                 aoff: crate::addr::DevOff,
                 b: &str,
                 boff: crate::addr::DevOff,
                 o: &str,
                 ooff: crate::addr::DevOff,
                 sib: &mut i64|
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
        let op = pointwise_broadcast_opspec(
            op_func_from_str(f),
            1,
            hd,
            &[
                In::sliced(&rbo(a), aoff).ew(),
                In::sliced(&rbo(b), boff).ew(),
            ],
            o,
            ooff.into_raw_elems(),
            false, // RoPE per-(row,head) [1,hd] block (rows==1): rank-3 flat
        )
        .map_err(|e| Error { message: e })?;
        ops.push(emit_sdsc_tiled(
            &name,
            &op,
            &SdscFoldSet::new(op.iter.cores_used()),
            sib,
            layout,
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
        if let Some(l) = layout {
            for r in [R::Rot, R::Xc, R::Rs] {
                l.synth(out_id.synth(r), &[rows, hd]);
            }
        }
        // rot = x[rows,hd] @ P[hd,hd] — the same P kernel for every row (m=rows GEMM, k=n=hd).
        ops.push(assemble_matmul_off(
            &format!("rope_rot_o{t}"),
            crate::sdsc_abstract::MatM::of_head_major_rows(rows),
            crate::sdsc_abstract::MatN::of_head_dim(hd),
            crate::sdsc_abstract::MatK::of_head_dim(hd),
            crate::sdsc_abstract::MatY::unbatched(),
            crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                bmm_site::TapeLoweringSite::witness(),
            ),
            &rb(&x, rows, hd),
            crate::addr::DevOff::ZERO,
            &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
            crate::addr::DevOff::ZERO,
            &rb(&rot, rows, hd),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
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
        ops.push(pw2(
            &format!("rope_xc_o{t}"),
            "multiply",
            hm_rows,
            hd_cols,
            In::full(&hh(&x)),
            cos_in,
            &hh(&xc),
            sym_id_base,
            layout,
        ));
        ops.push(pw2(
            &format!("rope_rs_o{t}"),
            "multiply",
            hm_rows,
            hd_cols,
            In::full(&hh(&rot)),
            sin_in,
            &hh(&rs),
            sym_id_base,
            layout,
        ));
        ops.push(pw2(
            &format!("rope_add_o{t}"),
            "add",
            hm_rows,
            hd_cols,
            In::full(&hh(&xc)),
            In::full(&hh(&rs)),
            &hh(&out),
            sym_id_base,
            layout,
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
        let (hd_u, stk_u) = (hd as usize, stk as usize);
        let n_slabs = hd_u / stk_u;
        // `dev_off(&[mq,total],1,[r,col])` IS `Nest(["row","feat"],[mq,total])`; going through the
        // nest means this loop names a COORDINATE and never writes a stride.
        let doff =
            |r: usize, col: usize| crate::addr::rc_of(mq, total, r as u32, col as u32, Df::Fp16);
        let ew_slab = |ops: &mut Vec<EmittedOp>,
                       name: String,
                       f: &'static str,
                       a: &str,
                       aoff: crate::addr::DevOff,
                       b: &str,
                       boff: crate::addr::DevOff,
                       o: &str,
                       ooff: crate::addr::DevOff,
                       sib: &mut i64|
         -> Result<(), Error> {
            let op = pointwise_broadcast_opspec(
                op_func_from_str(f),
                mq,
                stk,
                &[
                    In::sliced(&rbo(a), aoff).ew(),
                    In::sliced(&rbo(b), boff).ew(),
                ],
                o,
                ooff.into_raw_elems(),
                // cols == stk, and `stickmajor` needs cols > 64, so it cannot fire either way — this
                // is the same rank-3 flat form regardless of the row count.
                true,
            )
            .map_err(|e| Error { message: e })?;
            ops.push(emit_sdsc_tiled(
                &name,
                &op,
                &SdscFoldSet::new(op.iter.cores_used()),
                sib,
                layout,
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
        if let Some(l) = layout {
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
                .at(crate::addr::Idx::<crate::addr::Row>::n(p_in as u32 * stk))
                .slab(s as u32)
                .dev();
            let a_place = crate::sdsc_abstract::OperandPlacement::of_token_stream_by_slab(
                crate::sdsc_abstract::QueryRowCount::of_mq(mq),
                heads,
                hd,
                0,
                p_in as u32,
                Df::Fp16,
            );
            let o_place = crate::sdsc_abstract::OperandPlacement::of_token_stream_by_slab(
                crate::sdsc_abstract::QueryRowCount::of_mq(mq),
                heads,
                hd,
                0,
                s as u32,
                Df::Fp16,
            );
            ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
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
                &rb(&x, mq, total),
                a_place,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
                p_off,
                &rb(&rot, mq, total),
                o_place,
                sym_id_base,
                layout,
            ));
        }
        // PREFILL KEEPS THE PER-HEAD P MATMUL — the form measured coherent on granite-8b (`7d27d734`),
        // one `[mq,hd]` window per head with the whole head dim contracted. `heads` ops instead of
        // `n_slabs`, and the prompt is right.
        for h in (0..heads as usize).filter(|_| !rot_slab_form) {
            let off_h = doff(0, h * hd_u);
            ops.push(assemble_matmul_off(
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
                &rb(&x, mq, total),
                off_h,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
                crate::addr::DevOff::ZERO,
                &rb(&rot, mq, total),
                off_h,
                sym_id_base,
                layout,
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
        ops.push(pw2(
            &format!("rope_xc_o{t}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rbo(&x)),
            In::full(&rbo(&cos)),
            &rbo(&xc),
            sym_id_base,
            layout,
        ));
        ops.push(pw2(
            &format!("rope_rs_o{t}"),
            "multiply",
            rows_all,
            cols_all,
            In::full(&rbo(&rot)),
            In::full(&rbo(&sin)),
            &rbo(&rs),
            sym_id_base,
            layout,
        ));
        // ADD, unconditionally: P already carries the rotate's sign, so there is no first-half/
        // second-half split left to make this op two ops (which the stick planes' interleaving would
        // have forced — first-half slabs are every OTHER plane, not a contiguous run).
        ops.push(pw2(
            &format!("rope_out_o{t}"),
            "add",
            rows_all,
            cols_all,
            In::full(&rbo(&xc)),
            In::full(&rbo(&rs)),
            &rbo(&out),
            sym_id_base,
            layout,
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
        if let Some(l) = layout {
            for r in [R::Rot, R::Xc, R::Rs] {
                l.synth(out_id.synth(r), &[heads * mq, hd]);
            }
        }
        let ew_rows = |ops: &mut Vec<EmittedOp>,
                       name: String,
                       f: &'static str,
                       rows: u32,
                       a: &str,
                       aoff: crate::addr::DevOff,
                       b: &str,
                       boff: crate::addr::DevOff,
                       o: &str,
                       ooff: crate::addr::DevOff,
                       sib: &mut i64|
         -> Result<(), Error> {
            let op = pointwise_broadcast_opspec(
                op_func_from_str(f),
                rows,
                hd,
                &[
                    In::sliced(&rbo(a), aoff).ew(),
                    In::sliced(&rbo(b), boff).ew(),
                ],
                o,
                ooff.into_raw_elems(),
                // cols == hd == stick, so `stickmajor` cannot fire (it needs cols>64) and this stays
                // the SAME rank-3 flat form the per-row path used — only the row count differs.
                false,
            )
            .map_err(|e| Error { message: e })?;
            ops.push(emit_sdsc_tiled(
                &name,
                &op,
                &SdscFoldSet::new(op.iter.cores_used()),
                sib,
                layout,
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
        ops.push(assemble_matmul_off(
            &format!("rope_rot_o{t}"),
            crate::sdsc_abstract::MatM::of_token_rows(heads * mq),
            crate::sdsc_abstract::MatN::of_head_dim(hd),
            crate::sdsc_abstract::MatK::of_head_dim(hd),
            crate::sdsc_abstract::MatY::unbatched(),
            crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                bmm_site::TapeLoweringSite::witness(),
            ),
            &rb(&x, heads * mq, hd),
            crate::addr::DevOff::ZERO,
            &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
            crate::addr::DevOff::ZERO,
            &rb(&rot, heads * mq, hd),
            crate::addr::DevOff::ZERO,
            sym_id_base,
            layout,
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
            &mut ops,
            format!("rope_xc_o{t}"),
            "multiply",
            all,
            &x,
            crate::addr::DevOff::ZERO,
            &cos,
            crate::addr::DevOff::ZERO,
            &xc,
            crate::addr::DevOff::ZERO,
            sym_id_base,
        )?;
        ew_rows(
            &mut ops,
            format!("rope_rs_o{t}"),
            "multiply",
            all,
            &rot,
            crate::addr::DevOff::ZERO,
            &sin,
            crate::addr::DevOff::ZERO,
            &rs,
            crate::addr::DevOff::ZERO,
            sym_id_base,
        )?;
        ew_rows(
            &mut ops,
            format!("rope_add_o{t}"),
            "add",
            all,
            &xc,
            crate::addr::DevOff::ZERO,
            &rs,
            crate::addr::DevOff::ZERO,
            &out,
            crate::addr::DevOff::ZERO,
            sym_id_base,
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
            ops.push(assemble_matmul_off(
                &format!("rope_rot_r{r}_h{h}_o{t}"),
                crate::sdsc_abstract::MatM::single_row(),
                crate::sdsc_abstract::MatN::of_head_dim(hd),
                crate::sdsc_abstract::MatK::of_head_dim(hd),
                crate::sdsc_abstract::MatY::unbatched(),
                crate::ir::bridge::tiled_op_sdsc_op::SharedKernelBmmForm::batch_inner_proven(
                    bmm_site::TapeLoweringSite::witness(),
                ),
                &rb(&x, 1, hd),
                off,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &p),
                crate::addr::DevOff::ZERO,
                &rb(&rot, 1, hd),
                off,
                sym_id_base,
                layout,
            ));
            ew_ph(
                &mut ops,
                format!("rope_xc_r{r}_h{h}_o{t}"),
                "multiply",
                &x,
                off,
                &cos,
                coff,
                &xc,
                off,
                sym_id_base,
            )?; // x·cos
            ew_ph(
                &mut ops,
                format!("rope_rs_r{r}_h{h}_o{t}"),
                "multiply",
                &rot,
                off,
                &sin,
                coff,
                &rs,
                off,
                sym_id_base,
            )?; // rot·sin
            ew_ph(
                &mut ops,
                format!("rope_add_r{r}_h{h}_o{t}"),
                "add",
                &xc,
                off,
                &rs,
                off,
                &out,
                off,
                sym_id_base,
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
        sym_id_base,
        layout,
    } = a;
    // ⭐⭐⭐ THE OPERANDS COME FROM THE PROGRAM. main read them off `node.inputs` / `node.op` /
    // `ir.tensors[..].rows`; the program states every one of them, so this is the same reading through
    // the door this file's header describes — not a record the producer filled in beside it.
    let ops_in = attn_operands(_name, k, r, geom)?;
    let mq = ops_in.mq;
    let (k_id, v_id) = (ops_in.k_id, ops_in.v_id);
    let (nqh, nkvh, hd) = (geom.nqh(), geom.nkvh(), geom.hd());
    let cap = ops_in.cap;
    let stick = Fp16::ELEMS_PER_STICK; // 64
    let pool = crate::sdsc_abstract::PagedKvPool::new(nkvh as usize, hd as usize);
    // ── PAGED-ATTENTION COMPUTE EXTENT, READ OFF THE PROGRAM ──
    //
    // ⭐⭐⭐ IT WAS `AttnAt::active_cap`, THE RUNG, RESOLVED HERE. The rung is a PRODUCER input: it
    // decides how many rows of the resident cache `KtirFunc::attn` takes an access tile of, so the
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
    let active_cap: u32 = prefix_rows;
    // ⛔⛔⛔ THE RUNG AND THE PROGRAM MUST AGREE ABOUT WHETHER THERE IS A PREFIX AT ALL, because
    // that is the one place the two consumers could compute different things and neither would fault.
    // `ActiveCap::NONE` means `nb == 0` here — the new-token block alone seeds and finalizes the
    // softmax — and the producer says the same thing by taking NO access tile of the resident cache
    // (the tensor is still named, so its identity survives; see `KtirFunc::attn`'s zero-length
    // segment). A rung that swept nothing against a program that reads the prefix — or the reverse —
    // is an emulator that cannot be an oracle for this node, so it is a build error naming both.
    // The swept extent cannot exceed the buffer it sweeps: the cache parameter's own view states
    // `cap` rows, and a program taking a window past that addresses bytes nothing wrote.
    if prefix_rows > cap {
        return err(format!(
            "AttnDecode t{}: the program reads {prefix_rows} row(s) of the resident cache t{k_id} \
             whose own view states only {cap} — the capacity read off that view is not the buffer \
             this program addresses.",
            ops_in.out_id,
        ));
    }
    // ⭐⭐⭐ THE MULTIPLIER COMES FROM THE PROGRAM, AND IT IS THE ONLY READING OF IT. `KtirFunc::attn`
    // states `attention_multiplier` inline (`self.scalar(f64::from(scale))`) because the emulator
    // interprets these ops and the three attention optimizers recognise the scale by pattern; the
    // DEVICE reads it as a bound `[1,1]` const, whose slot is resolved below FROM THIS VALUE.
    //
    // ⛔ IT WAS A CALLER-STATED `f32` CHECKED AGAINST THIS DERIVATION, WHICH IS A SIDECAR. The check
    // could only ever pass — every program that lowered at all agreed with its caller — so the value
    // the caller passed was never the source of anything, only a second copy of one config number
    // travelling `SubtileIR → params → SuperDSC` past the IR. A third-party producer had to supply a
    // number its own program already states. There is one reading now, and no default: a program with
    // no multiplier is a build error naming it.
    let scale_val = program_score_scale(&k.func).ok_or_else(|| Error {
        message: format!(
            "AttnDecode t{}: the program states no score multiplier. `KtirFunc::attn` scales its \
             scores with an `arith.mulf` against an `arith.constant`, and the descriptor's `[1,1]` \
             scale const is resolved from that value, so a program without one cannot be lowered.",
            ops_in.out_id,
        ),
    })?;
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
                ops_in.out_id,
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
                ops_in.out_id
            ),
        })?;
    let width = bundle_rows.width();
    let mq_pad = width.pad().rows();
    let t = ops_in.out_id;
    let q = crate::place::act_name(ops_in.q_id); // roped Q [mq, nqh·hd]
    let new_k = crate::place::act_name(ops_in.new_k_id); // roped new-K [mq_pad, nkvh·hd] (worker zero-pads rows)
    let new_v = crate::place::act_name(ops_in.new_v_id); // new-V       [mq_pad, nkvh·hd]
    let kc = crate::place::act_name(k_id); // natural K cache [nqh, cap, hd] (seg2, GQA-replicated, slab-major)
    let vc = crate::place::act_name(v_id); // natural V cache [nqh, cap, hd] (seg2, GQA-replicated, slab-major)
    let kct = crate::place::act_name(kct_resident_tid(k_id)); // resident Kᵀ scratch [nqh, hd, cap]
    // attention_multiplier (config) — see the ORIGINAL header doc: NO 1/sqrt(hd) recompute.
    let scale_idx = layout
        .and_then(|l| {
            l.scalarmul_scales
                .iter()
                .position(|s| s.to_bits() == scale_val.to_bits())
        })
        .ok_or_else(|| Error {
            message: format!(
                "AttnDecode t{t}: scale {scale_val} absent from BundleLayout.scalarmul_scales \
                 (registry desync)"
            ),
        })?;
    let _scale = crate::place::act_name(scalarmul_scale_tid(scale_idx)); // unused directly: torch-spyre splits into sqrt_scale on both Q and K.
    let sqrt_scale_val = scale_val.sqrt();
    let sqrt_scale_idx = layout
        .and_then(|l| {
            l.scalarmul_scales
                .iter()
                .position(|s| s.to_bits() == sqrt_scale_val.to_bits())
        })
        .ok_or_else(|| Error {
            message: format!(
                "AttnDecode t{t}: √scale {sqrt_scale_val} absent from scalarmul_scales (registry \
                 desync)"
            ),
        })?;
    let sqrt_scale = crate::place::act_name(scalarmul_scale_tid(sqrt_scale_idx)); // [1,1] = √attention_multiplier
    let pmask = crate::place::act_name(ATTN_MASK_TID); // [nqh, cap]    prefix validity (worker-tiled, mb-broadcast over mq)
    let cmask = crate::place::act_name(ATTN_CAUSAL_TID); // [mq, mq_pad] causal triu (worker-tiled)
    // [mq, nqh·hd] — the identity; every scratch name below is a rendering of it.
    let attn_id = PlaceId::Act(t);
    let n = |r: SynthRole| syn(layout, attn_id.synth(r));
    let ident = crate::place::act_name(IDENTITY_TID);

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
                &format!("attn_kzero{kvh}_o{t}"),
                "identity",
                crate::sdsc_abstract::RowCount::of_zero_pad_rows(pad_rows),
                crate::sdsc_abstract::BlockCols::of_head_dim(hd),
                &[In::full(&rb(&zero, pad_rows, hd)).ew()],
                &rb(&new_k, mq_pad.row_axis_extent(), nkvh * hd),
                // FLAT, and that is the finding: this lands past the packed data, in a region no
                // consumer reads (see the note above), so it is placed row-major rather than through
                // the stick law. Saying `flat` makes which arrangement was meant explicit instead of
                // leaving `mq*nkvh*hd + kvh*hd` for a reader to infer.
                crate::addr::Nest::flat(
                    &["row", "feat"],
                    &[mq_pad.row_axis_extent(), nkvh * hd],
                    Df::Fp16,
                )
                .view()
                .at(crate::addr::Idx::<crate::addr::Row>::n(mq))
                .at(crate::addr::Idx::<crate::addr::Feat>::n(kvh * hd))
                .dev(),
                sym_id_base,
                layout,
            ));
            ops.push(assemble_pointwise_broadcast_off(
                &format!("attn_vzero{kvh}_o{t}"),
                "identity",
                crate::sdsc_abstract::RowCount::of_zero_pad_rows(pad_rows),
                crate::sdsc_abstract::BlockCols::of_head_dim(hd),
                &[In::full(&rb(&zero, pad_rows, hd)).ew()],
                &rb(&new_v, mq_pad.row_axis_extent(), nkvh * hd),
                crate::addr::Nest::flat(
                    &["row", "feat"],
                    &[mq_pad.row_axis_extent(), nkvh * hd],
                    Df::Fp16,
                )
                .view()
                .at(crate::addr::Idx::<crate::addr::Row>::n(mq))
                .at(crate::addr::Idx::<crate::addr::Feat>::n(kvh * hd))
                .dev(),
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
    let krep_nslab = (hd / stick).max(1);
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
                &rb(&new_k, mq_pad.row_axis_extent(), nkvh * hd),
                slab_off,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &ident),
                ident_off,
                &rb(&new_k_rep, mq_pad.row_axis_extent(), nkvh * hd),
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
                &rb(&new_v, mq_pad.row_axis_extent(), nkvh * hd),
                slab_off,
                &Stk::<KernelTag>::kernel(hd as usize, hd as usize, &ident),
                ident_off,
                &rb(&new_v_rep, mq_pad.row_axis_extent(), nkvh * hd),
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
    let (qs, new_k_scaled) = (n(SynthRole::Qs), n(SynthRole::NewKScaled));
    if let Some(l) = layout {
        l.synth(attn_id.synth(SynthRole::Qs), &[mq, nqh * hd]);
        l.synth(
            attn_id.synth(SynthRole::NewKScaled),
            &[mq_pad.row_axis_extent(), nkvh * hd],
        );
    }
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_qs_o{t}"),
        "multiply",
        // Q is `[mq, nqh*hd]`: the chunk's REAL rows, the full query feature width.
        crate::sdsc_abstract::RowCount::of_query_rows(crate::sdsc_abstract::QueryRowCount::of_mq(
            mq,
        )),
        crate::sdsc_abstract::BlockCols::of_feature_cols(nqh * hd),
        &[In::full(&rbo(&q)).ew(), In::scalar(&rbo(&sqrt_scale)).ew()],
        &rbo(&qs),
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));
    ops.push(assemble_pointwise_broadcast_off(
        &format!("attn_nks_o{t}"),
        "multiply",
        // new-K is allocated `[mq_pad, nkvh*hd]`: the scale covers the PADDED rows, zeros included.
        crate::sdsc_abstract::RowCount::of_padded_chunk_rows(mq_pad),
        crate::sdsc_abstract::BlockCols::of_feature_cols(nkvh * hd),
        &[
            In::full(&rbo(&new_k_rep)).ew(),
            In::scalar(&rbo(&sqrt_scale)).ew(),
        ],
        &rbo(&new_k_scaled),
        crate::addr::DevOff::ZERO,
        sym_id_base,
        layout,
    ));

    // ── (3) the unified score/softmax/output computation — ONE algorithm for any mq (see
    // ir::bridge::tiled_op_sdsc_op::attn's module doc for the torch-spyre correspondence). The head
    // geometry travels as the minted type, not as three integers this call could reorder, and the
    // width travels fused to its row laws on the boundary's one carrier.
    ops.extend(assemble_attn(
        t,
        geom,
        bundle_rows,
        cap,
        active_cap,
        &qs,
        &new_k_scaled,
        &new_v_rep,
        &kct,
        &vc,
        &pmask,
        &cmask,
        attn_id,
        rows_are_requests,
        sym_id_base,
        layout,
    )?);

    // ── (4) KV CACHE WRITE: append this step's new_k_scaled (nkvh-wide, dedup)/new_v_rep (nqh-wide)
    // into the resident cache. K writes only the nkvh representative slots (kvh*gqa) — `kctpost`
    // (below) only ever reads those; the other slots were always dead. V is UNCHANGED (nqh writes,
    // genuinely all read via `vc`'s own nqh-replicated convention). Writes K ALREADY sqrt-scaled, so
    // future steps' prefix scores need no further K scaling.
    //
    // REQUEST-MAJOR, so that one request's kv-head copies are CONSECUTIVE and fuse into a
    // single launch. A group resolves one request's page table and write cursor, so the run
    // breaks where the request changes (`GroupKind::Slot`'s `req`) — kv-head-major order would
    // alternate requests every op and leave every copy its own launch, which is what made a
    // batch of 8 slower than 8 separate forwards. One request ⇒ this loop runs once and the
    // order is exactly what it was.
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
    for req in 0..if per_request { mq } else { 1 } {
        // ⛔ NO REQUEST TERM IN THE ADDRESS. `req` here indexes a ROW OF THE ACTIVATION and nothing more:
        // which page that row's token lands in is the host's page map, applied as a per-launch shift. A page
        // holds slots, not requests, so the address inside it is `(kv head, slot, feature)`.
        let rq = if per_request {
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
        for s in 0..slabs.get() {
            for h in 0..if heads_on_y { 1 } else { nkvh } {
                let a_place = if heads_on_y {
                    crate::sdsc_abstract::OperandPlacement::of_plane_walk_by_head(
                        &src_nest,
                        crate::addr::Idx::<crate::addr::Row>::n(req),
                        s,
                        slabs,
                    )
                } else {
                    crate::sdsc_abstract::OperandPlacement::at_head(
                        &src_nest,
                        crate::addr::Idx::<crate::addr::Row>::n(req),
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
                for (base, src, dst) in [("kc", &new_k_scaled, &kc), ("vc", &new_v_rep, &vc)] {
                    ops.push(crate::ir::bridge::tiled_op_sdsc_op::assemble_matmul_placed(
                        &if heads_on_y {
                            format!("cachewr_{base}s{s}{rq}_o{t}")
                        } else {
                            format!("cachewr_{base}{h}s{s}{rq}_o{t}")
                        },
                        // The kv stream's REAL rows this copy covers: one per request-copy, else the chunk's.
                        crate::sdsc_abstract::MatM::of_query_rows(
                            crate::sdsc_abstract::QueryRowCount::of_mq(if per_request {
                                1
                            } else {
                                mq
                            }),
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
                        &rb(src, mq, nkvh * hd),
                        a_place,
                        // I[64,64] — the identity for EVERY slab, so the shared 2-D kernel a `y` batch demands is
                        // exactly what an identity copy needs. Offset 0: see the residency argument above.
                        &Stk::<KernelTag>::kernel(stick as usize, stick as usize, &ident),
                        crate::addr::DevOff::ZERO,
                        &Stk::<KernelTag>::kernel(
                            crate::sdsc_abstract::PagedKvPool::PLANE_SLOTS,
                            hd as usize,
                            dst,
                        ),
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
    let kct_reqs = if per_request { mq } else { 1 };
    for (req, kvh) in (0..kct_reqs).flat_map(|r| (0..nkvh).map(move |k| (r, k))) {
        let rq = if per_request {
            format!("_r{req}")
        } else {
            String::new()
        };
        // WHOLE-PAGE re-transpose, structurally the proven full-`cap` form with the page (equal to
        // that cap) substituted — NOT the slab-granular variant, whose on-card attempt regressed
        // decode. Source and destination take the identical `layer + page` shift, which is why
        // natural K lives in the page rather than a segment of its own.
        ops.push(assemble_restickify_kt_2d(
            &format!("attn_kctpost{kvh}{rq}_o{t}"),
            crate::sdsc_abstract::KtTileSlots::of_page(),
            crate::sdsc_abstract::KtTileFeats::of_head_dim(hd),
            &kc,
            // The two planes are the whole content of this op: it reads natural K and writes Kᵀ, both at
            // the same `(kv head)` block of the page the launch was shifted to.
            crate::addr::DevOff::from_view_step(pool.addr(crate::sdsc_abstract::KvCoord::block(
                crate::sdsc_abstract::KvPlane::Knat,
                kv_head_of(kvh, nkvh)?,
            ))),
            &kct,
            crate::addr::DevOff::from_view_step(pool.addr(crate::sdsc_abstract::KvCoord::block(
                crate::sdsc_abstract::KvPlane::Kt,
                kv_head_of(kvh, nkvh)?,
            ))),
            sym_id_base,
            layout,
        ));
        // ⭐ TAGGED WITH ITS ROW, for the same reason the cache writes are: this op reads natural K and
        // writes Kᵀ within ONE row's page, and with no request term left in the address the tag is the only
        // thing that tells the runtime which page that is.
        if let Some(o) = ops.last_mut() {
            o.kv_request = req;
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
/// address this target can simply name". The KTIR can name it — `KtirFunc::matmul` emits the corner as
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
    let src_name = rb(&src.name(), 1, stk);
    let dst = rbo(&out.name());
    Ok((0..hidden / stk)
        .map(|j| {
            assemble_pointwise_broadcast_off(
                &format!("lmlast{j}_o{owner}"),
                "identity",
                crate::sdsc_abstract::RowCount::of_token_rows(1),
                crate::sdsc_abstract::BlockCols::of_one_stick(crate::sdsc_abstract::Lanes::FP16),
                &[In::sliced(
                    &src_name,
                    // row `j*mq + row` of a `[.., stk]` staging — a corner, not a product.
                    crate::addr::rc_of((j + 1) * mq + row + 1, stk, j * mq + row, 0, Df::Fp16),
                )
                .ew()],
                &dst,
                // Column `j*stk` of the single-row `[1, hidden]` last-hidden staging — a corner on
                // that tensor, not a product. At one row the stick plane term is 0, so this is the
                // same address either way; saying it as a coordinate keeps the last hand-built offset
                // in this file from being the one nobody notices.
                crate::addr::rc_of(1, hidden, 0, j * stk, Df::Fp16),
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
pub fn matmul(
    name: &str,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
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
    base_addressed(name, &w, "weight (W)")?;
    base_addressed(name, &out, "output")?;
    // ⭐ THE PRECISION IS THE WEIGHT VIEW'S ELEMENT TYPE, and arity agrees with it by construction:
    // `KtirFunc::matmul_fp8` is the only builder that views a weight through `view_fp8`, and it is
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
                w_name: w.name(),
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
    // `KtirFunc::matmul` views the same buffer as its natural `[n, k]`, so the two checks swap sides
    // and nothing else about them changes.
    if w.c_len != k {
        return err(format!(
            "MatmulTile t{}: W cols {} != A cols (K) {k}",
            out.tid, w.c_len
        ));
    }
    if w.r_len != n {
        return err(format!(
            "MatmulTile t{}: W rows {} != out cols (N) {n}",
            out.tid, w.r_len
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
    // TYPE-SAFE device width (the padding/alignment invariant): `DeviceWidth::for_output` is the SOLE
    // rule, SHARED with the worker's weight zero-pad + `kernel0`, so they cannot diverge.
    let n_dev = DeviceWidth::for_output(m, n, k).get();
    let macs = m as u64 * n_dev as u64 * k as u64;
    // ── GUARD #11 (util floor) computed on the PROVEN partition `CoreSplit::plan` (Kani: disjoint +
    //    covering, #50-free) — the SAME split the emit uses (`matmul_split_map` defers to CoreSplit for
    //    batch=1). A FLOP-heavy matmul left on <8 cores is the 1.45× sengraph regression this emitter
    //    exists to fix; the padding above is what fills them for a prime-stick output. ──
    let sp = CoreSplit::plan(m, n_dev);
    let cores = sp.ncores();
    if macs >= (1 << 20) && cores < 8 {
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
    // ⭐ THE `try_` FORM, WHICH IS WHAT IT WAS BUILT FOR. `assemble_matmul_seeded` unwraps the
    // builder's `Err` into a `panic!` — right for the sub-stick witness its doc names ("not reachable
    // from model input"), wrong for a refusal that depends on the CALLER's shapes. Two now do:
    // `resolve_seg_base`'s placement-footprint check, and `refuse_reduction_core_split`'s
    // unschedulable-`in`-split seal. As a panic either exits 101 with NO stage label, so a driver
    // tabulating `REFUSED <stage> <message>` shows it as a blank row — which is exactly how
    // `swiglu_mlp_granite_flat`'s reduction split first surfaced. The `Result` form was added for this
    // and had no caller; this is that caller.
    let op = try_assemble_matmul_seeded(
        &op_name,
        m,
        n_dev,
        k,
        1,
        &rb(&a.name(), m, k),
        &Stk::<KernelTag>::kernel(k as usize, n_dev as usize, w.name()),
        &rb(&out.name(), m, n_dev),
        sym_id_base,
        layout,
    )
    .map_err(|message| Error {
        message: format!("MatmulTile t{}: {message}", out.tid),
    })?;
    // Default path: one f16 matmul.
    Ok(vec![op])
}


/// A BARE ROW REDUCTION — `[rows, cols]` → `[rows, 1]`, one `sfp` op along the stick axis.
///
/// ⭐ THE ASSEMBLER WAS ALREADY WRITTEN; WHAT WAS MISSING WAS A DOOR — the same story as
/// [`transpose`]. `assemble_reduce` / `assemble_reduce_seeded` have carried the `sum`/`max`/`mean`
/// reduce since the SubtileIR path, and `assemble_rmsnorm` calls them, but only from INSIDE a fused
/// kind. A producer whose program states a bare `linalg.reduce` — every Triton `tl.sum(x, 1)` or
/// `tl.max(x, 1)` that is not part of a recognised rmsnorm — had no entry point at all.
///
/// # ⛔⛔⛔ A MULTI-ROW MAX IS REFUSED, AND IT IS THE SHARPEST SILENT WRONG ANSWER IN THIS FILE
///
/// [`reduce.rs`](crate::ir::bridge::tiled_op_sdsc_op::reduce) records a MEASURED device defect at its
/// own code, above `reduce_opspec_off`:
///
/// > The on-card reduce-**MAX** returns 0 (the seed) when `rows>1` (PROVEN via attn diag: mxp=0 over
/// > [nqh,cap]) but is CORRECT at rows=1 (the rmsnorm `rmamax` works). reduce-SUM is fine multi-row
/// > (the score reduce gives sane scores), so only the softmax max-reduce needs splitting into nqh
/// > single-row reduces.
///
/// So a `max` over more than one row LOWERS, BAKES, produces a well-formed descriptor, exits 0 under
/// `dxp_standalone` — and returns ZERO for every row. Nothing in the compile path can see it, because
/// `dxp_standalone` executes no arithmetic. A decoder's softmax is `m = tl.max(qk, 1)` over BLOCK_M
/// rows (64 in every configuration this backend compiles), so this is not a corner: it is the exact
/// shape the target kernel states.
///
/// It is refused BY NAME rather than emitted, and the refusal names the remedy `reduce.rs` itself
/// records — one single-row reduce per row. That splitting is NOT done here: it is `rows` descriptors
/// instead of one, so it is an emission-shape decision with its own cost (64 SuperDSCs per softmax)
/// and its own addressing (a per-row element offset, which is what `reduce_opspec_off` exists for).
/// Choosing it silently on the producer's behalf is exactly what this door does not do.
///
/// `Sum` is unaffected at any row count, and `Max` at `rows == 1` is the case the shipped rmsnorm
/// `rmamax` proves, so both are lowered.
pub fn reduce(
    name: &str,
    kind: crate::ktir_node::ReduceKind,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    use crate::ktir_node::ReduceKind;
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
    // ⛔⛔⛔ THE MEASURED DEVICE DEFECT, FIRST, so no other complaint can mask it.
    if matches!(kind, ReduceKind::Max) && rows > 1 {
        return err(format!(
            "{name}: a `max` reduce over t{}'s {rows} rows is REFUSED, and not because it cannot be \
             described. `ir/bridge/tiled_op_sdsc_op/reduce.rs` records, at its own code and proven by \
             an attention diagnostic (mxp=0 over [nqh,cap]), that the ON-CARD reduce-MAX returns 0 — \
             THE SEED — whenever `rows > 1`, and is correct only at `rows == 1`. So this would emit a \
             well-formed descriptor, bake, exit 0 under `dxp_standalone` (which executes no \
             arithmetic), and hand back ZERO for all {rows} rows. A softmax built on it computes \
             `exp(x - 0)`, silently. THE REMEDY, which `reduce.rs` names: split it into {rows} \
             single-row reduces, one per row, addressed through `reduce_opspec_off`'s per-operand \
             element offsets. That is {rows} descriptors instead of one and an addressing decision \
             this door will not take on the producer's behalf — state it, or reduce with `sum`, which \
             is correct multi-row.",
            data.tid
        ));
    }
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
            &rb(&data.name(), rows, cols),
            &rb(&out.name(), rows, 1),
            sym_id_base,
            layout,
        ),
    ])
}

/// THE REDUCE DOOR'S FAIL-CLOSED HALF, and the multi-row `max` is the reason this module exists.
///
/// ⛔⛔⛔ The refusal these tests pin is the ONLY thing standing between a decoder's softmax and a
/// silent wrong answer. `ir/bridge/tiled_op_sdsc_op/reduce.rs` records, at its own code and proven by
/// an attention diagnostic, that the on-card reduce-MAX returns 0 — the SEED — whenever `rows > 1`.
/// A multi-row max therefore emits a well-formed descriptor, bakes, and exits 0 under
/// `dxp_standalone`, which executes no arithmetic. NOTHING in the compile path can see it. So the
/// guard is tested before it is trusted, and every case carries its accepting control.
#[cfg(test)]
mod reduce_tests {
    use super::*;
    use crate::ktir_node::ReduceKind;

    /// A whole-buffer operand: the view IS the window and the store windows tile it.
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
            is_out,
            is_fp8: false,
        }
    }

    /// `[data, out]` — the parameter list a reduce's `split_out(.., 1)` reads.
    fn parms(rows: u32, cols: u32) -> Vec<Region> {
        vec![reg(1, rows, cols, false), reg(2, rows, 1, true)]
    }

    fn lower(kind: ReduceKind, rows: u32, cols: u32) -> Result<Vec<EmittedOp>, Error> {
        let mut sid = 0i64;
        reduce("probe", kind, &parms(rows, cols), &mut sid, None)
    }

    /// The refusal message, or a panic naming that something was emitted. Hand-written because
    /// `EmittedOp` is deliberately not `Debug`, so `expect_err` cannot be used — and a test that reads
    /// a refusal must fail loudly when there is none.
    fn refusal(kind: ReduceKind, rows: u32, cols: u32) -> String {
        match lower(kind, rows, cols) {
            Ok(ops) => panic!(
                "expected a refusal, but {} descriptor(s) were emitted for a {rows}-row {:?} reduce",
                ops.len(),
                kind
            ),
            Err(e) => e.message,
        }
    }

    /// ⛔⛔⛔ THE MEASURED DEVICE DEFECT. A `max` over more than one row must be refused, and the
    /// refusal must name the seed, the row count and the remedy — a reader who only sees "refused"
    /// will reach for the extents.
    #[test]
    fn a_multi_row_max_is_refused_and_the_refusal_names_the_seed() {
        let m = refusal(ReduceKind::Max, 64, 128);
        for want in ["THE SEED", "64", "single-row"] {
            assert!(
                m.contains(want),
                "the refusal must name `{want}` — it is the difference between a reader fixing the \
                 shape and a reader understanding that the device returns zero. Got: {m}"
            );
        }
    }

    /// THE CONTROL THAT MAKES THE ABOVE MEAN SOMETHING: the same shape with `sum` is fine, because
    /// `reduce.rs` records reduce-SUM as correct multi-row. If this ever refuses too, the test above is
    /// passing for the wrong reason.
    #[test]
    fn a_multi_row_sum_is_lowered_because_only_max_is_affected() {
        let ops = lower(ReduceKind::Sum, 64, 128)
            .expect("reduce-SUM is correct multi-row; only the max returns the seed");
        assert_eq!(ops.len(), 1, "a row reduction is one descriptor");
    }

    /// THE SECOND CONTROL: `max` at `rows == 1` is the case the shipped rmsnorm `rmamax` proves on
    /// card, so the guard must be about the ROW COUNT and not about `max` itself.
    #[test]
    fn a_single_row_max_is_lowered_because_that_is_the_case_the_device_gets_right() {
        let ops = lower(ReduceKind::Max, 1, 128)
            .expect("`max` at rows == 1 is correct on card — the rmsnorm `rmamax` uses it");
        assert_eq!(ops.len(), 1, "a row reduction is one descriptor");
    }

    /// An output wider than one column would leave every column but the first holding whatever the
    /// buffer held, so it is named rather than emitted.
    #[test]
    fn an_output_that_is_not_one_column_per_row_is_refused() {
        let mut sid = 0i64;
        let parms = vec![reg(1, 64, 128, false), reg(2, 64, 8, true)];
        let e = reduce("probe", ReduceKind::Sum, &parms, &mut sid, None)
            .err()
            .expect("a reduce writes exactly one value per row");
        assert!(
            e.message.contains("ONE value per row"),
            "the refusal must say what a row reduction writes; got: {}",
            e.message
        );
    }

    /// And a row count that disagrees between the data and the accumulator.
    #[test]
    fn a_row_count_that_disagrees_between_data_and_output_is_refused() {
        let mut sid = 0i64;
        let parms = vec![reg(1, 64, 128, false), reg(2, 32, 1, true)];
        let e = reduce("probe", ReduceKind::Sum, &parms, &mut sid, None)
            .err()
            .expect("a reduce writes one value per row it reads");
        assert!(
            e.message.contains("one value per row it reads"),
            "the refusal must name the disagreement; got: {}",
            e.message
        );
    }
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
/// alternative, and the builder's OTHER refusal — the per-core 8×8 block division — arrives as an
/// `Error` through [`super::try_assemble_transpose`]. The panicking form is never called from here.
///
/// ⚠️ THE CORE DIVISION NARROWS THIS FURTHER THAN THE STICK LAW ALONE, AND THAT IS A DEFECT ELSEWHERE,
/// NOT A LAW OF THE DEVICE. `distribute_cores` splits `mb` alone to all 32 cores, so today only a row
/// extent that is a multiple of `8 · 32 = 256` gives each core whole 8-blocks — the vendor's own golden
/// `sdsc_interslicetranspose.json` splits BOTH axes and its shape (`mb 384`) is therefore REFUSED here.
/// See [`super::transpose_opspec`]'s guard: the fix is a core division measured against that fixture,
/// which is out of scope of an entry point and must not be guessed from one fixture point.
pub fn transpose(
    name: &str,
    r: &[Region],
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, Error> {
    let (ins, out) = split_out(name, r, layout, 1)?;
    let a = ins[0];
    // The INPUT's view IS the op's `[mb, out]`: `assemble_transpose` iterates the input shape and the
    // output is its transpose. Read from the input so the two cannot be read from the same place and
    // agree vacuously.
    let (mb, cols) = (a.v_rows, a.v_cols);
    // ⛔⛔⛔ THE STICK LAW, FIRST, so the historically fatal shape can never fall past it into a
    // shape complaint about something else.
    let stk = Fp16::ELEMS_PER_STICK;
    if mb == 0 || cols == 0 || !mb.is_multiple_of(stk) || !cols.is_multiple_of(stk) {
        return err(format!(
            "{name}: transposing t{} `[{mb}, {cols}]` into t{} `[{cols}, {mb}]` — BOTH extents must \
             be whole multiples of the {stk}-element SEN169_FP16 stick and {} is not. An \
             `interslicetranspose_fp16` sticks its input on the column extent and its output on the \
             8x8 inter-slice block over (columns, rows), so each extent is a stick extent of one side; \
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
                 `assemble_transpose` names a tensor and its two extents, with no column corner and \
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
    // ⛔ THE FALLIBLE FORM, because the builder makes a refusal this body cannot pre-check: the
    // per-core division must land on the output's 8×8 block, and what the division IS depends on
    // `distribute_cores`' answer for this shape. Restating that here would mean duplicating the
    // divider; calling the panicking `assemble_transpose` would abort the build with a bare panic
    // where a producer needs an error against its own op. See `super::try_assemble_transpose`.
    super::try_assemble_transpose(
        &format!("transpose_o{}", out.tid),
        mb,
        cols,
        &a.name(),
        &out.name(),
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
        .filter(|x| !x.is_out && scale_idx_of(layout, x.tid).is_none())
        .collect();
    if ins.len() != arity {
        return err(format!(
            "{name}: {} tensor input(s) for an op that reads {arity}",
            ins.len()
        ));
    }
    // Every operand these bodies emit is base-addressed; a stated row corner cannot ride one.
    base_addressed(name, &out, "output")?;
    for (i, x) in ins.iter().enumerate() {
        base_addressed(name, x, &format!("input {i}"))?;
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
            is_out,
            is_fp8: false,
        }
    }

    /// A binary node's parameter list in producer order: `[in0, in1, out]`.
    fn node(a: Region, b: Region, o: Region) -> Vec<Region> {
        vec![a, b, o]
    }

    // ── (ii) REFUSAL + A CONTROL THAT STILL LOWERS ────────────────────────────────────────────

    /// ⛔ THE SHAPE THAT MOTIVATED THE GUARD, AND IT IS LIVE. `LoweredOp::BiasAdd =>
    /// SubOp::Elementwise(EwKind::Add)` (`crates/compiler/subtile/src/subtile_ir.rs`) whose bias is
    /// rank-1 `[D]` (`crates/compiler/macros/src/shape.rs`, `sig_bias_add`), and `BiasAdd` is ABSENT
    /// from that file's `elementwise` column-tiling list, so its operands are taken WHOLE — a
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
        let Err(e) = elementwise("add_s3", Elementwise::Add, &r, &mut sym, None) else {
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
        let ops = elementwise("add_s3", Elementwise::Add, &r, &mut sym, None)
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
        let Err(e) = elementwise("sub_s3", Elementwise::Sub, &r, &mut sym, None) else {
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
        let Err(e) = elementwise("silu_s3", Elementwise::Silu, &r, &mut sym, None) else {
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
        let ops = elementwise("sub_s3", Elementwise::Sub, &r, &mut sym, None)
            .expect("a same-extent subtract is structurally an add");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_name, "sub_o7");
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
            // UNARY: one input, so the quantifier is exercised at arity 1.
            for a in EXTENTS {
                let ins = [reg(5, a.0, a.1, false)];
                assert_eq!(
                    pointwise_extents_agree("p", Elementwise::Silu, &ins, &out).is_ok(),
                    a == o,
                    "unary out={o:?} in={a:?}"
                );
            }
            // BINARY: admitted only when BOTH inputs match — neither operand may be the broadcast.
            for a in EXTENTS {
                for b in EXTENTS {
                    let ins = [reg(5, a.0, a.1, false), reg(6, b.0, b.1, false)];
                    assert_eq!(
                        pointwise_extents_agree("p", Elementwise::Add, &ins, &out).is_ok(),
                        a == o && b == o,
                        "binary out={o:?} in0={a:?} in1={b:?}"
                    );
                }
            }
        }
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
            is_out,
            is_fp8: false,
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
        let admitted = pointwise_extents_agree("p", Elementwise::Add, &ins, &out).is_ok();
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
        let admitted = pointwise_extents_agree("p", Elementwise::Silu, &ins, &out).is_ok();
        assert_eq!(admitted, (arow, acol) == (orow, ocol));
    }
}
