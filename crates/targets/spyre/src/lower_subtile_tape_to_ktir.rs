// SPDX-License-Identifier: Apache-2.0
//! SubtileIR → **KTIR**: the PRODUCER half of `SubtileIR → KTIR → SuperDSC`.
//!
//! ⭐⭐⭐ THE PROGRAM IS CONSTRUCTED, NEVER PRINTED. `ktir-core`'s `Operation` / `IRFunction` ARE the
//! interchange: the emulator executes the value directly and `#[forward]` bakes it as const data.
//! Nothing here renders MLIR and nothing anywhere parses it.
//!
//! ⭐ ONE NODE, ONE PROGRAM. Each [`SubtileNode`] becomes one KTIR `func` — inputs loaded from HBM,
//! output stored back — carried on an [`EmittedOp`] as `ktir`. Both consumers read that SAME value:
//! `-Fspyre-emu` interprets it, `-Fspyre-hw` lowers it through
//! [`crate::ktir_superdsc_door`].
//!
//! ⛔ NOTHING HERE EMITS A SuperDSC DESCRIPTOR. That is the consumer's half, in its own file, and a
//! node lowered straight to SuperDSC from here would be a second path to the same format.
//!
//! [`EmittedOp`]: crate::lower_subtile_tape_to_superdsc::EmittedOp

use crate::lower_subtile_tape_to_superdsc::*;
// ⭐ THE REQUEST TYPE NOW LIVES IN `ktir-superdsc`, and is NAMED here rather than re-exported. `KtirNode`
// is what this producer BUILDS and the lowering consumes, so it belongs to the leaf crate that defines
// the contract.
// ⭐ THE EMITTER IS `ktir_superdsc::emit`. This producer names the one item it actually builds with
// directly, rather than picking it out of the glob above — see that module's note on why the
// emitter's own import is private.
use ktir_core::arena::Arena;
use ktir_core::attrkey::AttrKey;
use ktir_core::ir::{Attr, Operation, Ssa};
use ktir_core::irtype::IrType;
use ktir_core::opkind::OpKind;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ktir_node::{ActiveCap, KtirNode};
use scratchy_subtile::model_geometry::{with_config_attn_geometry, with_config_head_dim};
use scratchy_subtile::subtile_ir::{EwKind, RopeForm, SubOp, SubtileIR, SubtileNode};
use scratchy_subtile::subtile_ir::{TensorId, TensorRegion};
use scratchy_subtile::superdsc_opspec::{DataFormat, DeviceTileLayout, Df, ItDim};

fn lower_matmul_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    // `(tensor_id, rows, cols)` for an activation that is a SYNTHETIC beyond `ir.tensors` — the
    // prefill lm-head tail's `[1, hidden]` LAST_HIDDEN. `None` for every graph tensor.
    synth_shape: Option<(usize, u32, u32)>,
    _sym_id_base: &mut i64,
    _layout: Option<&BundleLayout>,
    _quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let a = &node.inputs[0];
    let w = &node.inputs[1];
    // Route the ENTRY through TileIR (node_to_single_tile_op), same as SiluMul/RmsNorm/RopeRotate/
    // AttnDecode/Elementwise/SumReduce/ScalarMul.
    //
    // ⛔ IT IS CALLED FOR ITS `Err`, AND ITS DIMS ARE DELIBERATELY UNREAD. This is the one place a
    // malformed `MatmulTile` node is REFUSED before a program is built. The `mb`/`in`/`out` sizes it
    // carries are built straight from `a.region.rows.len` / `a.region.cols.len` /
    // `node.output.region.cols.len` (subtile_tape_to_tile_ir.rs's MatmulTile arm, zero
    // transformation), and `KtirFunc::matmul` reads those same regions itself — so binding them here
    // as `m`/`k`/`n` would be the SECOND derivation of one quantity, which is the defect family this
    // lowering keeps paying for. One derivation, in the builder.
    crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    // ⭐⭐⭐ ONE NODE, ONE PROGRAM — with the KTIR matmul's own tiling: M across the grid, N in
    // column blocks, and the contraction as an accumulating loop. See `KtirFunc::matmul`.
    let mut st = KtirFunc::new(ir);
    st.synth_shape = synth_shape;
    // ⭐⭐⭐ A PROGRAM IS NAMED BY ITS NODE, NOT BY ITS OUTPUT TENSOR.
    //
    // The emulator keys a module's functions BY NAME (`module.get_function(&node.func)`), so a
    // name is an identity, not a label: two distinct programs sharing one is one program.
    //
    // ⛔ AND AN OUTPUT TENSOR DOES NOT IDENTIFY A NODE. The front end splits a wide op into
    // COLUMN CHUNKS that all write the same tensor — MEASURED on granite-3.1-2b, whose logits
    // scale is emitted as `cols=0+8192`, `16384+8192`, `24576+8192`, `32768+8192`, … over one
    // `[.., 49155]` output. Named by that tensor, every chunk was `scalarmul_n1130`, the module
    // kept ONE, and every chunk's launch ran it: the surviving chunk was the ragged tail
    // `cols=49152+3`, so the model's logits were 3 columns of 49155 and the rest stale.
    //
    // `SubtileId` is the dense node index (`subtile_ir.rs:54-63`) — the identity the graph already
    // carries, one per node, which is exactly one per program.
    let name = Arena::global().str(format!("matmul_s{}", node.id.index()));
    // ⭐ ARITY IS THE PRECISION. Two operands is the fp16 contraction; three is W8A8 — activation,
    // packed fp8 weight, and the per-column weight scale the checkpoint ships — which carries the
    // quantize chain the card performs.
    match node.inputs.get(2) {
        Some(wscale) => st.matmul_fp8(a, w, wscale, &node.output),
        None => st.matmul(a, w, &node.output),
    }
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::Matmul);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}
/// Lower a shape-preserving [`SubOp::Elementwise`] node (Add/Mul binary, Silu
/// unary) to ONE pointwise [`SdscOp`]. All operands are `[rows, cols]` (the
/// output shape); `op_func` + arity per the [`EwKind`].
/// f16 ELEMENT budget for one shape-preserving elementwise node's whole-region live set — the same
/// number, and the same reasoning, as [`ktir_n_block`]'s `BLOCK_MN_BUDGET`: a few `[rows, cols]`
/// tiles resident together inside a core's 2 MB LX.
const EW_LX_ELEMS: u32 = 1024 * 1024;

/// How many `[rows, cols]` tiles of one lowering are LIVE AT ONCE — what the budget is divided by.
///
/// ⛔ THE BUDGET IS THE LIVE SET, NOT ONE TILE. Sizing a block so a single tile fits is what
/// produced `ArithMulf: LX capacity exceeded: 2097152 + 1048576 > 2097152` on a `[64, 8192]` block:
/// each tile was 1 MB and three of them were resident. Silu's decomposition is the worst case here
/// (`x`, `neg`, `exp`, the splat `1.0`, the denominator, the result); a binary op holds three.
fn ew_live_tiles(kind: EwKind) -> u32 {
    match kind {
        EwKind::Silu => 6,
        _ => 3,
    }
}

/// How many ROWS of a `cols`-wide region keep `live` tiles inside the LX.
///
/// ⛔ NOT ONE ROW. A row at a time is correct and fits trivially, but it emits `m` copies of every
/// op, and a forward's cost is dominated by PER-OP work: at the m=96 prefill rung that put 3.7 s of
/// a 5.6 s forward outside the GEMMs entirely. Blocking at the widest height that still fits is what
/// keeps both the LX bound and the op count.
fn rows_per_block(cols: u32, live: u32) -> u32 {
    (EW_LX_ELEMS / live.max(1) / cols.max(1)).max(1)
}

/// `tr` narrowed to `h` rows starting `off` rows into its own region.
fn sub_rows(tr: &TensorRegion, off: u32, h: u32) -> TensorRegion {
    TensorRegion {
        tensor: tr.tensor,
        region: scratchy_subtile::subtile_ir::Region {
            rows: scratchy_subtile::subtile_ir::Range::new(tr.region.rows.start + off, h),
            cols: tr.region.cols,
        },
    }
}

/// The node's declared output `[rows, cols]` — the one read of `node.output.region`, so every
/// program states the same fact the same way. See [`KtirNode::out_shape`].

/// The stem a program's name takes for an [`EwKind`] — the node kind the consumer's dispatch reads.
///
/// ⛔ ENUMERATED, NEVER `_`, so a new `EwKind` is an E0004 here rather than a program whose name
/// says nothing about what it computes. The TWO the device has no primitive for are named too:
/// they must reach the consumer as themselves and be refused BY NAME there, not silently take the
/// nearest arm (`gelu` is not quick-gelu, and neither is erf-gelu).
///
/// ⭐ IT WAS THREE, AND `Sub` WAS THE THIRD BY MISTAKE. `OpFunc::Subtract` exists and spells `"sub"`;
/// the consumer now lowers a same-extent subtract as the ordinary pointwise op it is. What is still
/// refused there is a subtract whose right operand BROADCASTS (`[m, 1]`, the LayerNorm
/// mean-centering) — refused by the operand-extent guard, on the real condition, rather than by kind.
/// An [`EwKind`] as the leaf crate's own [`Elementwise`].
///
/// ⛔ ENUMERATED, NEVER `_`, for the same reason [`ew_kind_stem`] is: a new `EwKind` must be an E0004
/// here rather than silently taking the nearest arm. The two the device has no primitive for cross
/// as THEMSELVES, so the refusal names them at the lowering rather than being pre-empted here.
fn ew_kind_program(kind: EwKind) -> ktir_superdsc::ktir_node::Elementwise {
    use ktir_superdsc::ktir_node::Elementwise as E;
    match kind {
        EwKind::Add => E::Add,
        EwKind::Mul => E::Mul,
        EwKind::Sub => E::Sub,
        EwKind::Silu => E::Silu,
        EwKind::Gelu => E::Gelu,
        EwKind::QuickGelu => E::QuickGelu,
        EwKind::GeluErf => E::GeluErf,
    }
}

fn ew_kind_stem(kind: EwKind) -> &'static str {
    match kind {
        EwKind::Add => "add",
        EwKind::Mul => "mul",
        EwKind::Sub => "sub",
        EwKind::Silu => "silu",
        EwKind::Gelu => "gelu",
        EwKind::QuickGelu => "quickgelu",
        EwKind::GeluErf => "geluerf",
    }
}

/// The ROW-AT-A-TIME form of [`lower_elementwise_node`], for a region too wide to hold whole. Rows
/// of an elementwise are independent, so `m` rows are `m` one-row computations, unrolled at emit —
/// the same shape [`KtirFunc::silu_mul`] uses and for the same reason.
fn lower_elementwise_node_rows<F: RopeForm>(
    node: &SubtileNode<F>,
    kind: EwKind,
    name: &'static str,
    mut st: KtirFunc<'_, F>,
) -> Result<EmittedOp, SuperDscError> {
    let cols = node.output.region.cols.len;
    let blk = rows_per_block(cols, ew_live_tiles(kind));
    let dims = vec![i64::from(blk), i64::from(cols)];
    let mut off = 0u32;
    while off < node.output.region.rows.len {
        let h = blk.min(node.output.region.rows.len - off);
        let dims = if h == blk {
            dims.clone()
        } else {
            vec![i64::from(h), i64::from(cols)]
        };
        // Each operand carries its OWN region offset, so the block is relative to that operand.
        let y = match kind {
            EwKind::Add | EwKind::Mul | EwKind::Sub => {
                let a = st.load_region(&sub_rows(&node.inputs[0], off, h));
                let b = st.load_region(&sub_rows(&node.inputs[1], off, h));
                let op = match kind {
                    EwKind::Add => OpKind::ArithAddf,
                    EwKind::Mul => OpKind::ArithMulf,
                    _ => OpKind::ArithSubf,
                };
                st.binop(op, a, b, dims.clone())
            }
            EwKind::Silu => {
                let x = st.load_region(&sub_rows(&node.inputs[0], off, h));
                let neg = st.negate(x, dims.clone());
                let e = st.unop(OpKind::MathExp, neg, dims.clone());
                let one = st.splat_one(dims.clone());
                let den = st.binop(OpKind::ArithAddf, one, e, dims.clone());
                st.binop(OpKind::ArithDivf, x, den, dims.clone())
            }
            ref other => {
                return Err(SuperDscError(format!(
                    "no KTIR lowering for elementwise {other:?} on t{}",
                    node.output.tensor.index() as u32
                )));
            }
        };
        st.store_region(y, &sub_rows(&node.output, off, h));
        off += h;
    }
    let k = st.finish_shaped(
        name,
        ktir_superdsc::ktir_node::Program::Elementwise(ew_kind_program(kind)),
    );
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(e)
}

fn lower_elementwise_node<F: RopeForm>(
    node: &SubtileNode<F>,
    // The graph the node belongs to — the shapes its program's views state.
    ir: &SubtileIR<F>,
    kind: EwKind,
    _sym_id_base: &mut i64,
    _layout: Option<&BundleLayout>,
) -> Result<EmittedOp, SuperDscError> {
    // ⭐ ONE NODE, ONE PROGRAM — WHOLE TILES WHERE THEY FIT, ROWS WHERE THEY DO NOT.
    //
    // ⛔ AND THE CONDITION MATTERS BOTH WAYS. Row-blocking unconditionally emits `m` copies of every
    // op: at the m=96 prefill rung that took the generated model source from ~101 MB to 149.7 MB,
    // with rustc single-threaded over it. Blocking NOTHING is the other failure — a `[96, 8192]`
    // f16 operand is 1,572,864 bytes against a 2,097,152-byte LX, so the MLP's elementwise died with
    // `TensorSplat: LX capacity exceeded: 1572864 + 1572864 > 2097152`. So block exactly when the
    // whole region cannot fit, which for `m == 1` is never: decode emits the same `[1, cols]` tiles
    // it always did, and only the wide prefill rows pay the op count.
    //
    // The budget is [`ktir_n_block`]'s, for the same reason it gives there: a live set of a few
    // `[rows, cols]` f16 tiles (two operands plus the result, plus silu's temporaries) inside a
    // 2 MB LX.
    let mut st = KtirFunc::new(ir);
    // ⭐⭐ THE NAME CARRIES THE KIND, NOT JUST "AN ELEMENTWISE". A program's name is the node's
    // identity for both consumers, and the consumer's dispatch reads the node KIND off it — but
    // `Elementwise` is not one kind: `add`, `multiply`, `sub`, `silu` and `gelu` are five DIFFERENT
    // device primitives, and quick-gelu/erf-gelu are two the device does not have at all. A single
    // `ew_` stem would leave the consumer to recover which one from the ops, which is the op-graph
    // recognition this path exists to avoid. The `EwKind` is what the front end already decided, so
    // it goes in the name it already writes.
    let name = Arena::global().str(format!("{}_s{}", ew_kind_stem(kind), node.id.index()));
    let (rows, cols) = (node.output.region.rows.len, node.output.region.cols.len);
    if rows > 1
        && u64::from(rows) * u64::from(cols) * u64::from(ew_live_tiles(kind))
            > u64::from(EW_LX_ELEMS)
    {
        return lower_elementwise_node_rows(node, kind, name, st);
    }
    let dims = vec![i64::from(rows), i64::from(cols)];
    let y = match kind {
        EwKind::Add => {
            let a = st.load_region(&node.inputs[0]);
            let b = st.load_region(&node.inputs[1]);
            st.binop(OpKind::ArithAddf, a, b, dims)
        }
        EwKind::Mul => {
            let a = st.load_region(&node.inputs[0]);
            let b = st.load_region(&node.inputs[1]);
            st.binop(OpKind::ArithMulf, a, b, dims)
        }
        EwKind::Sub => {
            let a = st.load_region(&node.inputs[0]);
            let b = st.load_region(&node.inputs[1]);
            st.binop(OpKind::ArithSubf, a, b, dims)
        }
        // silu(x) = x / (1 + exp(-x)) — the same decomposition the fused gate-up arm performs.
        EwKind::Silu => {
            let x = st.load_region(&node.inputs[0]);
            let neg = st.negate(x, dims.clone());
            let e = st.unop(OpKind::MathExp, neg, dims.clone());
            let one = st.splat_one(dims.clone());
            let den = st.binop(OpKind::ArithAddf, one, e, dims.clone());
            st.binop(OpKind::ArithDivf, x, den, dims)
        }
        ref other => {
            return Err(SuperDscError(format!(
                "no KTIR lowering for elementwise {other:?} on t{}",
                node.output.tensor.index() as u32
            )));
        }
    };
    st.store_region(y, &node.output);
    let k = st.finish_shaped(
        name,
        ktir_superdsc::ktir_node::Program::Elementwise(ew_kind_program(kind)),
    );
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(e)
}

fn lower_silumul_node<F: RopeForm>(
    node: &SubtileNode<F>,
    // The graph the node belongs to — the shapes its program's views state.
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
    _layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "SiluMul t{} expects 2 inputs (gate, up), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("silumul_s{}", node.id.index()));
    st.silu_mul(&node.inputs[0], &node.inputs[1], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::SiluMul);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a whole [`SubOp::RmsNorm`] by DECOMPOSING into the 6-op sequence IBM's
/// `torch_spyre` uses (`decompositions.py:409 spyre_rms_norm`):
///   sq=x·x → mean=mean(sq) → meps=mean+eps → inv=rsqrt(meps) → tmp=x·inv → y=tmp·gamma
/// `inputs[0]`=x `[m,cols]`, `inputs[1]`=gamma `[1,cols]`, `eps` is the attr. The
/// reduce produces `[m, one-stick]`; `inv` is broadcast over `out` in the scale
/// step (RedStick, the on-card-proven `alpha_=0` broadcast read); `gamma` is
/// broadcast over rows (`mb`=-1); `eps` is a `[1,1]` scalar INPUT (its value is a
/// runtime const, provisioned like a weight — #51). Synthetic intermediates are
/// allocated by the coloring pass (like `lower_silumul_node`'s `<out>_silu`).
fn lower_rmsnorm_node<F: RopeForm>(
    node: &SubtileNode<F>,
    // The graph the node belongs to — the shapes its program's views state.
    ir: &SubtileIR<F>,
    eps: f32,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "RmsNorm t{} expects 2 inputs (x, gamma), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    // ⛔ THE DIVISOR IS AN IMMEDIATE, NOT A REGISTRY SLOT. `subtile→superdsc` binds `1/cols` at the
    // reserved `RMS_INVCOLS_TID` as a `[1, stick]` row and registers ONLY the epsilon, so pushing the
    // column count into `scalarmul_scales` too added a slot per rmsnorm node and moved the tid of
    // every constant after it. The ported body emits that path's own descriptors and reads
    // `RMS_INVCOLS_TID` itself; this value exists only for the KTIR the emulator interprets, where an
    // immediate costs nothing and is invisible to the device's constant surface.
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("rmsnorm_s{}", node.id.index()));
    st.rmsnorm(&node.inputs[0], &node.inputs[1], &node.output, eps);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RmsNorm);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RopeRotate`] / the rotate of [`SubOp::RopeAppend`] to in-bundle
/// SuperDSC ops via the PERMUTATION-MATMUL form (NeoX). The 32-wide rotate-half
/// would violate the 64-fp16-stick constraint, so instead `rot = x·P` where `P` is a
/// fixed `[hd,hd]` sign-permutation (`P[i+half,i]=-1` for i<half, `P[i-half,i]=+1`
/// for i≥half) — a 64-stick-aligned matmul. Then `out = x·cos + rot·sin`. `inputs` =
/// (x, cos, sin); `x` `[mq, heads·hd]` (row-major). One `mb=1 [1,hd]` block is emitted
/// per (row `r`, head `h`) at `rope_prefill_block_offset(r,h,total,hd)`; cos/sin are the
/// worker's `[mq, total]` per-position head-tiled table, read at `rope_prefill_cos_offset(r,total)`
/// (row `r`'s head-0 slice serves every head). `P` is the resident `t{ROPE_P_TID}` weight.
/// Decode (mq=1) reduces to the original per-head loop, byte-identical.
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
fn lower_rope_node<F: RopeForm, const HD: u32>(
    node: &SubtileNode<F>,
    // The graph the node belongs to — the shapes its program's views state.
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
    _layout: Option<&BundleLayout>,
    // ⭐ THE ROW KIND, THREADED. It used to be read from a thread-local `Cell<bool>` set once per
    // bundle, so this function could not be reasoned about locally and no caller was forced to state
    // which kind it was emitting. That ambient carrier is deleted; see `sdsc_abstract::QueryRows`.
    //
    // It matters HERE specifically: at `hd == stick` the collapsed form below is chosen and it is the
    // ONLY branch in this function that knows a row might be an independent sequence. At hd > stick
    // (head_dim 128) `head_major_collapse_valid` is FALSE, the collapse is skipped, and a decode batch
    // falls into the path commented "PREFILL (mq>1) hd>stick: SLAB rope" — whose rows are consecutive
    // positions of ONE sequence. That is correct today only because the worker stages cos/sin per
    // REQUEST and the two layouts agree by construction; nothing in a type says so.
    _rows_are_requests: bool,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    // ⭐ THE FIRST THREE ARE THE ROTATION: x, cos, sin. A `RopeAppend` node carries more — the V
    // tensor and the KV-cache destinations — because its cache write flows through GRAPH EDGES
    // rather than through this op: the new K/V are results the host threads, so for this emitter an
    // append IS a rotate. Fewer than three would be a malformed node and says so.
    if node.inputs.len() < 3 {
        return Err(SuperDscError(format!(
            "rope t{} needs at least 3 inputs (x, cos, sin), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let total = node.output.region.cols.len;
    if HD == 0 || !total.is_multiple_of(HD) {
        return Err(SuperDscError(format!(
            "rope t{}: {total} cols is not a whole number of {HD}-wide heads",
            node.output.tensor.index() as u32
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("rope_s{}", node.id.index()));
    st.rope(
        node.inputs[0].tensor,
        node.inputs[1].tensor,
        node.inputs[2].tensor,
        node.output.tensor,
        total,
        HD,
        node.output.region.rows.len,
        // The cos/sin tables are pre-tiled to the rope's full width by the host, so position `ri`'s
        // row starts at `ri · tbl_cols` — the table's OWN column count, not the head dim.
        node.inputs[1].region.cols.len,
    );
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::Rope);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::AttnDecode`] to ONE KTIR program stating the whole node — the single statement
/// BOTH consumers read: `-Fspyre-emu` interprets it, `-Fspyre-hw` lowers it to SuperDSC through
/// `ktir_to_superdsc`. There is no second attention emitter; the `*_sdsc` sibling this doc used to
/// point at was a side path to SuperDSC and is deleted.
///
/// Batched over q-heads (BatchMatmul, batch=`num_q_heads`), reading the resident transposed-K +
/// replicated K/V cache (seg2, worker-filled) over the full `cap` masked by `t{ATTN_MASK_TID}`.
/// Decode (mq=1): per head `scores[1,cap] = Q·Kᵀ·scale + mask`, softmax, `out=probs·V`.
/// Viewed as `[nqh, cap]` for the ew/softmax ops (mb=nqh, out=cap; `[nqh,1,cap]`≡
/// `[nqh,cap]` bytes). Softmax reduce-outputs are one stick (like rmsnorm's mean).
///
/// ⭐⭐ `HD` IS A CONST GENERIC — same reason as [`lower_rope_node`]. The head dim parameterises the device
/// layout (slabs = head_dim/lanes; the head-major collapse is byte-identical ONLY at head_dim == lanes),
/// so every decision it drives must be a branch on a const, not on a value. The value becomes a const at
/// ONE dispatch in `lower_one_node`.
fn lower_attn_node<F: RopeForm, const NQH: u32, const NKVH: u32, const HD: u32>(
    node: &SubtileNode<F>,
    // The graph the node belongs to — the shapes its program's views state.
    ir: &SubtileIR<F>,
    // The geometry witness the door minted, carrying the GQA divisibility proof. Every head count
    // and head dim this function uses is read off it, so none of them is a value the node handed
    // over and there is nothing here to check against the consts.
    geom: scratchy_subtile::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
    cap: u32,
    active_cap: ActiveCap,
    // TRUE when this bundle's query rows are separate requests (a batched decode) rather than
    // consecutive positions of one sequence (a prefill chunk). Only the KV writes care: a prompt's
    // rows share a page and differ in slot, requests differ in both.
    rows_are_requests: bool,
    _sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    // ⭐ THE GEOMETRY IS THE COMPILER'S HERE, NOT THE NODE'S. `lower_one_node`'s door instantiated
    // this function FROM the node's own `ModelAttnGeometry`, so there is no second copy of the head
    // counts in scope to disagree with the consts. What the node is still asked for is its scale,
    // which the geometry does not carry — and which this program STATES, as an immediate.
    let scale_val = match &node.op {
        SubOp::AttnDecode { scale, .. } => *scale,
        _ => {
            return Err(SuperDscError(
                "lower_attn_node: node is not AttnDecode".into(),
            ));
        }
    };
    let (_nqh, _nkvh, hd) = (geom.nqh(), geom.nkvh(), geom.hd());
    if node.inputs.len() < 5 {
        return Err(SuperDscError(format!(
            "AttnDecode t{}: expects 5 inputs [q, prefix_k, prefix_v, new_k, new_v], found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let stick = Fp16::ELEMS_PER_STICK; // 64
    // ── PAGED-ATTENTION COMPUTE EXTENT ── see the header doc above for `active_cap` semantics.
    let active_cap: u32 = active_cap.resolve(cap, stick);
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
            return Err(SuperDscError(format!(
                "AttnDecode t{}: resident K/V cache [cap={cap}, hd={hd}] physical span {span} B \
                 exceeds the {MAX_SPAN_BYTES} B hardware addressing limit. {split_report}.",
                node.output.tensor.index() as u32,
            )));
        }
    }
    // GENERAL-mq: mq = the chunk's query-row count (1 = decode; >1 = prefill / chunked-prefill). ONE
    // algorithm below for any mq — see ir::bridge::tiled_op_sdsc_op::attn's module doc (the port of
    // torch-spyre's spyre__sdpa_overrideable).
    let attn_tile_op =
        crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    let mq = attn_tile_op
        .dims
        .iter()
        .find(|d| d.name == "mb")
        .map(|d| d.size)
        .unwrap_or(1);
    // ⛔ ONE SOURCE FOR THE PADDED ROW COUNT AND THE ROW LAWS: `attn_bundle_rows` is the parse
    // boundary from the bundle's runtime width to the pad law WITH the geometry's consts in scope,
    // and the SAME carrier rides into `assemble_attn`, so the two cannot disagree. A decode width
    // (one row, or rows that are requests) must be a baked ladder rung there — the pad is
    // `Rung<MQ>`'s compile-time arithmetic and every row extent is `RungRowLaws`' const
    // (`MaskRows = NQH*MQ` by the compiler), and an unlisted width is this loud bake error, the
    // same boundary discipline as the geometry door in `lower_one_node`. A prefill chunk's width is
    // a runtime quantity and takes the runtime arm of the same law.
    // `PaddedRows` below, because everything THIS function does with the value is a ROW question — the
    // staging tensors' `[mq_pad, nkvh·hd]` row extent and the row sweeps over them; the slot/score-width
    // roles are `assemble_attn`'s, drawn there from the same carrier.
    let _bundle_rows = scratchy_subtile::sdsc_abstract::attn_bundle_rows(
        geom,
        mq,
        rows_are_requests,
    )
    .ok_or_else(|| {
        SuperDscError(format!(
            "AttnDecode t{}: a decode bundle at {mq} query rows — not a width the decode ladder \
                 bakes (1, or PagedKvPool::BATCH_RUNGS). The decode width is a const of the bundle \
                 (`Rung<MQ>`); bake a listed rung, or grow the ladder and its dispatch together.",
            node.output.tensor.index() as u32
        ))
    })?;
    // attention_multiplier (config) — see the ORIGINAL header doc: NO 1/sqrt(hd) recompute.
    //
    // ⭐ RESOLVED AS THE REGISTRY-DESYNC CHECK, not to be threaded — the same shape as
    // `_sqrt_scale_idx` below. The descriptor's slot is resolved by the ported body from
    // the multiplier its caller states at the door; what this proves is that the value the program
    // states inline is also REGISTERED, so the two consumers cannot disagree about which constants the
    // bundle has. (`attn_at` proves the third leg — that the caller's value IS the program's.)
    let _scale_idx = layout
        .and_then(|l| l.scalarmul_scales.iter().position(|s| s.to_bits() == scale_val.to_bits()))
        .ok_or_else(|| {
            SuperDscError(format!(
                "AttnDecode t{}: scale {scale_val} absent from BundleLayout.scalarmul_scales (registry desync)",
                node.output.tensor.index() as u32
            ))
        })?;
    let sqrt_scale_val = scale_val.sqrt();
    let _sqrt_scale_idx = layout
        .and_then(|l| l.scalarmul_scales.iter().position(|s| s.to_bits() == sqrt_scale_val.to_bits()))
        .ok_or_else(|| {
            SuperDscError(format!(
                "AttnDecode t{}: √scale {sqrt_scale_val} absent from scalarmul_scales (registry desync)",
                node.output.tensor.index() as u32
            ))
        })?;
    // [mq, nqh·hd] — the identity; every scratch name below is a rendering of it.

    // ⭐⭐⭐ ONE NODE, ONE PROGRAM. The SuperDSC decomposition of attention — zero the padding rows,
    // rope, per-head scores, the online softmax, then the KV cache write — existed because dxp
    // schedules at descriptor grain. KTIR states the whole node as one func, with the head split as
    // its grid and the query rows as its loop.
    //
    // ⛔ AND THE DEVICE-SIDE CACHE WRITE HAS NO KTIR COUNTERPART. It appended this step's K/V into
    // the resident cache at a baked slot, which is the addressing the launch's `KvShifts` describe.
    // The KTIR path threads KV through the graph's OWN tensors — the prefix cache is a source and
    // the new K/V a result — so there is no slot to write to here.
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("attn_s{}", node.id.index()));
    // The prefix cache written THIS forward spans the full structural capacity while only
    // `decode_position` of its rows are valid at run time, so it — and only it — carries the
    // runtime length mask. The new-token segment is always valid.
    let mask_prefix = node.inputs.len() >= 5
        && matches!(
            node.op,
            SubOp::AttnDecode {
                producer: scratchy_subtile::subtile_ir::KvCacheProducer::SameForwardRopeAppend { .. },
                ..
            }
        );
    st.attn(
        &node.inputs[0],
        &node.inputs[1..],
        &node.output,
        geom.nqh(),
        geom.hd(),
        geom.gqa(),
        // The VALUE — the program states it inline, exactly as `subtile→superdsc` does. The registry
        // slot resolved above is still required (the descriptor reads it); the consumer resolves it
        // from the multiplier its own caller states, and checks that against this immediate.
        scale_val,
        mask_prefix,
        // ⭐ THE RUNG. `active_cap` resolved above is the swept KV extent this bundle was baked
        // for; sweeping the tensor's full capacity instead is what the ladder exists to avoid.
        active_cap,
    );
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::Attn);
    // ⛔⛔⛔ AND NOTHING RIDES BESIDE IT. `AttnFacts` used to be stapled here: eleven fields computed
    // off this node and read by the consumer INSTEAD OF THE PROGRAM, so information flowed
    // `SubtileIR → AttnFacts → SuperDSC` straight past the IR. Six tensor identities and the resident
    // capacity are things this program STATES — its parameters are `q`, `out`, the mask, then each
    // segment's K and V view, and the resident cache's view states its own row count — so the
    // consumer reads them there (`attn_operands`). The four that are facts of the MODEL and the
    // BUNDLE (`geom`, `scale`, `active_cap`, `rows_are_requests`) are stated by the CALLER at the
    // geometry door, which is main's own channel for them; see
    // [`crate::ktir_superdsc_door::BundleAttnParams`] and [`attn_bundle_params`].
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// The stable name of a HOST-ROUTED (data-movement) SubOp, for the host-routed
/// reconnaissance set. These ops never become SuperDSC tiles — the host threads
/// activations across them — so this is a label, not a lowering.
fn host_glue_kind<F: RopeForm>(op: &SubOp<F>) -> &'static str {
    match op {
        SubOp::RopeRotate { .. } => "RopeRotate",
        SubOp::RopeAppend { .. } => "RopeAppend",
        SubOp::AttnDecode { .. } => "AttnDecode",
        SubOp::RmsNormReduce { .. } => "RmsNormReduce",
        SubOp::RmsNormApply { .. } => "RmsNormApply",
        // The pure-compute ops are never passed here.
        _ => "?",
    }
}

/// Lower one [`SubOp::ScalarMul`] node (granite embedding/residual/attn/logits multipliers) to a pointwise
/// `mul` by its scale constant. `out[i,j] = x[i,j] · scale`: operand-0 is `x` (full), operand-1 is the
/// `[1,1]` scale const [`scalarmul_scale_tid`] broadcast on BOTH dims — exactly the ATTN_SCALE mechanism
/// (a bound scalar × tensor, proven on-card). The split/address/fold are the standard pointwise structure
/// (Kani-proven via CoreSplit/dev_off/OpDims); the scale VALUE is the shared SEN169 leaf the worker binds.
/// NO weight-fold, NO host-route.
fn lower_scalarmul_node<F: RopeForm>(
    node: &SubtileNode<F>,
    // The graph the node belongs to — the shapes its program's views state.
    ir: &SubtileIR<F>,
    scale: f32,
    _sym_id_base: &mut i64,
) -> Result<EmittedOp, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "ScalarMul t{} expects 1 input, found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("scalarmul_s{}", node.id.index()));
    let (rows, cols) = (node.output.region.rows.len, node.output.region.cols.len);
    // Whole tiles where they fit, rows where they do not — the same bound, and the same reason, as
    // `lower_elementwise_node`: granite scales an `[m, intermediate]` region, and at the m=96
    // prefill rung `[96, 8192]` f16 is 1,572,864 bytes against a 2,097,152-byte LX.
    let by_row = rows > 1 && u64::from(rows) * u64::from(cols) * 3 > u64::from(EW_LX_ELEMS);
    let dims = if by_row {
        vec![1, i64::from(cols)]
    } else {
        vec![i64::from(rows), i64::from(cols)]
    };
    if by_row {
        let blk = rows_per_block(cols, 3);
        let mut off = 0u32;
        while off < rows {
            let h = blk.min(rows - off);
            let dims = vec![i64::from(h), i64::from(cols)];
            let x = st.load_region(&sub_rows(&node.inputs[0], off, h));
            let sc = st.splat(f64::from(scale), dims.clone());
            let y = st.binop(OpKind::ArithMulf, x, sc, dims);
            st.store_region(y, &sub_rows(&node.output, off, h));
            off += h;
        }
    } else {
        let x = st.load_region(&node.inputs[0]);
        let sc = st.splat(f64::from(scale), dims.clone());
        let y = st.binop(OpKind::ArithMulf, x, sc, dims);
        st.store_region(y, &node.output);
    }
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::ScalarMul);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(e)
}

/// Narrow a node to its FIRST row: the OUTPUT and the leading (activation) input keep their columns but
/// contract to one row. Used by the m>1 prefill lm-head tail, whose activation is a `[1, ·]` slice
/// written at offset 0 by the ops above it, so the whole tail lowers through the SAME `lower_*_node` the
/// PROVEN m=1 decode path uses — no parallel emitter for the folded form.
///
/// ONLY `inputs[0]` is contracted. A matmul's `inputs[1]` is the WEIGHT `[k, n]`, whose `rows` is the
/// REDUCTION extent K, not the query count — narrowing it to one row claims K=1 and fails
/// `lower_matmul_node`'s `W rows != A cols (K)` guard. `inputs[2]` (the fp8 per-channel `w_scale`) is
/// likewise not row-indexed by M. Both are M-independent and pass through untouched.
fn node_at_one_row<F: RopeForm>(node: &SubtileNode<F>) -> SubtileNode<F> {
    let one_row = |tr: &scratchy_subtile::subtile_ir::TensorRegion| {
        scratchy_subtile::subtile_ir::TensorRegion {
            tensor: tr.tensor,
            region: scratchy_subtile::subtile_ir::Region {
                rows: scratchy_subtile::subtile_ir::Range::new(tr.region.rows.start, 1),
                cols: tr.region.cols,
            },
        }
    };
    let mut inputs = node.inputs.clone();
    if let Some(a) = inputs.first_mut() {
        *a = one_row(a);
    }
    SubtileNode {
        id: node.id,
        op: node.op,
        inputs,
        output: one_row(&node.output),
    }
}

/// Lower the m>1 PREFILL lm-head matmul as the m=1 tail it really is: the SAME node, re-lowered
/// with its activation sliced to the LAST prompt row. Only that row's logits are ever read, so
/// running the vocab-wide matmul over all `mq` rows is `mq`× the work for one row of answer.
///
/// ⛔⛔⛔ THE SLICE IS NOT THE WHOLE TAIL, AND BELIEVING IT WAS COST THE FIRST GENERATED TOKEN ON
/// EVERY PROMPT. This body used to be four lines: set the activation region's rows to
/// `[mq-1, 1)` and re-lower, on the reasoning that "a KTIR view STATES its start address and a tile
/// window states its row, so row `mq-1` of `[mq, hidden]` is an address this target can simply name",
/// and that main's `hidden/64`-copy materialization was therefore a workaround for a limit that is not
/// KTIR's.
///
/// The KTIR does name it — [`KtirFunc::matmul`] puts the row on its activation access tile as an
/// `arith.constant`. The CONSUMER cannot carry it: `lower_ktir_to_superdsc::regions` discarded the row
/// corner outright, and every SuperDSC operand spelling (`rb(name, rows, cols)`) names a tensor rather
/// than an offset into one, which is exactly WHY main materializes. So the tail read the buffer base.
/// MEASURED on card, granite-3.1-2b fp8, prompt `hi` (14 tokens, m=15 rung, `PREFILL_PATH` line
/// identical to main's): ours `yun! How can`, main `Hello! How can` — and still wrong on `hi there`,
/// which fills the rung exactly (`m_used=15 prefill_m=15`), so it was never the pad row it resembled.
///
/// So this is main's `lower_prefill_lm_head_at_m1` again, both halves: extract row
/// `selector_lastrow_col(mq)` into the `[1, hidden]` synthetic [`LAST_HIDDEN_TID`], then re-lower the
/// SAME node at `m=1` reading that synthetic at row 0. The extraction is its own KTIR program, because
/// ONE NODE ONE PROGRAM is per NODE KIND and this is two kinds of work; its consumer arm is
/// `lower_ktir_to_superdsc::lmlast`, which is main's copy loop verbatim.
///
/// The re-lowered matmul goes through [`lower_matmul_node`], NOT a hand-rolled emit: that path owns
/// the N-column blocking a vocab-wide output needs and (for an arity-3 weight) the whole per-token
/// fp8 activation-quantize chain, which a hand emit would silently skip and feed the fp8 kernel raw
/// fp16.
fn lower_prefill_lm_head_at_m1<F: RopeForm>(
    node: &SubtileNode<F>,
    // The graph the node belongs to — the shapes its program's views state.
    ir: &SubtileIR<F>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    quantized: &mut std::collections::HashSet<String>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let a = node.inputs.first().ok_or_else(|| {
        SuperDscError(format!(
            "prefill lm_head n{}: MatmulTile with no A operand",
            node.output.tensor.index()
        ))
    })?;
    let (mq, hidden) = (a.region.rows.len, a.region.cols.len);
    let stk = crate::lower_subtile_tape_to_superdsc::Fp16::ELEMS_PER_STICK;
    if hidden % stk != 0 {
        return Err(SuperDscError(format!(
            "prefill lm_head n{}: hidden={hidden} is not a whole {stk}-fp16 stick, so the last prompt \
             row is not a run of whole stick-groups — the per-stick extraction cannot address it",
            node.output.tensor.index()
        )));
    }
    // The row to extract, from the SSOT the Kani proof `selector_lastrow_picks_last_row` pins.
    let row = scratchy_subtile::sdsc_abstract::selector_lastrow_col(mq as usize) as u32;
    let last_hidden = TensorId::from_index(LAST_HIDDEN_TID as usize);
    // ── Half 1: the extraction, as its own program. ──
    let mut st = KtirFunc::new(ir);
    // ⛔ `row`, NOT `a.region.rows.start + row` — main's address is `j·mq + row` over a buffer it takes
    // to be exactly `[mq, hidden]`, so the two spellings agree only at `rows.start == 0` and main's is
    // the authority. They agree HERE by construction: the `row >= mq` refusal below would have failed
    // the build otherwise, and it does not.
    st.last_row_extract(a.tensor, last_hidden, mq, row, hidden);
    let xname = Arena::global().str(format!("lmlast_s{}", node.id.index()));
    let mut extract = EmittedOp::bare(xname.to_string());
    let mut k = st.finish_shaped(xname, ktir_superdsc::ktir_node::Program::LmLast);
    // ⭐ THE NODE THESE COPIES BELONG TO. They write the reserved staging, but main names them
    // `lmlast{j}_o{node output tid}` — the tail's own matmul suffix — so the whole tail's descriptors
    // read as one node's. The program cannot state it (it never addresses the logits buffer), so the
    // producer that built BOTH halves does. See `KtirNode::node_out_tid`.
    k.node_out_tid = Some(ktir_superdsc::ktir_node::BufferId::new(
        node.output.tensor.index() as u32,
    ));
    extract.ktir = Some(k);
    // ── Half 2: the SAME node at m=1, reading the synthetic at row 0. ──
    let mut at_m1 = node_at_one_row(node);
    at_m1.inputs[0] = scratchy_subtile::subtile_ir::TensorRegion {
        tensor: last_hidden,
        region: scratchy_subtile::subtile_ir::Region {
            rows: scratchy_subtile::subtile_ir::Range::new(0, 1),
            cols: scratchy_subtile::subtile_ir::Range::new(0, hidden),
        },
    };
    let mut ops = vec![extract];
    ops.extend(lower_matmul_node(
        &at_m1,
        ir,
        Some((last_hidden.index(), 1, hidden)),
        sym_id_base,
        layout,
        quantized,
    )?);
    Ok(ops)
}

/// Outcome of lowering ONE SubtileIR node. Shared by the UNROLLED
/// [`lower_graph_to_ktir`] walk and the RE-ROLLED tape-driven walk so the
/// per-op `match` lives exactly once (the reroll just changes WHICH nodes are
/// walked + how many times the body runs, not how each op lowers).
pub(crate) enum NodeLowering {
    Ops(Vec<EmittedOp>),
    /// An op kind not yet lowerable (collected into the hard-error worklist).
    Unhandled(String),
    /// A recognized data-movement op routed host-side (not a SuperDSC tile).
    HostRouted(&'static str),
}

/// ⭐⭐⭐ THE BUNDLE'S ATTENTION PARAMETERS, READ WHERE MAIN READ THEM.
///
/// `ibm/main`'s `lower_one_node` lowered each node during the tape walk, so `SubOp::AttnDecode`'s
/// `geom` and `scale` were in its hand and `active_cap` / `rows_are_requests` were its own walk
/// parameters. This split lowers KTIR → SuperDSC one pass later, per BUNDLE, so the same four facts
/// are read HERE — off the graph's own `AttnDecode` nodes and off this walk's parameters — and travel
/// to the door as [`crate::ktir_superdsc_door::BundleAttnParams`], an argument of the call.
///
/// ⛔ THE MODEL FACTS MUST BE THE MODEL'S, SO A DISAGREEMENT IS AN ERROR AND NOT A CHOICE. One
/// `#[forward]` expansion is one model, so every `AttnDecode` node in one graph carries the same
/// geometry and the same multiplier. If two ever differed, one value per bundle could not describe
/// both, and picking the first would silently give one layer another layer's registry slot — so this
/// refuses instead, naming both.
///
/// `None` when the graph has no attention node: there is then nothing for the four to be facts of, and
/// an attention program arriving at the door without them is that door's own build error.
pub(crate) fn attn_bundle_params<F: RopeForm>(
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> Result<Option<crate::ktir_superdsc_door::BundleAttnParams>, SuperDscError> {
    let mut found: Option<(ktir_superdsc::head_counts::ModelAttnGeometry, u32)> = None;
    for n in &ir.nodes {
        let SubOp::AttnDecode { geom, .. } = &n.op else {
            continue;
        };
        let t = n.output.tensor.index() as u32;
        match found {
            None => found = Some((*geom, t)),
            Some((g0, t0)) => {
                if g0 != *geom {
                    return Err(SuperDscError(format!(
                        "AttnDecode t{t0} declares geometry ({g0}) while AttnDecode t{t} declares \
                         ({geom}). One bundle is one model, and the geometry door the lowering \
                         crosses takes ONE geometry for the whole bundle; a graph carrying two would \
                         give one layer the other's."
                    )));
                }
            }
        }
    }
    Ok(found.map(|(geom, _)| {
        crate::ktir_superdsc_door::BundleAttnParams {
            geom,
            rows_are_requests,
        }
    }))
}

/// The rotary lowering, waiting for its head dim to become a const — the consumer side of
/// [`with_config_head_dim`]. It exists so the arms of that door can be GENERATED: a callback trait
/// takes one impl and any number of arms, where a `match` written here would need one line per
/// head dim, written by a human, and therefore a list of head dims in a source file.
struct LowerRope<'a, F: RopeForm> {
    node: &'a SubtileNode<F>,
    /// The graph the node belongs to — the shapes its program's views state.
    ir: &'a SubtileIR<F>,
    sym_id_base: &'a mut i64,
    layout: Option<&'a BundleLayout>,
    rows_are_requests: bool,
}

impl<F: RopeForm> scratchy_subtile::model_geometry::OnHeadDim for LowerRope<'_, F> {
    type Out = Result<Vec<EmittedOp>, SuperDscError>;
    fn on_head_dim<const HD: u32>(self) -> Self::Out {
        lower_rope_node::<F, HD>(
            self.node,
            self.ir,
            self.sym_id_base,
            self.layout,
            self.rows_are_requests,
        )
    }
}

/// The attention lowering, waiting for its head geometry to become consts — the consumer side of
/// [`with_config_attn_geometry`], and the same reason as [`LowerRope`].
struct LowerAttn<'a, F: RopeForm> {
    node: &'a SubtileNode<F>,
    /// The graph the node belongs to — the shapes its program's views state.
    ir: &'a SubtileIR<F>,
    cap: u32,
    active_cap: ActiveCap,
    rows_are_requests: bool,
    sym_id_base: &'a mut i64,
    layout: Option<&'a BundleLayout>,
}

impl<F: RopeForm> scratchy_subtile::model_geometry::OnAttnGeometry for LowerAttn<'_, F> {
    type Out = Result<Vec<EmittedOp>, SuperDscError>;
    fn on_geometry<const NQH: u32, const NKVH: u32, const HD: u32>(
        self,
        geom: scratchy_subtile::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
    ) -> Self::Out {
        lower_attn_node::<F, NQH, NKVH, HD>(
            self.node,
            self.ir,
            geom,
            self.cap,
            self.active_cap,
            self.rows_are_requests,
            self.sym_id_base,
            self.layout,
        )
    }
}

/// Lower ONE [`SubtileNode`] to its SuperDSC op(s) — the single source of the
/// per-`SubOp` match. `ir` is needed only for `AttnDecode`'s cache-capacity lookup.
/// ⭐⭐⭐ ONE NODE, ONE PROGRAM.
///
/// The SuperDSC lowering decomposes a node into several descriptor ops, because dxp schedules at
/// that grain. KTIR does not: a node's whole computation is one `func`, with its inputs loaded from
/// HBM and its output stored back, and the work division stated INSIDE it as the grid and its
/// loops. So this walk yields one [`EmittedOp`] per node, carrying that program.
pub(crate) fn lower_one_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    // Swept KV extent for AttnDecode (paged-attn ladder rung); FULL ⇒ full cap. See `lower_attn_node`.
    active_cap: ActiveCap,
    // See `lower_attn_node`: whether this bundle's rows are separate requests.
    rows_are_requests: bool,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
    // Threaded to `lower_matmul_node` so fp8 activation quantization is shared across matmuls (see there).
    quantized: &mut std::collections::HashSet<String>,
) -> NodeLowering {
    use NodeLowering::{HostRouted, Ops, Unhandled};
    // ── PREFILL (m>1) LM-HEAD TAIL, FOLDED TO m=1 — the prefill bundle produces the FIRST generated
    // token's logits itself, so TTFT is ONE forward, not two. ──
    // The vocab-wide (≈49159-col) lm_head cannot run at m>1: it ALWAYS time-tiles, and per-row (m>1)
    // time-tiling is unimplemented (design-risk-4 Err). Only the LAST prompt token's logits are ever
    // read, so the tail runs at m=1 over `last_hidden[1, hidden]` = row `selector_lastrow_col(mq)` of the
    // final-norm output — which is exactly the shape the PROVEN decode path lowers. See
    // [`LAST_HIDDEN_TID`] for why the extraction is per-stick copies rather than a one-hot matmul.
    //
    // DETECT BY VOCAB-WIDTH, not `output tid == ir.result`: granite has a LOGITS ScalarMul AFTER the
    // lm_head matmul, so `ir.result` is the ScalarMul's output — the matmul's OWN tid never equals it (the
    // observed miss: the lm_head matmul t1129 kept time-tiling because ir.result was the ScalarMul t1130).
    // The lm_head matmul AND the logits ScalarMul are the ONLY ops whose output spans the result cols
    // (vocab); every intermediate is hidden/intermediate width. So BOTH re-lower at m=1.
    // The mq=1 DECODE bundle is UNAFFECTED (out_rows==1 ⇒ not the prefill tail ⇒ lowered as before).
    let result_cols = ir.tensors[ir.result.index() as u32 as usize].cols;
    // NOT WHEN THE ROWS ARE REQUESTS. The fold is sound only because a prompt's rows are one
    // sequence, so all but the last row's logits are dead. In a batched-decode bundle each row is a
    // DIFFERENT request and every row's logits are sampled — folding to the last row would hand the
    // whole batch request B-1's token. The tail then time-tiles at m=B, which is now addressable:
    // the per-trip advance comes from `StickLayout::dev_off` (`time_tile_sticklayout_stride_tiles_disjoint`),
    // not the flat row-major form that aliased above one row.
    let is_prefill_lm_head_tail = node.output.region.cols.len == result_cols
        && node.output.region.rows.len > 1
        && !rows_are_requests;
    match &node.op {
        // The rest of the arch vocabulary. It reaches this emitter because the SHARED
        // front end expresses every op instead of asserting the unsupported ones away
        // in `lower_region` — which is the point: the IR carries the fact and the
        // TARGET says whether it has a kernel. Enumerated, never `_`, so adding a
        // SubOp is E0004 here rather than a surprise at emission.
        SubOp::TanhSoftCap
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::GateSplit { .. }
        | SubOp::GateApply
        | SubOp::GateScale
        | SubOp::LoadPixels { .. }
        | SubOp::LoadPosEmbeds { .. }
        | SubOp::EmbeddingGather { .. }
        | SubOp::VisionRope
        | SubOp::VarlenAttention { .. }
        | SubOp::EncoderAttn { .. }
        | SubOp::GatedDeltaNet
        | SubOp::GemmaMoe { .. }
        | SubOp::Moe { .. }
        | SubOp::Mean => Unhandled(format!("{:?} has no SuperDSC kernel", node.op)),
        // ⛔ THE ONE PLACE THAT MUST IMPLEMENT IT, so the refusal lives here
        // and names the required lowering rather than the op.
        SubOp::Reshape => Unhandled(
            "SubOp::Reshape reached the SuperDSC lowering. It must become a RESTICKIFY \
                 (a real re-laying copy), NOT a placement alias: `dev_off_stk` places (i,j) \
                 at (j/stk)*(a*stk)+i*stk+(j%stk) where `a` is the ROW COUNT, so two views \
                 over one buffer with different extents disagree about every element. And \
                 `declare_arrangement` will NOT catch an alias — it keys on tensor NAME, and \
                 an alias gives the two views two names."
                .to_string(),
        ),
        SubOp::MatmulTile { .. } if is_prefill_lm_head_tail => {
            match lower_prefill_lm_head_at_m1(node, ir, sym_id_base, layout, quantized) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::MatmulTile { .. } => {
            match lower_matmul_node(node, ir, None, sym_id_base, layout, quantized) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        // ⛔ NO BODY, AND THAT IS THE HONEST STATE. This arm used to lower a `SubtileNode` STRAIGHT
        // TO SuperDSC descriptors — the same violation as the five `*_sdsc` bypasses that were
        // deleted, and the last one left: it was the only producer arm handing the bake an
        // `EmittedOp` with a descriptor and no KTIR program.
        //
        // It is deleted rather than ported because NOTHING IN THIS REPOSITORY CONSTRUCTS A
        // `SubOp::SumReduce` NODE. Every occurrence of the variant is the enum declaration, an
        // arity/ABI table row, a re-roll hash-class arm, the host `eval_node` reference
        // implementation, or a consumer `match` arm — there is no site that builds one, so the
        // split-K combine its doc describes is not emitted by the shared front end. `ibm/main` is
        // the same: it dispatches `lower_sumreduce_node` from `lower_one_node` (main 10698) over a
        // node kind nothing produces, so that body is unreachable there too.
        //
        // A port would therefore be main's text written to satisfy a rule, with no model able to
        // exercise it. A future arch that DOES emit one gets this loud bake error, which names where
        // the body comes from — not a second path to SuperDSC.
        SubOp::SumReduce => Unhandled(format!(
            "SubOp::SumReduce t{} has no KTIR lowering. Nothing in this repository constructs the \
             node, so no body here has ever run; the port source is `ibm/main`'s \
             `lower_sumreduce_node` (main 8432-8466), a variadic pointwise `add` over \
             `[rows, cols]`. Port it through `KtirFunc` — never straight to a descriptor.",
            node.output.tensor.index() as u32,
        )),
        SubOp::Elementwise(kind) => {
            match lower_elementwise_node(node, ir, *kind, sym_id_base, layout) {
                Ok(o) => Ops(vec![o]),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::SiluMul => match lower_silumul_node(node, ir, sym_id_base, layout) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        // ⛔ SPYRE'S RMSNORM MULTIPLIES BY THE STORED GAIN. The gemma-class (1 + w)
        // convention needs a different kernel, and running the Scale one over a
        // zero-centred gain scales every normalized activation by roughly nothing — a
        // model that loads, runs, and is quietly wrong. So it refuses BY NAME.
        //
        // It can reach here at all because the shared front end now EXPRESSES the
        // convention instead of asserting it away in `lower_region`. That is the trade:
        // the IR carries the fact, and the target says whether it has a kernel for it.
        SubOp::RmsNorm {
            eps,
            gain: scratchy_subtile::subtile_ir::GainConvention::Scale,
        } => match lower_rmsnorm_node(node, ir, *eps, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::RmsNorm {
            gain: scratchy_subtile::subtile_ir::GainConvention::OnePlusScale,
            ..
        } => Unhandled(format!(
            "RmsNorm t{} uses the (1 + w) gain convention, for which this emitter has \
             no kernel — its rmsnorm multiplies by the stored gain",
            node.output.tensor.index() as u32,
        )),
        SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
            // ⭐⭐ THE ONE PLACE THE HEAD DIM STOPS BEING A VALUE. Every head_dim-dependent decision
            // downstream is a branch on a CONST, which is reviewable and guardable; a branch on a
            // value is neither. The door's arms are every head dim the workspace's model configs
            // declare, generated by the build script — a head dim with no arm is a loud bake error,
            // and it is answered by a `config.json`, never by an edit here.
            match with_config_head_dim(
                *head_dim,
                LowerRope {
                    node,
                    ir,
                    sym_id_base,
                    layout,
                    rows_are_requests,
                },
            ) {
                Some(Ok(v)) => Ops(v),
                Some(Err(e)) => Unhandled(e.0),
                None => Unhandled(format!(
                    "RopeRotate/RopeAppend head_dim {} has no const-generic instantiation. The head \
                     dim parameterises the device layout (slabs = head_dim/lanes, and the head-major \
                     collapse is valid only at head_dim == lanes), so it must be a const, not a \
                     value. The instantiations are read from the model configs in scope ({}); this \
                     head dim belongs to none of them.",
                    head_dim.get(),
                    scratchy_subtile::model_geometry::geometry_sources(),
                )),
            }
        }
        SubOp::AttnDecode {
            layout: kv, geom, ..
        } => {
            let cap = ir.tensors[kv.cache_tensor().index() as u32 as usize].rows;
            // ⭐⭐ THE ONE PLACE THE MODEL'S HEAD GEOMETRY STOPS BEING VALUES for the attention path.
            // The `#[forward]` macro parsed these numbers out of the model config and minted them
            // onto the tape node; the lowering runs INSIDE that macro's expansion, so this door is
            // where they meet the type system — one instantiation of the whole attention lowering
            // per geometry the workspace's configs declare, and the arms come from the build
            // script's read of those same files. A geometry with no arm is a loud bake error; a
            // geometry whose kv-head count does not divide its query-head count has no arm at all,
            // because `AttnGeometry` cannot be named at it.
            match with_config_attn_geometry(
                *geom,
                LowerAttn {
                    node,
                    ir,
                    cap,
                    active_cap,
                    rows_are_requests,
                    sym_id_base,
                    layout,
                },
            ) {
                Some(Ok(v)) => Ops(v),
                Some(Err(e)) => Unhandled(e.0),
                None => Unhandled(format!(
                    "AttnDecode geometry ({geom}) has no const-generic instantiation. The head \
                     counts and the head dim parameterise the device layout (GQA grouping, slabs, \
                     head strides), so they must be consts, not values. The instantiations are read \
                     from the model configs in scope ({}); this geometry belongs to none of them.",
                    scratchy_subtile::model_geometry::geometry_sources(),
                )),
            }
        }
        SubOp::RmsNormReduce { .. } | SubOp::RmsNormApply { .. } => {
            HostRouted(host_glue_kind(&node.op))
        }
        // ScalarMul (granite embedding/residual/attn/logits multipliers): on-device pointwise `mul` by a
        // bound `[1,1]` scale const (the ATTN_SCALE mechanism) — NOT folded into weights, NOT host-routed.
        // The vocab-wide LOGITS ScalarMul is the second half of the lm_head tail, so in the m>1 prefill
        // bundle it re-lowers at m=1 over the single logits row the matmul above wrote (same reason, same
        // reshape — see is_prefill_lm_head_tail above).
        SubOp::ScalarMul { scale } if is_prefill_lm_head_tail => {
            match lower_scalarmul_node(&node_at_one_row(node), ir, *scale, sym_id_base) {
                Ok(o) => Ops(vec![o]),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::ScalarMul { scale } => {
            match lower_scalarmul_node(node, ir, *scale, sym_id_base) {
                Ok(o) => Ops(vec![o]),
                Err(e) => Unhandled(e.0),
            }
        }
    }
}
/// The PRODUCER half over a whole UNROLLED graph: one KTIR program per node, plus the bundle layout
/// every one of them resolves its addresses through.
///
/// ⛔⛔⛔ THIS USED TO BE CALLED `lower_graph_to_superdsc`, AND KEEPING THAT NAME THROUGH THE SPLIT IS
/// WHAT WENT WRONG. It is `ibm/main`'s function, text unchanged — but main's `lower_one_node` returned
/// SuperDSC descriptors and this one returns KTIR programs, so the body silently stopped producing what
/// its name, its doc and its own worklist error (`"{} SuperDSC op(s) emitted ok"`) all still claim. A
/// caller that asked it for descriptors got `EmittedOp::bare` — `op: None`, `time: 1` — and every
/// question it then asked of them ("how many descriptors?", "is this op tiled?") was answered about a
/// program instead. The WHOLE lowering is [`lower_graph_to_superdsc`] below; this is its first half.
pub fn lower_graph_to_ktir<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // Swept KV extent for the decode attention (paged-attn ladder rung); FULL ⇒ full cap (byte-identical).
    active_cap: ActiveCap,
    // See `lower_attn_node`. Prefill callers pass false.
    rows_are_requests: bool,
) -> Result<(Vec<EmittedOp>, BundleLayout), SuperDscError> {
    // GLOBAL ≤7-segment memory layout (task #55) computed ONCE for the whole bundle.
    // Every op resolves each tensor's HBM address from this (by `t{id}` name) so a
    // tensor shared across ops gets the SAME address — fixing the per-op `arg_index`
    // segment-aliasing bug. Threaded as `Some(&layout)` into every node-lowering; the
    // populated layout (incl. synthetic seg3 offsets) is RETURNED for the manifest.
    // ⛔ NO LAYER CLASSES: this is the UNROLLED lowering, which has no layer loop and therefore no
    // layer boundary to split the weight segment on. An empty map is what tells the layout that
    // banking is not expressible here, leaving the tail spill as the only lever (`&Default::default()`
    // rather than a bool, so there is one spelling of "the layer structure" and not two).
    let bundle_layout =
        compute_bundle_layout(ir, weight_ids, rows_are_requests, &Default::default())?;
    let layout = Some(&bundle_layout);
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(ir.nodes.len());
    // The SINGLE monotonic negative-symbol-id counter for the WHOLE bundle (design
    // risk #1): every tiled tensor/core gets `-(++sym_id_base)`, threaded through
    // each node-lowering so ids never collide across ops in one bundle (torch-spyre
    // `symbol_id_offset_counter`). A per-op restart would alias addresses and
    // silently corrupt — we assert disjointness below.
    let mut sym_id_base: i64 = 0;
    // Collect EVERY distinct unhandled op kind (the full worklist) in one pass —
    // reconnaissance, not a silent skip: a non-empty set is a HARD error so a
    // partial (silently-wrong) bundle is never baked (guard-every-crash rule).
    let mut unhandled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // Ops explicitly ROUTED host-side (RoPE rotate / rope_append / decode attention /
    // pre-chunked RmsNorm halves) — recognized data-movement glue, not a silent skip.
    let mut host_routed: std::collections::BTreeSet<&'static str> =
        std::collections::BTreeSet::new();
    // UNROLLED walk: lower every node via the shared per-op `lower_one_node`. (The
    // RE-ROLLED tape-driven path reuses the SAME helper, walking only the loop body
    // once — see `lower_subtile_tape_to_superdsc`.) MatmulTile's Err is still a hard
    // stop; everything else collects into the worklist/host-routed sets below.
    // fp8 activation-quantize dedup, bundle-scoped: keyed on the activation `t{id}`, so q/k/v (same rms₁
    // output) share ONE quant, while a different layer's activation (distinct tid) never false-shares.
    let mut fp8_quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    for node in &ir.nodes {
        match lower_one_node(
            node,
            ir,
            active_cap,
            rows_are_requests,
            &mut sym_id_base,
            layout,
            &mut fp8_quantized,
        ) {
            NodeLowering::Ops(v) => ops.extend(v),
            NodeLowering::Unhandled(s) => {
                // A MALFORMED matmul is an immediate hard stop (it must never bake);
                // every other op kind accumulates into the worklist for one combined Err.
                if matches!(node.op, SubOp::MatmulTile { .. }) {
                    return Err(SuperDscError(s));
                }
                unhandled.insert(s);
            }
            NodeLowering::HostRouted(s) => {
                host_routed.insert(s);
            }
        }
    }
    if !unhandled.is_empty() {
        return Err(SuperDscError(format!(
            "{} SubtileIR op kind(s) have no KTIR construction ({} program(s) built ok, \
             {} host-routed). WORKLIST: [{}]",
            unhandled.len(),
            ops.len(),
            host_routed.len(),
            unhandled.into_iter().collect::<Vec<_>>().join(", "),
        )));
    }
    Ok((ops, bundle_layout))
}

/// SubtileIR → KTIR → SuperDSC for a whole UNROLLED graph — [`lower_graph_to_ktir`] followed by the
/// ONE KTIR → SuperDSC lowering over each program it built. These are the descriptors main's
/// `lower_graph_to_superdsc` returned: the same builders, in node order, threading the same two
/// bundle-scoped accumulators main's single walk threaded.
///
/// ⛔ THE TWO ACCUMULATORS BELONG TO THE CONSUMER NOW, WHICH IS WHY THEY ARE MINTED HERE AND NOT
/// UPSTREAM. main incremented the negative-symbol-id counter and inserted into the fp8
/// activation-quantize dedup set as it built each DESCRIPTOR, so both are per-bundle facts of the
/// descriptor pass — the producer's copies are threaded but never written (`lower_matmul_node`'s
/// `_quantized`). `ktir_groups_via_superdsc` mints them at exactly this grain for exactly this reason.
pub fn lower_graph_to_superdsc<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    active_cap: ActiveCap,
    rows_are_requests: bool,
) -> Result<(Vec<EmittedOp>, BundleLayout), SuperDscError> {
    let (programs, mut bundle_layout) =
        lower_graph_to_ktir(ir, weight_ids, active_cap, rows_are_requests)?;
    // The four facts no KTIR states, read off the graph's own `AttnDecode` nodes and this walk's own
    // parameters — the same call `lower_subtile_tape_to_ktir`'s re-rolled walk makes.
    let attn_params = attn_bundle_params(ir, rows_are_requests)?;
    let mut sym_id_base: i64 = 0;
    let mut fp8_quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut ops: Vec<EmittedOp> = Vec::with_capacity(programs.len());
    for e in &programs {
        let k = e.ktir.as_ref().ok_or_else(|| {
            SuperDscError(format!(
                "{}: no KTIR program — the node lowering declined this op, and a bundle short a \
                 program computes something else",
                e.op_name
            ))
        })?;
        ops.extend(
            crate::ktir_superdsc_door::lower(
                k,
                &mut sym_id_base,
                Some(&bundle_layout),
                &mut fp8_quantized,
                attn_params,
            )
            .map_err(|err| {
                SuperDscError(format!("{}: KTIR -> SuperDSC: {}", e.op_name, err.message))
            })?,
        );
    }
    // (The former negative-symbol-id disjointness guard is GONE: the concrete-unroll
    // Synthetic intermediates were assigned seg3 offsets ABOVE the colored
    // intermediates during the walk; grow the Intermediate segment's byte count to
    // cover them so the executor allocates a seg3 region large enough (task #55/#56).
    //
    // ⛔ AFTER THE CONSUMER PASS, NOT AFTER THE PRODUCER'S. `BundleLayout::synth` is called by
    // `assemble_rmsnorm` / `matmul_fp8_descriptors` / `lmlast` — all descriptor builders — so at the
    // end of `lower_graph_to_ktir` nothing has been declared yet and this grew seg3 by zero. That is
    // the same ordering fact the re-rolled walk pays for with its explicit declare pass.
    let seg3 = SegRole::Intermediate.segment();
    let synth_high = bundle_layout.synth.borrow().next;
    if synth_high > bundle_layout.segment_bytes[seg3] {
        bundle_layout.segment_bytes[seg3] = synth_high;
    }
    Ok((ops, bundle_layout))
}
/// WHAT A LAUNCH BINDS: for every program, which SubtileIR tensor each of its parameters points
/// at, plus the graph-level facts the worker needs to place those tensors — how many are sources,
/// which one is the result, every tensor's shape, and the id of the runtime attention mask if the
/// graph has one.
///
/// ⛔ THE PAIRING IS CARRIED, NOT RE-DERIVED. A parameter is an ADDRESS; nothing in a finished
/// program says which buffer that address should be, so the only thing that can say is the
/// construction that minted the parameter. Re-deriving it downstream would be a second
/// implementation of the lowering, and the first divergence would show up as a program silently
/// reading someone else's tensor.
pub struct BundleWiring {
    pub nodes: Vec<NodeArgs>,
    pub num_sources: u32,
    pub result_tensor: u32,
    /// `tensor_shapes[id] = (rows, cols)` for every tensor. When `attn_mask` is set, the LAST entry
    /// is the synthetic mask tensor `[1, capacity]`.
    pub tensor_shapes: Vec<(u32, u32)>,
    /// Tensor id of the shared attention runtime length-mask source, if the graph has a maskable
    /// decode. The host fills it `[1, capacity]` per forward step (0 on valid columns, -inf past
    /// the decode position); it is NOT one of `num_sources` and is written by no node.
    pub attn_mask: Option<u32>,
    /// EVERY compile-time scalar the programs read, in registry order: entry `i` is the value the
    /// worker must bind at [`scalarmul_scale_tid`]`(i)` — the model's own multipliers and RMSNorm
    /// epsilons, exactly the set and exactly the order `subtile→superdsc` registers.
    ///
    /// ⛔ NOTHING IS SEEDED INTO IT AND NOTHING EXTRA IS PUSHED. The index IS the device tid, so an
    /// added entry moves every constant after it. Two algebraic identities were once seeded at the
    /// front for this construction's `linalg.*` `outs` seeds, and the mean-of-squares divisor was
    /// pushed beside each epsilon; both are gone — the seeds are immediates, and `1/cols` is bound at
    /// the reserved `RMS_INVCOLS_TID` the way the proven path binds it.
    ///
    /// ⛔ CARRIED HERE BECAUSE A CONSTANT A DESCRIPTOR READS IS A BOUND TENSOR. `dxp_standalone`
    /// has no immediate operand, so `KtirFunc::splat_scale` reads these off reserved tids that
    /// appear in `func.arguments` like any other buffer — which means BOTH consumers of this KTIR
    /// must fill them, the card path through `wiring::constant_steps` and the emulator through its
    /// own source binding. This is the one list they read, so they cannot disagree about it.
    pub scalarmul_scales: Vec<f32>,
}

/// One program's parameter order: `args[i]` is the tensor the i-th parameter addresses.
pub struct NodeArgs {
    pub args: Vec<usize>,
}

/// Lower `ir` for its WIRING alone — the same per-node arms a bake runs, read for the parameter
/// pairings rather than for the programs.
///
/// ⛔ THE LAYOUT IS NOT OPTIONAL, even though the pairing does not depend on placement. It is also
/// the REGISTRY the lowering writes constants into — `lower_attn_node` registers its attention
/// scale in `BundleLayout.scalarmul_scales` and refuses without one ("registry desync") — so a
/// wiring pass that threads `None` does not get placement-independent answers, it gets no answers.
pub fn graph_wiring<F: RopeForm>(
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
) -> Result<BundleWiring, SuperDscError> {
    // ⛔ NO LAYER CLASSES, for the same reason [`lower_graph_to_superdsc`] passes none: this walks
    // the UNROLLED graph, which has no layer loop and so no boundary a weight BANK may fall on.
    let layout = compute_bundle_layout(ir, weight_ids, false, &Default::default())?;
    let mut sym_id_base: i64 = 0;
    let mut quantized = std::collections::HashSet::new();
    let mut nodes = Vec::with_capacity(ir.nodes.len());
    let mut mask: Option<(u32, u32)> = None;
    for node in &ir.nodes {
        let lowered = lower_one_node(
            node,
            ir,
            ActiveCap::FULL,
            false,
            &mut sym_id_base,
            Some(&layout),
            &mut quantized,
        );
        let ops = match lowered {
            NodeLowering::Ops(v) => v,
            // A host-routed op runs off the device, so it has no program and binds nothing. It
            // still occupies a position in the node order, so it contributes an empty entry.
            NodeLowering::HostRouted(_) => Vec::new(),
            NodeLowering::Unhandled(why) => return Err(SuperDscError(why)),
        };
        for e in ops {
            let Some(k) = e.ktir else { continue };
            if let Some(mid) = k.mask {
                // ⭐ THE CAPACITY COMES OFF THE PROGRAM, not off the node. The mask parameter's own
                // `ktdp.construct_memory_view` states `[1, capacity]`, so the number this pass needs to
                // size the host buffer is read where every other extent is read. It used to ride on
                // `KtirNode::mask` as the second half of a pair the LOWERING never looked at.
                let regs =
                    ktir_superdsc::emit::lower_ktir_to_superdsc::regions(&k)
                        .map_err(|e| SuperDscError(e.message))?;
                let cap = regs
                    .iter()
                    .find(|r| r.tid == mid.get())
                    .map(|r| r.v_cols)
                    .ok_or_else(|| {
                        SuperDscError(format!(
                            "the mask buffer t{mid} is not among this program's parameters, so its \
                             `[1, capacity]` view states no capacity to size the host buffer with"
                        ))
                    })?;
                let m = (mid.get(), cap);
                if mask.is_some_and(|prev| prev != m) {
                    return Err(SuperDscError(format!(
                        "the attention mask must be uniform across layers (one shared prefix \
                         capacity), but {:?} and {m:?} were both emitted",
                        mask.unwrap()
                    )));
                }
                mask = Some(m);
            }
            nodes.push(NodeArgs {
                args: k.bindings.iter().map(|b| b.get() as usize).collect(),
            });
        }
    }
    let mut tensor_shapes: Vec<(u32, u32)> = ir.tensors.iter().map(|t| (t.rows, t.cols)).collect();
    let attn_mask = match mask {
        Some((mid, cap)) => {
            // The mask tensor sits at id == ir.tensors.len(); append its `[1, capacity]` shape so
            // host buffer allocation covers it.
            if mid as usize != tensor_shapes.len() {
                return Err(SuperDscError(format!(
                    "the mask's tensor id is {mid} but the next free id is {} — the mask is a \
                     synthetic source placed one past the graph, and an id that is not the next \
                     free one is an id some real tensor already answers to",
                    tensor_shapes.len()
                )));
            }
            tensor_shapes.push((1, cap));
            Some(mid)
        }
        None => None,
    };
    Ok(BundleWiring {
        nodes,
        num_sources: ir.num_sources,
        result_tensor: ir.result.index() as u32,
        tensor_shapes,
        attn_mask,
        scalarmul_scales: layout.scalarmul_scales.clone(),
    })
}
/// RE-ROLLED tape-driven SuperDSC lowering — the mirror of `lower_subtile_tape_to_tk_tape`
/// for SuperDSC (the fix for the 40-min unrolled-bundle compile). Walks the rerolled
/// `tape` (`subtile_tape::reroll_subtile_tape`): pre-loop Computes → `prefix`; the
/// `OpenLoop(Const(iters))`..`CloseLoop` body → `body` (lowered ONCE via the shared
/// `lower_one_node`); post-loop Computes → `suffix`. Collects the per-layer tid map
/// (`Compute::per_layer_out` for outputs + `ComputeInput::External::per_layer` for
/// weights) for the executor's per-iteration address advance.
pub fn lower_subtile_tape_to_ktir<F: RopeForm>(
    tape: &scratchy_subtile::subtile_tape::SubtileTape,
    ir: &SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    // Swept KV extent for the decode attention (paged-attn ladder rung); FULL ⇒ full cap (byte-identical).
    // The driver calls this once per rung with a different `active_cap`; every rung shares the SAME
    // resident KV (storage stays `cap`) and differs only in the body's swept extents.
    active_cap: ActiveCap,
    // See `lower_attn_node`. Prefill callers pass false.
    rows_are_requests: bool,
) -> Result<RolledSuperDsc, SuperDscError> {
    use scratchy_subtile::subtile_tape::{ComputeInput, Instr, LoopBound};
    // ⛔ THE `set_rows_are_requests` / `RestoreRar` DANCE IS DELETED. It pushed the row KIND into a
    // thread-local for the duration of one bundle's lowering, with a `Drop` guard to restore it, purely
    // so the matmul splitter would not need the fact in its signature. That made the kind ambient: no
    // site was obliged to receive it, ~97 sites branched on the row COUNT instead, and a batched decode
    // was emitted as a prefill chunk everywhere the count could not tell them apart. The kind is a
    // COMPILE-TIME CONSTANT of the bundle (this crate is driven by a proc macro that knows the model and
    // the rung as literals), so it belongs in a type — `sdsc_abstract::QueryRows<ROWS_ARE_REQUESTS>` —
    // and in the signatures that need it.
    //
    // What survives is ONE perf gate: a decode batch must not split `mb` (the PT array holds the weight
    // stationary and streams M through it, so an `mb` split reloads the weight per split and cancels the
    // amortization batching exists to buy). That is a throughput/compatibility choice, not a statement
    // about what a row means, and it is named accordingly. See `matmul/dims.rs` for the const-generic
    // fix that removes even this.
    let _prev_split_gate =
        crate::ir::bridge::tiled_op_sdsc_op::matmul::set_split_mb_forbidden(rows_are_requests);
    struct RestoreSplitGate(bool);
    impl Drop for RestoreSplitGate {
        fn drop(&mut self) {
            crate::ir::bridge::tiled_op_sdsc_op::matmul::set_split_mb_forbidden(self.0);
        }
    }
    let _restore_split_gate = RestoreSplitGate(_prev_split_gate);
    // ⭐ THE LAYER STRUCTURE FIRST, because the layout needs it: a weight BANK boundary may only fall
    // on a LAYER boundary, and `compute_bundle_layout` is where the weight segment is packed. Same
    // tape, same `ComputeInput::External::per_layer` the walk below reads — collected once, here.
    let per_layer_ext = per_layer_external_tids(tape);
    let mut bundle_layout =
        compute_bundle_layout(ir, weight_ids, rows_are_requests, &per_layer_ext)?;
    // ON-CARD RESIDUAL (unconditional): thread the loop-carried hidden IN-PLACE, no copy. Pre-scan the
    // loop body for hidden_in (first body node input[0]) + hidden_out (last body node output) and ALIAS
    // hidden_out's placement to hidden_in's → every iteration reads+writes ONE resident buffer, so the
    // residual threads with NO host round-trip and NO device copy (replaces the host thread_hidden).
    // WAR-safe: within a layer hidden_in's last read (computing h1 = h_in+attn) precedes hidden_out's
    // write (the final h_out = h1+mlp add); across iters the shared buffer carries the residual. The
    // shim's thread_hidden becomes a no-op once the placements coincide.
    {
        let (mut pseg, mut pf, mut plast, mut psuf): (u8, Option<u32>, Option<u32>, Option<u32>) =
            (0, None, None, None);
        for instr in tape.instrs() {
            match instr {
                Instr::OpenLoop { .. } => pseg = 1,
                Instr::CloseLoop { .. } => pseg = 2,
                Instr::Compute { node, .. } => {
                    if pseg == 1 {
                        if pf.is_none() {
                            pf = Some(node.index() as u32);
                        }
                        plast = Some(node.index() as u32);
                    } else if pseg == 2 && psuf.is_none() {
                        psuf = Some(node.index() as u32);
                    }
                }
                _ => {}
            }
        }
        if let (Some(f), Some(l)) = (pf, plast)
            && let Some(p) = bundle_layout
                .placements
                .get(&(ir.nodes[f as usize].inputs[0].tensor.index() as u32))
                .cloned()
        {
            let hout = ir.nodes[l as usize].output.tensor.index() as u32;
            // hidden_out (last body op) → hidden_in's buffer: per-iter residual threads in place.
            bundle_layout.placements.insert(hout, p);
            // suffix_in (first suffix op's input[0]) → the same buffer: the body→suffix seam threads
            // in place too, so BOTH host routings (thread_hidden + thread_to_suffix) are unnecessary.
            if let Some(s) = psuf {
                let sin = ir.nodes[s as usize].inputs[0].tensor.index() as u32;
                bundle_layout.placements.insert(sin, p);
            }
        }
    }
    let layout = Some(&bundle_layout);
    let mut sym_id_base: i64 = 0;
    let (mut prefix, mut body, mut suffix): (Vec<EmittedOp>, Vec<EmittedOp>, Vec<EmittedOp>) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut iters: u32 = 0;
    let mut seg: u8 = 0; // 0 = prefix, 1 = body (inside the layer loop), 2 = suffix
    // fp8 activation-quantize dedup (see `lower_matmul_node`): keyed on the activation `t{id}`. Within the
    // once-walked body, q/k/v (same rms₁ tid) share ONE quant; distinct-tid activations never false-share,
    // and the reusing matmul lands in the SAME segment as the quant it reuses (shared tid ⇒ same segment).
    let mut fp8_quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut unhandled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // ⭐ SEEDED FROM THE PRE-PASS RATHER THAN REBUILT. `per_layer_external_tids` already read every
    // per-layer WEIGHT/KV class off this same tape — it had to, because `compute_bundle_layout` needs
    // the layer boundaries to decide weight BANKS, and that runs before any op is emitted. The walk
    // below then adds only what the pre-pass cannot see (node outputs, the resident Kᵀ), and its own
    // `or_insert_with` is a no-op for anything already here.
    //
    // ⛔ A DISAGREEMENT BETWEEN THE TWO IS CAUGHT, NOT ASSUMED AWAY: a class the pre-pass missed was
    // never banked, so its layers sit at their un-banked offsets and the per-layer FORMULA check
    // after this walk refuses the bundle naming that tensor.
    let mut per_layer: std::collections::BTreeMap<u32, Vec<u32>> = per_layer_ext.clone();
    // Which weight BANKS each launch group addresses (0 = prefix, 1 = body, 2 = suffix), accumulated
    // as the ops are emitted. A launch has ONE base per segment, so a group may address exactly one;
    // `group_bank` checks that after the walk.
    let mut group_weight_banks: [std::collections::BTreeSet<u32>; 3] = Default::default();
    // First/last body Compute node (for the residual-stream hidden in/out tids).
    let mut first_body_node: Option<u32> = None;
    let mut last_body_node: Option<u32> = None;
    // First SUFFIX Compute node — its input[0] is the residual the suffix reads (the
    // last layer's post-attn residual, e.g. t780). The rerolled body writes its output
    // to the REPRESENTATIVE-iteration tid (hidden_out_tid, e.g. t360), NOT t780, so the
    // body→suffix seam must thread hidden_out → suffix_in at runtime (else the suffix
    // reads an unwritten slot = 0 → rmsnorm(0)=inf → garbage logits).
    let mut first_suffix_node: Option<u32> = None;
    // Matmul KERNEL weights (layer-0 tid, in=k, out=n) for the device re-tile manifest.
    let mut kernel0: Vec<(u32, u32, u32)> = Vec::new();
    // LAYER blocked-f16 (SCRATCHY_LAYER_KSPLIT): collected down_proj `(dp_per_layer tids, dev_out, b_blocks)`
    // for POST-LOOP block-weight placement overlay (bundle_layout is borrowed by `layout` inside the loop).
    for instr in tape.instrs() {
        match instr {
            Instr::OpenLoop { bound, .. } => {
                iters = match bound {
                    LoopBound::Const(it) => *it,
                    LoopBound::Runtime(_) => {
                        return Err(SuperDscError(
                            "reroll-superdsc: a runtime-bounded layer loop is unsupported (the \
                             decode layer count is a compile-time Const)"
                                .into(),
                        ));
                    }
                };
                seg = 1;
            }
            Instr::CloseLoop { .. } => seg = 2,
            Instr::Compute {
                node,
                inputs,
                per_layer_out,
                ..
            } => {
                let n = &ir.nodes[node.index()];
                // Collect matmul KERNEL weights for the device re-tile manifest: w =
                // inputs[1] is [k,n]=[in,out] row-major; the PT array needs the device
                // tile layout [out/64, in, 64]. in=A.cols (k), out=output.cols (n).
                // Arity-3 fp8 W8A8 matmuls MUST be collected here too: their weight (inputs[1]; inputs[2] is
                // w_scale) needs the 1-byte packed RetileDescriptor from the `fp8_weight_tids` branch below.
                // They were EXCLUDED (arity-2 only), so the shim staged them FLAT → scrambled weights → garbage
                // (`scr chat` incoherent). in_k/out derive identically (inputs[0].cols=k, output.cols=n); the
                // KSPLIT branch (out>16384) never fires for fp8 projs. Root-caused + on-card confirmed 2026-07-16.
                if matches!(n.op, SubOp::MatmulTile { .. })
                    && (n.inputs.len() == 2 || n.inputs.len() == 3)
                {
                    kernel0.push((
                        n.inputs[1].tensor.index() as u32,
                        n.inputs[0].region.cols.len,
                        // DEVICE out extent via the TYPE-SAFE `DeviceWidth` (the SAME rule as
                        // `lower_matmul_node`'s `n_dev` and the worker's weight zero-pad): granite lm_head
                        // 49159→49664. So the RetileDescriptor device_size, the per-core address, and the
                        // staged buffer all agree by construction.
                        DeviceWidth::for_output(
                            n.output.region.rows.len,
                            n.output.region.cols.len,
                            n.inputs[0].region.cols.len,
                        )
                        .get(),
                    ));
                    // KSPLIT fp32-merge (lm_head first-test, n>16384): the split weight ALSO needs B
                    // block-weight descriptors `[KB,N]` under reserved tids `ksplit_block_tid(b)`, matching
                    // the emit branch's block matmuls. Same gate + condition. The kernel0 descriptor loop
                    // below builds each as `[dev_out/64, KB, 64]` (in=KB) — exactly the block matmul's read
                    // (`ksplit_block_weight_kslice_offset_matches_read`). Worker gathers the K-slice bytes.
                }
                if seg == 1 {
                    if first_body_node.is_none() {
                        first_body_node = Some(node.index() as u32);
                    }
                    last_body_node = Some(node.index() as u32);
                }
                if seg == 2 && first_suffix_node.is_none() {
                    first_suffix_node = Some(node.index() as u32);
                }
                // Per-layer OUTPUT tids (the node's output tensor in each layer copy).
                if iters > 1 && per_layer_out.len() as u32 == iters {
                    let outs: Vec<u32> = per_layer_out
                        .iter()
                        .map(|nid| ir.nodes[nid.index()].output.tensor.index() as u32)
                        .collect();
                    per_layer
                        .entry(n.output.tensor.index() as u32)
                        .or_insert(outs);
                }
                // Per-layer WEIGHT/external tids.
                for ci in inputs.iter() {
                    if let ComputeInput::External {
                        tensor,
                        per_layer: pl,
                        ..
                    } = ci
                        && iters > 1
                        && pl.len() as u32 == iters
                    {
                        per_layer
                            .entry(tensor.index() as u32)
                            .or_insert_with(|| pl.iter().map(|t| t.index() as u32).collect());
                    }
                }
                // RESIDENT kct per-layer registration (kill-the-restickify residency): the score reads a
                // per-layer RESIDENT Kᵀ kernel `kct_resident_tid(k_id)`. It's neither an External nor a node
                // output, so the loops above never add it — insert it manually (mirror the downproj block
                // insert below): map each layer's K-cache source tid k_Lv → kct_resident_tid(k_Lv), keyed
                // under the layer-0 kct tid. REQUIRED so the seg2 uniformity guard validates kct's per-layer
                // stride matches k/v — the executor advances kct's seg2 base by v·kv_stride automatically, so
                // a non-uniform kct packing would SILENTLY read the wrong layer (runtime-errors-need-compile-
                // time-checks). per_layer[k_id] was just inserted by the External loop above.
                if let SubOp::AttnDecode { layout: kv, .. } = &n.op
                    && iters > 1
                    && let Some(k_layers) =
                        per_layer.get(&(kv.cache_tensor().index() as u32)).cloned()
                    && k_layers.len() as u32 == iters
                {
                    let kct_tids: Vec<u32> = k_layers
                        .iter()
                        .map(|&k_lv| kct_resident_tid(k_lv))
                        .collect();
                    per_layer
                        .entry(kct_resident_tid(kv.cache_tensor().index() as u32))
                        .or_insert(kct_tids);
                }
                // LAYER blocked-f16 (SCRATCHY_LAYER_KSPLIT): down_proj (large K) → B per-layer block weights.
                // Push each block's kernel0 `[KB,dev_out]` descriptor + its per-layer tid list; collect the
                // down_proj weight's per-layer list for the POST-LOOP placement overlay (block (v,b) at
                // dp_Lv.offset + b·KB·dev_out·2, so weight_stride is unchanged). All offsets Kani-proven.
                // ⭐ THE WEIGHT BANK THIS OP'S GROUP NEEDS. Recorded per group as the ops are emitted
                // because it is a property of the OPERANDS, which only the node knows; a launch binds
                // ONE base per segment, so the set for each group must end up with at most one
                // member. Empty for every op of every unbanked bundle.
                for r in &n.inputs {
                    let tid = r.tensor.index() as u32;
                    if let Some(p) = bundle_layout.placements.get(&tid)
                        && matches!(p.role, SegRole::Weight)
                    {
                        group_weight_banks[(seg as usize).min(2)].insert(p.bank);
                    }
                }
                match lower_one_node(
                    n,
                    ir,
                    active_cap,
                    rows_are_requests,
                    &mut sym_id_base,
                    layout,
                    &mut fp8_quantized,
                ) {
                    NodeLowering::Ops(v) => match seg {
                        0 => prefix.extend(v),
                        1 => body.extend(v),
                        _ => suffix.extend(v),
                    },
                    NodeLowering::Unhandled(s) => {
                        if matches!(n.op, SubOp::MatmulTile { .. }) {
                            return Err(SuperDscError(s));
                        }
                        unhandled.insert(s);
                    }
                    NodeLowering::HostRouted(_) => {}
                }
            }
            Instr::AllocSlot { .. } | Instr::FreeSlot { .. } => {}
        }
    }
    if !unhandled.is_empty() {
        return Err(SuperDscError(format!(
            "reroll-superdsc: {} unhandled op kind(s): [{}]",
            unhandled.len(),
            unhandled.into_iter().collect::<Vec<_>>().join(", "),
        )));
    }
    if iters == 0 {
        return Err(SuperDscError(
            "reroll-superdsc: no layer loop in the rerolled tape (no OpenLoop/CloseLoop) — \
             reroll_subtile_tape found no repeating body"
                .into(),
        ));
    }
    // ⛔⛔⛔ THE SHARED PLAN IS COMPLETED HERE, BEFORE ANY BUNDLE CAN SNAPSHOT IT.
    //
    // MEASURED, granite-3.1-2b on the card: main's decode bundle places rmsnorm scratch for tids
    // {448, 458, 1128}; ours placed {448, 458}. t1128 is the FINAL `model.norm` — the lm-head tail,
    // which lives in the SUFFIX bundle. Every other placement and every segment but seg0 was
    // byte-identical to main; seg0 was short by exactly one rmsnorm's five scratch buffers
    // (Sq16 4096 + Xn 4096 + Mean/Meps/Rinv 128 each = 8576 B at m=1, and 67 sticks × (3n − 2) across
    // the whole ladder).
    //
    // WHY, and it is an ORDERING fact, not a lowering one. `subtile→superdsc` declares a synthetic
    // intermediate during the TAPE WALK — `lower_rmsnorm_node` calls `assemble_rmsnorm`, which calls
    // `BundleLayout::synth`, for prefix, body AND suffix nodes alike — so its layout is complete
    // before the first `emit_bundle`. On this path the producer emits only a KTIR program, and
    // `assemble_rmsnorm` runs LATER, in the consumer (`ktir_groups_via_superdsc`), once per bundle.
    // `emit_bundle` then does `layout.map(bake_layout)` right after lowering ITS OWN ops, and codegen
    // emits prefix → body → suffix. So the BODY's snapshot — which is the layout the RUNTIME loads,
    // sizes its segments from and resolves every address through — held the prefix's and the body's
    // synths and none of the suffix's.
    //
    // The suffix's descriptors were built against the suffix's own later, complete snapshot, so they
    // address seg0 past the extent the runtime allocated from the body's. That is a DMA to an address
    // with no IOMMU translation behind it on the very first forward (the suffix runs in every one),
    // which the card answers with a response block carrying `status=Error`, reported as
    // `scheduler rejected submission` and then a bare `predict: sync rc=-1` naming no operand. It is
    // also why a descriptor-footprint audit beside the emitter was silent: at the moment the suffix's
    // ops were checked, the suffix's own snapshot did contain them.
    //
    // ⭐ THIS DECLARES, IT DOES NOT EMIT. `BundleLayout::synth_bytes` is first-one-wins, so running
    // the consumer over prefix → body → suffix here fixes every synthetic's offset in the SAME order
    // the tape walk would have, and the real pass in `ktir_groups_via_superdsc` finds each one already
    // declared and resolves the identical address. The symbol counter and the fp8-quantize set are
    // throwaways because only the layout is wanted; the descriptors are dropped.
    // The SAME four facts the real consumer pass is handed — read once here, off the graph's own
    // `AttnDecode` nodes and this walk's own parameters. See [`attn_bundle_params`].
    let attn_params = attn_bundle_params(ir, rows_are_requests)?;
    for ops in [&prefix, &body, &suffix] {
        let mut declare_syms: i64 = 0;
        let mut declare_fp8: std::collections::HashSet<String> = std::collections::HashSet::new();
        for e in ops.iter() {
            let Some(k) = e.ktir.as_ref() else { continue };
            crate::ktir_superdsc_door::lower(
                k,
                &mut declare_syms,
                Some(&bundle_layout),
                &mut declare_fp8,
                attn_params,
            )
            .map_err(|err| {
                SuperDscError(format!(
                    "{}: declaring the shared plan's synthetics: {}",
                    e.op_name, err.message
                ))
            })?;
        }
    }
    // Device re-tile manifest: each matmul kernel weight (layer 0) + its per-layer
    // copies → a RetileDescriptor built SOLELY from the DeviceTileLayout witness (the
    // SAME source per_core_addr uses for the stride), so the shim's host re-tile and the
    // on-card per-core address cannot diverge. KERNEL layout = [in,out] sticked on out.
    // fp8 W8A8 weights (`input[1]` of an arity-3 MatmulTile) stage 1-byte / 128-elem stick (SEN143_FP8);
    // dense weights stay fp16 2-byte / 64-stick. The shim reads `word_length` from this descriptor to
    // size the H2D re-tile, so an fp8 weight MUST carry the fp8 descriptor or it is staged as 2-byte.
    let fp8_weight_tids: std::collections::HashSet<u32> = ir
        .nodes
        .iter()
        .filter(|n| matches!(n.op, SubOp::MatmulTile { .. }) && n.inputs.len() == 3)
        .map(|n| n.inputs[1].tensor.index() as u32)
        .collect();
    for (w_tid, in_k, out_n) in &kernel0 {
        let desc = if fp8_weight_tids.contains(w_tid) {
            // fp8 W8A8 weight = the AIU matmulfp8 PACKED tile. Each 128-byte stick holds 64 N-cols each
            // carrying its 2 K-bytes BYTE-ADJACENT — the matmulfp8 in-fold (gen_fp8_kernel_in_fold) reads the
            // 2-pack as "2 fp8 per fp16-width slot, CONTIGUOUS". Device order (outer→inner):
            // [N/64 n-sticks, K/2 k-pairs, 64 n-inner, 2 k-inner]; device[n_stick][k_outer][n_in][k_inner] =
            // host[64·n_stick + 2N·k_outer + n_in + N·k_inner] = weight[2·k_outer+k_inner][64·n_stick+n_in].
            // The [.,.,2,64] order (2 K-rows as two separate 64-N blocks) and a FLAT 128-stick layout BOTH
            // scramble the weights → garbage; only this [.,.,64,2] order runs coherent (on-card 2026-07-16,
            // `scr chat` → "Paris"). (K even + N%64==0 hold for every granite fp8 proj.)
            let k = *in_k as u64;
            let n = *out_n as u64;
            if !k.is_multiple_of(2) || !n.is_multiple_of(64) {
                return Err(SuperDscError(format!(
                    "fp8 W8A8 weight t{w_tid}: packed tile needs K({k})%2==0 and N({n})%64==0"
                )));
            }
            RetileDescriptor {
                device_size: vec![n / 64, k / 2, 64, 2],
                // DISK ORDER, like the fp16 tile below. The worker no longer transposes, so this map
                // reads the `[out, in]` buffer safetensors stores. Converting it is mechanical: a term
                // that stepped an IN index by `x` was `x*n` against `[in, out]` and becomes `x*1`; a
                // term that stepped an OUT index by `y` was `y*1` and becomes `y*k`. So
                // `[64, 2n, 1, n]` → `[64k, 2, k, 1]`, which resolves `host[o*k + i]` for every
                // coordinate (checked exhaustively against the transposed map's `host[i*n + o]`).
                //
                // MISSING THIS BRANCH is what made granite-3.1-8b emit garbage: it is fp8, so its
                // GEMM weights come through here and not the fp16 path, and they were still being read
                // as though something had transposed them.
                stride_map: vec![64 * k, 2, k, 1],
                stick_size: Fp8::ELEMS_PER_STICK,
                word_length: Fp8::WORD_LENGTH,
            }
        } else {
            let tile = DeviceTileLayout::<Fp16>::new(
                &["in", "out"],
                "out",
                &[*in_k as u64, *out_n as u64],
            )?;
            RetileDescriptor {
                device_size: tile.device_size(),
                // DISK ORDER: the worker binds a GEMM weight in the `[out, in]` orientation
                // safetensors stores it, so the re-tile reads it there rather than from a transposed
                // copy. Same elements, one fewer pass over the model at load.
                stride_map: tile.stride_map_disk_order(),
                stick_size: Fp16::ELEMS_PER_STICK,
                word_length: Fp16::WORD_LENGTH,
            }
        };
        bundle_layout.kernel_weights.insert(*w_tid, desc.clone());
        if let Some(layers) = per_layer.get(w_tid) {
            for &t in layers {
                bundle_layout.kernel_weights.insert(t, desc.clone());
            }
        }
    }
    let seg3 = SegRole::Intermediate.segment();
    let synth_high = bundle_layout.synth.borrow().next;
    if synth_high > bundle_layout.segment_bytes[seg3] {
        bundle_layout.segment_bytes[seg3] = synth_high;
    }
    // Per-layer SEGMENT strides for the executor: WEIGHTS (seg1) + KV (seg2) advance
    // by `v·stride` per iteration (the body's baked layer-0 offsets shift to layer-v
    // when the executor passes `seg_base + v·stride`). Verify UNIFORMITY here (a
    // build-time guard) — a non-uniform packing would make the seg-base advance read
    // the WRONG layer's weights (silent garbage). seg3 intermediates (the loop-carried
    // hidden) are EXCLUDED (host-threaded, not strided). 0 = no per-layer tensor there.
    let w_seg = SegRole::Weight.segment();
    let kv_seg = SegRole::Kv.segment();
    let mut weight_stride: u64 = 0;
    let mut kv_stride: u64 = 0;
    for tids in per_layer.values() {
        if tids.len() < 2 {
            continue;
        }
        let Some(p0) = bundle_layout.placements.get(&tids[0]) else {
            continue;
        };
        let seg = p0.segment;
        let target = if seg == w_seg {
            &mut weight_stride
        } else if seg == kv_seg {
            &mut kv_stride
        } else if matches!(p0.role, SegRole::Weight) {
            // ⛔⛔⛔ A PER-LAYER **WEIGHT** OUTSIDE THE STRIDED SEGMENT IS SILENT GARBAGE, and the
            // arm below would have `continue`d past it. The rolled body advances only seg{w_seg}'s
            // base, so layer v would read layer 0's copy of this tensor for all `iters` layers:
            // fluent output, wrong model. [`spill_weight_tail`] may only move the NON-per-layer tail,
            // and this is what holds it to that.
            //
            // ⛔ KEYED ON THE **ROLE**, NOT THE SEGMENT. A per-layer INTERMEDIATE colored into a
            // spill slot is legitimate and must keep falling through — the `WeightOverflow = 5`
            // attempt refused exactly that case (per-layer intermediate t448 in seg5) and read it as
            // proof no segment was available, when the real defect was taking a COLOR.
            return Err(SuperDscError(format!(
                "reroll-superdsc: per-layer WEIGHT t{} is placed in seg{seg}, which gets no \
                 per-layer stride — the rolled body advances only seg{w_seg}, so every one of the \
                 {} layers would read layer 0's copy. Only NON-per-layer weights (the final norm, \
                 the lm_head / tied embedding) may spill; see `spill_weight_tail`.",
                tids[0],
                tids.len(),
            )));
        } else {
            continue; // seg3 hidden / per-layer intermediate color: not stride-advanced
        };
        for w in tids.windows(2) {
            let (Some(a), Some(b)) = (
                bundle_layout.placements.get(&w[0]),
                bundle_layout.placements.get(&w[1]),
            ) else {
                return Err(SuperDscError(
                    "reroll-superdsc: a per-layer tensor is missing a layout placement".into(),
                ));
            };
            if a.segment != seg || b.segment != seg {
                return Err(SuperDscError(
                    "reroll-superdsc: a per-layer tensor changes segment across layers".into(),
                ));
            }
            // ⭐ A BANK BOUNDARY IS THE ONE PLACE THE STRIDE LEGITIMATELY DOES NOT APPLY: layer `v`
            // is `(v / lpb, (v % lpb)·stride)`, so crossing into the next bank resets the offset
            // instead of advancing it. Skipping the pair here is not a hole in the guard — the FULL
            // per-layer formula (bank AND offset, for every layer, not just consecutive pairs) is
            // checked below, which is strictly stronger than this pairwise delta ever was.
            if a.bank != b.bank {
                continue;
            }
            let d = b.offset.wrapping_sub(a.offset);
            if *target == 0 {
                *target = d;
            } else if *target != d {
                return Err(SuperDscError(format!(
                    "reroll-superdsc: NON-UNIFORM per-layer stride in seg{seg} ({} vs {} bytes) — \
                     the executor advances the segment base by v·stride, which needs layers packed \
                     at a uniform stride. Reorder compute_bundle_layout to pack each layer's \
                     {} contiguously.",
                    *target,
                    d,
                    if seg == w_seg { "weights" } else { "KV" },
                )));
            }
        }
    }
    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  ⭐ THE WEIGHT BANK CONTRACT — the launch-time arithmetic, proven here against the addresses
    //  that were actually baked.
    //
    //  The executor reaches layer `v` with `bank = v / layers_per_bank` and
    //  `off[SEG_WEIGHT] = (v % layers_per_bank) · weight_stride`. Every term of that is decided in
    //  `bank_weight_segment`, so all three checks below compare the FORMULA against the placements
    //  rather than re-deriving the policy — the failure mode is a layer reading another layer's
    //  weights, which is fluent, wrong output and nothing else.
    // ══════════════════════════════════════════════════════════════════════════════════════════
    //
    // `layers_per_bank` is READ OFF the placements: how many layers share bank 0. Every layer in one
    // bank (the unbanked case) gives `iters`, which makes the division a no-op.
    let mut layers_per_bank: u32 = iters.max(1);
    for tids in per_layer.values() {
        let Some(p0) = bundle_layout.placements.get(&tids[0]) else {
            continue;
        };
        if p0.segment != w_seg || !matches!(p0.role, SegRole::Weight) || tids.len() as u32 != iters
        {
            continue;
        }
        let in_bank0 = tids
            .iter()
            .filter(|t| {
                bundle_layout
                    .placements
                    .get(*t)
                    .is_some_and(|p| p.bank == 0)
            })
            .count() as u32;
        if in_bank0 == 0 {
            return Err(SuperDscError(format!(
                "reroll-superdsc: per-layer weight class t{} has NO layer in bank 0, but the rolled \
                 body is baked at LAYER 0 — its addresses would name a bank no launch binds.",
                tids[0],
            )));
        }
        layers_per_bank = layers_per_bank.min(in_bank0);
    }
    // Now hold EVERY per-layer weight to `(v / lpb, (v % lpb)·stride + its layer-0 offset)`. This is
    // the check the pairwise delta above cannot make: it validates the bank as well as the offset,
    // and it validates every layer rather than every consecutive pair.
    for tids in per_layer.values() {
        let Some(p0) = bundle_layout.placements.get(&tids[0]) else {
            continue;
        };
        if p0.segment != w_seg || !matches!(p0.role, SegRole::Weight) || tids.len() as u32 != iters
        {
            continue;
        }
        for (v, t) in tids.iter().enumerate() {
            let Some(p) = bundle_layout.placements.get(t) else {
                continue;
            };
            let want_bank = v as u32 / layers_per_bank;
            let want_off = (v as u64 % layers_per_bank as u64) * weight_stride + p0.offset;
            if p.bank != want_bank || p.offset != want_off {
                return Err(SuperDscError(format!(
                    "reroll-superdsc: per-layer weight t{t} (layer {v} of class t{}) is placed at \
                     bank {} offset {}, but the executor will address layer {v} at bank \
                     {want_bank} offset {want_off} ({} layer(s)/bank, stride {weight_stride} B, \
                     layer-0 offset {}). Every layer must sit where the per-layer advance looks, or \
                     that layer reads another layer's weights.",
                    tids[0], p.bank, p.offset, layers_per_bank, p0.offset,
                )));
            }
        }
    }
    // ⛔ AND ONE BANK PER LAUNCH GROUP. A launch is handed ONE base per segment, so a program whose
    // weight operands span two banks cannot be expressed AT ALL — there is no offset that reaches
    // both. `group_weight_banks` was accumulated over the ops as they were emitted, so this is the
    // set of banks each program actually addresses.
    let group_bank = |s: usize, what: &str| -> Result<u32, SuperDscError> {
        let banks = &group_weight_banks[s];
        match banks.len() {
            0 => Ok(0), // reads no weights at all: any bank will do, so bind bank 0
            1 => Ok(*banks.iter().next().expect("len 1")),
            _ => Err(SuperDscError(format!(
                "reroll-superdsc: the {what} program's weights span weight banks {banks:?}, and a \
                 launch has ONE base per segment — no offset reaches both. The non-per-layer weights \
                 (final norm, lm_head / tied embedding) are placed in ONE bank together for exactly \
                 this reason; a weight the {what} reads from another bank would have to be \
                 REPLICATED into every bank that reads it, which `bank_weight_segment` does not do."
            ))),
        }
    };
    let prefix_weight_bank = group_bank(0, "prefix")?;
    let suffix_weight_bank = group_bank(2, "suffix")?;
    // The BODY is baked at layer 0, which the loop above has already proven lives in bank 0, so any
    // OTHER bank in the body means it also reads a weight that is not per-layer — the replication
    // case, refused with its own name rather than as a stride mismatch three steps later.
    if group_bank(1, "body")? != 0 {
        return Err(SuperDscError(format!(
            "reroll-superdsc: the body program addresses weight bank(s) {:?}, but it is baked at \
             LAYER 0 and every launch of it advances bank 0's base. A NON-per-layer weight read \
             inside the layer loop would need replicating into every bank.",
            group_weight_banks[1],
        )));
    }
    // Residual-stream hidden in/out tids for the executor's loop-carried threading:
    // the FIRST body node's input[0] (the layer's hidden-in, e.g. the rmsnorm x) and
    // the LAST body node's output (the layer's hidden-out, the final residual add).
    let hidden_in_tid = first_body_node
        .and_then(|nid| ir.nodes[nid as usize].inputs.first())
        .map(|r| r.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    let hidden_out_tid = last_body_node
        .map(|nid| ir.nodes[nid as usize].output.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    // The suffix's residual input (first suffix node's input[0]) — the executor threads
    // hidden_out → suffix_in once after the layer loop (the body→suffix seam). Same
    // input-ordering convention as hidden_in_tid (rmsnorm input[0] = x = the residual).
    let suffix_in_tid = first_suffix_node
        .and_then(|nid| ir.nodes[nid as usize].inputs.first())
        .map(|r| r.tensor.index() as u32)
        .unwrap_or(u32::MAX);
    // ── DISCOVERY dump (SCRATCHY_SUPERDSC_SEGDUMP) ── which segment/tensor drives the
    //    footprint. Emit runs at cargo-build (AoT bake), so this lands in the build log.
    if std::env::var_os("SCRATCHY_SUPERDSC_SEGDUMP").is_some() {
        let sb = &bundle_layout.segment_bytes;
        let tot: u64 = sb.iter().sum();
        eprintln!(
            "[SEGDUMP] iters={iters} total={:.3}GB segs(GB)=[{}]",
            tot as f64 / 1e9,
            sb.iter()
                .map(|b| format!("{:.3}", *b as f64 / 1e9))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let mut pls: Vec<(&u32, &TensorPlacement)> = bundle_layout.placements.iter().collect();
        pls.sort_by_key(|(_, p)| std::cmp::Reverse(p.size));
        for (tid, p) in pls.into_iter().take(15) {
            eprintln!(
                "[SEGDUMP]   t{tid} seg{} role={:?} size={:.4}GB ({} B)",
                p.segment,
                p.role,
                p.size as f64 / 1e9,
                p.size,
            );
        }
        let syn = bundle_layout.synth.borrow();
        let mut szs: Vec<(&String, &u64)> = syn.sizes.iter().collect();
        szs.sort_by_key(|(_, s)| std::cmp::Reverse(**s));
        for (name, s) in szs.into_iter().take(10) {
            eprintln!(
                "[SEGDUMP]   synth {name} size={:.4}GB ({} B)",
                *s as f64 / 1e9,
                *s
            );
        }
    }
    let kv_request_stride = bundle_layout.kv_request_stride_bytes;
    // Stated only when there IS a request dimension, so an unpaged bundle refuses nothing.
    // ⛔ ALWAYS 0. There is no request dimension in a page any more, so there are no "request rows" for
    // the runtime to bound a shift by. A request is reached by its PAGE, through the host's block table.
    let kv_request_rows = 0u32;
    Ok(RolledSuperDsc {
        prefix,
        body,
        suffix,
        iters,
        layout: bundle_layout,
        per_layer,
        weight_stride,
        layers_per_bank,
        prefix_weight_bank,
        suffix_weight_bank,
        kv_stride,
        kv_request_stride,
        kv_request_rows,
        hidden_in_tid,
        hidden_out_tid,
        suffix_in_tid,
        attn_params,
    })
}
// ══════════════════════════════════════════════════════════════════════════════════════════════
//  LOWERING TO KTIR
//
//  ⭐⭐⭐ THE PROGRAM IS CONSTRUCTED, NEVER PRINTED. `ktir-core`'s `Operation` / `IRFunction` ARE the
//  interchange: the emulator executes the value directly and `#[forward]` bakes it as const data.
//  Nothing here renders MLIR and nothing anywhere parses it.
//
//  ⭐ THE TILING IS THE KTIR PATH's, WHICH IS PROVEN. Each node becomes ONE func whose work division
//  is the one that runs: M across the grid, N in column blocks, and the contraction as an
//  accumulating loop. It is not re-derived here and it is not SuperDSC's — a work division that
//  reads baked addressing, folds or core counts has no KTIR counterpart.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// The element type every emitted tensor carries.
const KTIR_ELEM: ktir_core::dtypes::DType = ktir_core::dtypes::DType::F16;

/// The one spelling `MemorySpace::parse` demands — it has no reverse renderer.
const KTIR_MEMORY_SPACE: &str = "HBM";

/// Per-func construction state. No cross-op SSA threading: each node's func is self-contained —
/// inputs loaded from HBM, output stored to HBM — so LX is bounded to one op's working set.
struct KtirFunc<'g, F: RopeForm> {
    graph: &'g SubtileIR<F>,
    a: &'static Arena,
    ops: Vec<Operation<'static>>,
    next_ssa: u32,
    /// Tensor -> the parameter carrying its HBM base (deduped, first-seen).
    ///
    /// ⛔ AN `Ssa`, NOT A NAME. A constructed parameter has no spelling for two ends to agree about.
    arg_of_tensor: std::collections::BTreeMap<usize, Ssa>,
    /// Tensor ids in first-seen order — the func's parameter order.
    arg_order: Vec<usize>,
    /// `(mask_tensor_id, prefix_capacity)` when this node's attention emits the runtime length
    /// mask. That tensor is synthetic (its id is beyond `graph.tensors`), so its shape is carried
    /// here rather than looked up.
    mask: Option<(u32, u32)>,
    /// `(tensor_id, rows, cols)` for a SYNTHETIC tensor this body names whose shape `graph.shape`
    /// cannot answer, because its id is beyond `graph.tensors`.
    ///
    /// ⭐ SAME MECHANISM AS [`Self::mask`], AND FOR THE SAME REASON — a synth's extent is CARRIED. Every
    /// other synth here is viewed through [`Self::view_shaped`], which takes its dims explicitly and so
    /// never asks; the prefill lm-head tail's `[1, hidden]` LAST_HIDDEN is the one that reaches
    /// [`Self::view_rows`] (as a matmul's activation), and that one derives `cols` from the graph.
    synth_shape: Option<(usize, u32, u32)>,
    /// The SPMD core grid this func runs at. A node raises it to split work across cores so the
    /// per-core LX working set stays small.
    grid: (u32, u32),
    /// The `index` zero every un-offset corner shares, minted once.
    c0: Option<Ssa>,
}

impl<'g, F: RopeForm> KtirFunc<'g, F> {
    fn new(graph: &'g SubtileIR<F>) -> Self {
        Self {
            graph,
            a: Arena::global(),
            ops: Vec::new(),
            next_ssa: 0,
            arg_of_tensor: std::collections::BTreeMap::new(),
            arg_order: Vec::new(),
            mask: None,
            synth_shape: None,
            grid: (1, 1),
            c0: None,
        }
    }

    fn fresh(&mut self) -> Ssa {
        let n = self.next_ssa;
        self.next_ssa += 1;
        Ssa(n)
    }

    fn push(&mut self, op: Operation<'static>) {
        self.ops.push(op);
    }

    fn typed(&self, mut op: Operation<'static>, ty: IrType<'static>) -> Operation<'static> {
        op.result_type = Some(ty);
        op
    }

    fn tensor_ty(&self, dims: Vec<i64>) -> IrType<'static> {
        IrType::Tensor {
            dims: self.a.ints(dims),
            elem: KTIR_ELEM,
        }
    }

    /// The parameter carrying a tensor's HBM base (deduped, first-seen).
    fn arg_for(&mut self, t: TensorId) -> Ssa {
        if let Some(a) = self.arg_of_tensor.get(&t.index()) {
            return *a;
        }
        let a = self.fresh();
        self.arg_of_tensor.insert(t.index(), a);
        self.arg_order.push(t.index());
        a
    }

    /// `construct_memory_view` over a parameter — the one place a view is built.
    ///
    /// ⛔ A PARAMETER IS AN `index` — A START ADDRESS, not a memref. This op's first operand IS that
    /// offset and the memref is its RESULT.
    fn view_of(&mut self, ptr: Ssa, dims: Vec<i64>, strides: Vec<i64>) -> Ssa {
        let a = self.a;
        let view = self.fresh();
        let op = Operation::new(a, Some(view), OpKind::KtdpConstructMemoryView, &[ptr])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(a, AttrKey::Strides, Attr::IntList(a.ints(strides)))
            .with_attr(a, AttrKey::MemorySpace, Attr::Str(KTIR_MEMORY_SPACE))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(KTIR_ELEM));
        let ty = IrType::MemRef {
            dims: a.ints(dims),
            elem: KTIR_ELEM,
        };
        let op = self.typed(op, ty);
        self.push(op);
        view
    }

    /// A view of the whole HBM tensor. Loop-invariant — build once, before any loop that tiles it.
    fn view(&mut self, t: TensorId) -> Ssa {
        let (rows, cols) = self.shape_of(t);
        self.view_shaped(t, rows, cols)
    }

    /// A tensor's `[rows, cols]` — from the graph, or from [`Self::synth_shape`] when the id is a
    /// synthetic beyond `graph.tensors` (`graph.shape` has no row to return for one).
    fn shape_of(&self, t: TensorId) -> (u32, u32) {
        if let Some((st, r, c)) = self.synth_shape
            && st == t.index()
        {
            return (r, c);
        }
        let s = self.graph.shape(t);
        (s.rows, s.cols)
    }

    /// A view with EXPLICIT 2-D sizes/strides — reinterpreting a buffer as another shape (rope
    /// views `[1, heads*hd]` as `[heads, hd]`; a matmul views its weight as its natural `[n, k]`).
    fn view_shaped(&mut self, t: TensorId, sr: u32, sc: u32) -> Ssa {
        let ptr = self.arg_for(t);
        self.view_of(
            ptr,
            vec![i64::from(sr), i64::from(sc)],
            vec![i64::from(sc), 1],
        )
    }

    /// A `[sr, sc]` view whose elements are PACKED e4m3fn — one byte each, row-major. The stride
    /// is in ELEMENTS like every other view here; what changes is the element type, which is what
    /// makes `ktdp.load` widen each byte on read instead of reading two bytes per element.
    fn view_fp8(&mut self, ptr: Ssa, sr: u32, sc: u32) -> Ssa {
        let a = self.a;
        let dims = vec![i64::from(sr), i64::from(sc)];
        let view = self.fresh();
        let op = Operation::new(a, Some(view), OpKind::KtdpConstructMemoryView, &[ptr])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(
                a,
                AttrKey::Strides,
                Attr::IntList(a.ints(vec![i64::from(sc), 1])),
            )
            .with_attr(a, AttrKey::MemorySpace, Attr::Str(KTIR_MEMORY_SPACE))
            .with_attr(
                a,
                AttrKey::Dtype,
                Attr::Dtype(ktir_core::dtypes::DType::Fp8E4m3),
            );
        let ty = IrType::MemRef {
            dims: a.ints(dims),
            elem: ktir_core::dtypes::DType::Fp8E4m3,
        };
        let op = self.typed(op, ty);
        self.push(op);
        view
    }

    /// ⭐⭐⭐ A VIEW OF `rows` ROWS STARTING AT `row_start` — the slice as its own buffer.
    ///
    /// A parameter is a START ADDRESS, so a row slice is expressed by ADVANCING that address, not
    /// by indexing a taller view: `base + row_start·cols` with a `[rows, cols]` shape.
    ///
    /// ⛔ AND THE DIFFERENCE IS NOT COSMETIC. The emulator reconstructs a grid-parallel matmul's M
    /// from the ACTIVATION VIEW'S HEIGHT (`metal.rs`'s `recognize_matmul_loop`: `m: a_shape[0]`,
    /// documented there as the M-from-grid reconstruction). A one-row matmul that reads row 95 of a
    /// `[96, k]` view therefore reconstructs as `m=96`, and its output tile is sized for 96 rows —
    /// MEASURED as the prefill lm-head tail asking for `[96, 16384]` f16 = 3,145,728 bytes against
    /// a 2,097,152-byte LX. Viewed as `[1, k]` at the right address, the same tail is `[1, 16384]`.
    /// ⭐ AND THE OFFSET RIDES THE ACCESS TILE'S ROW INDEX, NOT THE BASE ADDRESS. Both express "start
    /// at row r", but only one is legible to the GEMM offload. `matmul_operand_full`
    /// (`ktir-emulator/src/metal.rs:2054-2075`) takes a view's FIRST OPERAND as the resident root, so
    /// a base of `arith.addi %ptr, %off` names an SSA that resolves to no buffer and the whole
    /// segment leaves the GPU path — correct, and MEASURED at 50-150x (the prefill lm-head tail:
    /// 3618 ms against 23-70 ms for the same op at decode). The row index is where the emulator
    /// looks: `matmul_a_row_offset` (`:1997-2015`) reads the access tile's first index operand and
    /// requires an `arith.constant`, which becomes `m_row_off` and makes
    /// `resolve_gemm_operand_unified_off` read `base + m_row_off * k` (`:1041-1053`).
    ///
    /// Returns the view AND the row index its access tile must carry, so the two cannot disagree:
    /// the height stays `rows` (keeping the `m` reconstruction above), while the offset moves out of
    /// the address. `m > 1` overrides the index with the grid `pid` — the emulator's documented
    /// default, under which it reconstructs all M rows from the stick base.
    fn view_rows(&mut self, t: TensorId, row_start: u32, rows: u32) -> (Ssa, Ssa) {
        let cols = self.shape_of(t).1;
        let ptr = self.arg_for(t);
        let view = self.view_of(
            ptr,
            vec![i64::from(rows), i64::from(cols)],
            vec![i64::from(cols), 1],
        );
        (view, self.idx(row_start))
    }

    /// `construct_access_tile` of shape `[tr, tc]` at corner `[base_r, base_c]`.
    ///
    /// The corner and the SHAPE are independent: the index operands zip against the parent view's
    /// strides, while `Shape` says how much to take.
    fn tile(&mut self, view: Ssa, base_r: Ssa, base_c: Ssa, tr: u32, tc: u32) -> Ssa {
        let a = self.a;
        let acc = self.fresh();
        let dims = vec![i64::from(tr), i64::from(tc)];
        // `access_tile_set` / `access_tile_order` are omitted: the full set and the identity order
        // are what their absence means, and that is what every tile here uses.
        let op = Operation::new(
            a,
            Some(acc),
            OpKind::KtdpConstructAccessTile,
            &[view, base_r, base_c],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
        let op = self.typed(op, IrType::AccessTile { dims: a.ints(dims) });
        self.push(op);
        acc
    }

    /// `ktdp.load` of a `[tr, tc]` access tile.
    fn load_tile(&mut self, acc: Ssa, tr: u32, tc: u32) -> Ssa {
        let v = self.fresh();
        let a = self.a;
        let dims = vec![i64::from(tr), i64::from(tc)];
        // The loaded tile's SHAPE, as an attribute — what reads it is the map-window emitter, which
        // needs a load's shape to bind it as a kernel live-in (and `linalg.broadcast` to know its
        // input's extent). See `binop`.
        let op = Operation::new(a, Some(v), OpKind::KtdpLoad, &[acc]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(dims.clone())),
        );
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// The `index` value for a constant offset — zero is minted once and shared.
    fn idx(&mut self, off: u32) -> Ssa {
        if off == 0
            && let Some(c) = self.c0
        {
            return c;
        }
        let a = self.a;
        let f = self.fresh();
        let op = Operation::new(a, Some(f), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Int(i64::from(off)),
        );
        let op = self.typed(op, IrType::Index);
        self.push(op);
        if off == 0 {
            self.c0 = Some(f);
        }
        f
    }

    /// Region load. Honors the region offset — a fused operand can be a column/row slice of a
    /// larger tensor (the `up` half of a gate-up projection), defaulting to the whole tensor.
    fn load_region(&mut self, tr: &TensorRegion) -> Ssa {
        let (r, c) = (tr.region.rows.len, tr.region.cols.len);
        let view = self.view(tr.tensor);
        let rs = self.idx(tr.region.rows.start);
        let cs = self.idx(tr.region.cols.start);
        let acc = self.tile(view, rs, cs, r, c);
        self.load_tile(acc, r, c)
    }

    /// ⭐ THE PREFILL LAST-ROW EXTRACTION — `hidden/stk` single-stick copies of row `row` of
    /// `[mq, hidden]` into the `[1, hidden]` synthetic `dst`, which is what lets the vocab-wide
    /// lm-head tail run at `m=1` over a tensor whose ROW 0 is the last prompt token.
    ///
    /// ⛔ IT IS A COPY, NOT AN ADDRESS, AND THAT IS SETTLED. This program used not to exist: the tail
    /// simply sliced its activation region to row `mq-1`, on the reasoning that a KTIR tile window can
    /// name a row. It can — the corner below is that same `arith.constant` — but the SuperDSC operand
    /// spelling every emitter builds (`rb(name, rows, cols)`) names a TENSOR, not an offset into one,
    /// so the consumer had nowhere to put the row and read the buffer base instead. MEASURED,
    /// granite-3.1-2b fp8 on card: first generated token `yun` against main's `Hello`, on the same
    /// ladder rung, with the continuation fluent behind it.
    ///
    /// The copies stay SINGLE-STICK because that is the only representable form: `[mq, hidden]` is
    /// `RowBlocked`, so row `r`'s stick-group `j` is 64 contiguous elements at `(j·mq + r)·64`, and
    /// each end then walks at `lanes` — see the consumer (`lower_ktir_to_superdsc::lmlast`) for why
    /// the alternatives (a `rows·lanes` coordInfo stride, a one-hot `sel[1,mq] @ hidden`) are not.
    fn last_row_extract(&mut self, t: TensorId, dst_t: TensorId, mq: u32, row: u32, hidden: u32) {
        let stk = crate::lower_subtile_tape_to_superdsc::Fp16::ELEMS_PER_STICK;
        // The source as the buffer it IS — `[mq, hidden]`. The consumer reads `mq` off this view: it is
        // a term of the copy's address (`j·mq + row`), not decoration.
        let src = self.view_shaped(t, mq, hidden);
        // The destination is SYNTHETIC, so its extent is stated here rather than looked up.
        let dst = self.view_shaped(dst_t, 1, hidden);
        let r = self.idx(row);
        let zero = self.idx(0);
        for j in 0..hidden / stk {
            let c = self.idx(j * stk);
            let acc = self.tile(src, r, c, 1, stk);
            let v = self.load_tile(acc, 1, stk);
            self.store_tile(dst, zero, c, 1, stk, v);
        }
    }

    /// Region store of `val`. Honors the region offset.
    fn store_region(&mut self, val: Ssa, out: &TensorRegion) {
        let (r, c) = (out.region.rows.len, out.region.cols.len);
        let view = self.view(out.tensor);
        let rs = self.idx(out.region.rows.start);
        let cs = self.idx(out.region.cols.start);
        let acc = self.tile(view, rs, cs, r, c);
        self.push(Operation::new(self.a, None, OpKind::KtdpStore, &[val, acc]));
    }

    /// A tiled store of `val` (shape `[tr, tc]`) into `view` at `[base_r, base_c]`.
    fn store_tile(&mut self, view: Ssa, base_r: Ssa, base_c: Ssa, tr: u32, tc: u32, val: Ssa) {
        let acc = self.tile(view, base_r, base_c, tr, tc);
        self.push(Operation::new(self.a, None, OpKind::KtdpStore, &[val, acc]));
    }
}

impl<'g, F: RopeForm> KtirFunc<'g, F> {
    /// A binary elementwise over two whole tiles of the same shape.
    fn binop(&mut self, kind: OpKind, l: Ssa, r: Ssa, dims: Vec<i64>) -> Ssa {
        let v = self.fresh();
        let a = self.a;
        // ⭐ THE SHAPE IS AN ATTRIBUTE, NOT ONLY A TYPE. `linalg.matmul` and `tensor.splat` already
        // carry it, and the emulator's map-window planner reads exactly that attribute to decide an
        // op is TENSOR-valued (`metal.rs`'s `is_tensor_valued` = `attr(Shape).is_some()`). Without
        // it every elementwise op looked scalar, so none ever entered a fused window and the whole
        // GPU map path was compiled but unreachable — MEASURED as `map_region=0` on every forward.
        let op = Operation::new(a, Some(v), kind, &[l, r]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(dims.clone())),
        );
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// A compile-time scalar splatted to `dims`, as a KTIR IMMEDIATE.
    ///
    /// ⛔ AN IMMEDIATE REACHES THE EMULATOR AND NOTHING ELSE, so this is only for a value whose
    /// ADDRESS is never asked for: a `linalg.reduce` / `linalg.matmul` `outs` init, which is a
    /// destination-passing seed the SuperDSC lowering folds away rather than an operand it reads
    /// (`ktir_to_superdsc::reduce` never passes it to a descriptor; `ktir_to_superdsc::matmul`
    /// folds a zero one). Anything a descriptor really reads goes through
    /// [`KtirFunc::splat_scale`] and its bound reserved slot — and `ktir_to_superdsc` REFUSES an
    /// immediate that reaches a descriptor operand, by name, rather than inventing an address for
    /// it.
    ///
    /// ⚠️ ONLY DYADIC SCALES ARE EXACT. Granite's activation multipliers are constants of the
    /// model: the powers of two (2^-7, 2^-4) and 12.0 (1.5·2^3) are exact in f16, but the
    /// residual multiplier 0.22 (= 11/50) is non-dyadic and therefore exact in NO binary float —
    /// a proven ~1.3e-4 gap against the fp32 golden, not a defect of this emission.
    fn splat(&mut self, value: f64, dims: Vec<i64>) -> Ssa {
        let a = self.a;
        let c = self.fresh();
        let op = Operation::new(a, Some(c), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Float(value),
        );
        let op = self.typed(op, IrType::Scalar(KTIR_ELEM));
        self.push(op);
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::TensorSplat, &[c])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(KTIR_ELEM));
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }
}

impl<'g, F: RopeForm> KtirFunc<'g, F> {
    /// A bare scalar constant.
    fn scalar(&mut self, value: f64) -> Ssa {
        let a = self.a;
        let c = self.fresh();
        let op = Operation::new(a, Some(c), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Float(value),
        );
        let op = self.typed(op, IrType::Scalar(KTIR_ELEM));
        self.push(op);
        c
    }

    /// Splat an SSA scalar to `dims`.
    fn splat_of(&mut self, c: Ssa, dims: Vec<i64>) -> Ssa {
        let a = self.a;
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::TensorSplat, &[c])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(KTIR_ELEM));
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// `0`, splatted to `dims` — an IMMEDIATE, and it must stay one.
    ///
    /// ⛔ THIS IS A `linalg.*` `outs` SEED, WHICH NEVER BECOMES A DESCRIPTOR. It once read a bound
    /// registry constant, because a generic op-for-op walk lowered the seed splat itself. The ported
    /// bodies emit `subtile→superdsc`'s own descriptors, where the accumulator seed is part of the
    /// contraction and not an operand — so binding it added TWO slots at the FRONT of
    /// `BundleLayout::scalarmul_scales` and shifted every model scale's registry index by two, which
    /// changes the tid each constant reaches the device at. Nothing about the constant surface may
    /// differ from `subtile→superdsc`; an immediate is invisible to it.
    fn splat_zero(&mut self, dims: Vec<i64>) -> Ssa {
        let c = self.scalar(0.0);
        self.splat_of(c, dims)
    }

    /// `1`, splatted to `dims` — an IMMEDIATE, same reason as [`Self::splat_zero`].
    fn splat_one(&mut self, dims: Vec<i64>) -> Ssa {
        let c = self.scalar(1.0);
        self.splat_of(c, dims)
    }

    /// `-x`, as `0 - x`.
    ///
    /// ⛔ NOT `arith.negf`, BECAUSE THE DEVICE HAS NO NEGATE. `OpFuncs` (deeptools
    /// `sys-arch-spec/arch_enums.h`) carries `sub`/`subtract` but nothing that means negation, and
    /// this emitter's own `op_func_from_str` refuses an unknown name rather than let it become `add`
    /// — MEASURED on the pod as `op_func_from_str: unknown op name "neg" — would have silently
    /// become `add` (wrong op)`. The proven SubtileIR lowering never emits one either.
    ///
    /// ⭐ AND THE FIX BELONGS HERE, NOT DOWNSTREAM. KTIR's vocabulary is FIXED, so it cannot gain a
    /// `silu`/`sigmoid` op and this construction must keep writing them longhand — but WHICH legal
    /// ops it writes them out of is its choice, and choosing ones the target can name is what keeps
    /// `KTIR → SuperDSC` a rename. The zero is an IMMEDIATE ([`Self::splat_zero`]) — it must not be a
    /// registry constant, or the constant surface stops matching `subtile→superdsc`'s.
    fn negate(&mut self, x: Ssa, dims: Vec<i64>) -> Ssa {
        let z = self.splat_zero(dims.clone());
        self.binop(OpKind::ArithSubf, z, x, dims)
    }

    /// An `index` computed at run time — the per-core head arithmetic.
    fn index_op(&mut self, kind: OpKind, l: Ssa, r: Ssa) -> Ssa {
        let v = self.fresh();
        let op = Operation::new(self.a, Some(v), kind, &[l, r]);
        let op = self.typed(op, IrType::Index);
        self.push(op);
        v
    }

    /// A unary elementwise over a whole tile.
    fn unop(&mut self, kind: OpKind, x: Ssa, dims: Vec<i64>) -> Ssa {
        let v = self.fresh();
        let a = self.a;
        // The shape attribute, for the reason `binop` states.
        let op = Operation::new(a, Some(v), kind, &[x]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(dims.clone())),
        );
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// `linalg.reduce` over `dim` with `f`, into a `[1]` init.
    /// `out` is the RESULT shape, not always `[1]`: reducing a `[1, n]` row gives `[1]`, but
    /// reducing `[m, n]` along the key axis gives `[m]` — one value per query row, which is what
    /// lets a whole block of rows share one reduce instead of one reduce each.
    /// ⭐ THE RESULT SHAPE IS STATED AS AN ATTRIBUTE AS WELL AS A TYPE, and that is not redundant.
    /// The emulator's Metal offload reads a shape with `shape_attr_vec`, i.e. off `AttrKey::Shape`
    /// (`ktir-emulator/src/metal.rs`) — it does not look at the op's TYPE. With the shape only in the
    /// type, every window containing a reduce or a broadcast was refused with "broadcast input has no
    /// shape" / "broadcast has no output shape" and fell back to the interpreter: MEASURED as 161 of
    /// 216 refusals on granite-3.1-2b. Same value, stated where both readers look.
    fn reduce(&mut self, x: Ssa, init: Ssa, f: OpKind, dim: i64, out: Vec<i64>) -> Ssa {
        let a = self.a;
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::LinalgReduce, &[x, init])
            .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![dim])))
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(out.clone())))
            .with_attr(a, AttrKey::ReduceFn, Attr::Op(f));
        let ty = self.tensor_ty(out);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// `tensor.extract` of a `[1]` tile's only element.
    fn extract0(&mut self, t: Ssa) -> Ssa {
        let zero = self.idx(0);
        let v = self.fresh();
        let op = Operation::new(self.a, Some(v), OpKind::TensorExtract, &[t, zero]);
        let op = self.typed(op, IrType::Scalar(KTIR_ELEM));
        self.push(op);
        v
    }

    /// `linalg.transpose` of a `[r, c]` tile to `[c, r]`.
    fn transpose(&mut self, x: Ssa, r: u32, c: u32) -> Ssa {
        let a = self.a;
        let dims = vec![i64::from(c), i64::from(r)];
        let init = self.fresh();
        let op = Operation::new(a, Some(init), OpKind::TensorEmpty, &[])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(KTIR_ELEM));
        let ty = self.tensor_ty(dims.clone());
        let op = self.typed(op, ty);
        self.push(op);
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::LinalgTranspose, &[x, init]).with_attr(
            a,
            AttrKey::Permutation,
            Attr::IntList(a.ints(vec![1, 0])),
        );
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// A plain `A[m,k] @ B[k,n]` into a zero init.
    fn matmul_plain(&mut self, x: Ssa, y: Ssa, m: u32, n: u32) -> Ssa {
        let dims = vec![i64::from(m), i64::from(n)];
        let init = self.splat(0.0, dims.clone());
        let a = self.a;
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::LinalgMatmul, &[x, y, init]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(dims.clone())),
        );
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }
}

impl<'g, F: RopeForm> KtirFunc<'g, F> {
    /// A whole-tile value in f32 — the widened dtype the variance reduction runs in.
    fn f32_ty(&self, dims: Vec<i64>) -> IrType<'static> {
        IrType::Tensor {
            dims: self.a.ints(dims),
            elem: ktir_core::dtypes::DType::F32,
        }
    }

    /// An f32 scalar constant, splatted to `dims`.
    fn f32_splat(&mut self, value: f64, dims: Vec<i64>) -> Ssa {
        let a = self.a;
        let c = self.fresh();
        let op = Operation::new(a, Some(c), OpKind::ArithConstant, &[]).with_attr(
            a,
            AttrKey::Value,
            Attr::Float(value),
        );
        let op = self.typed(op, IrType::Scalar(ktir_core::dtypes::DType::F32));
        self.push(op);
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::TensorSplat, &[c])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(
                a,
                AttrKey::Dtype,
                Attr::Dtype(ktir_core::dtypes::DType::F32),
            );
        let ty = self.f32_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// An f32 binary elementwise. States its shape as an ATTRIBUTE as well as a type, the same reason
    /// as [`Self::reduce`]: `binop`/`unop` already do, and the offload reads only the attribute — a
    /// broadcast whose input is one of these was refused with "broadcast input has no shape".
    fn f32_binop(&mut self, kind: OpKind, l: Ssa, r: Ssa, dims: Vec<i64>) -> Ssa {
        let a = self.a;
        let v = self.fresh();
        let op = Operation::new(a, Some(v), kind, &[l, r]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(dims.clone())),
        );
        let ty = self.f32_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// `linalg.broadcast` of `x` into a fresh `dims` init along `dim`.
    fn broadcast(&mut self, x: Ssa, dims: Vec<i64>, dim: i64) -> Ssa {
        let a = self.a;
        let init = self.fresh();
        let op = Operation::new(a, Some(init), OpKind::TensorEmpty, &[])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
            .with_attr(a, AttrKey::Dtype, Attr::Dtype(KTIR_ELEM));
        let ty = self.tensor_ty(dims.clone());
        let op = self.typed(op, ty);
        self.push(op);
        let v = self.fresh();
        // The result shape as an ATTRIBUTE too — see `KtirFunc::reduce` for why both readers need it.
        let op = Operation::new(a, Some(v), OpKind::LinalgBroadcast, &[x, init])
            .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![dim])))
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// ⭐ RMSNORM — `out[i,j] = x[i,j] · (1/sqrt(mean_j(x[i,:]²) + eps)) · gamma[j]`.
    ///
    /// ⛔ THE VARIANCE IS COMPUTED IN f32, NOT THE TILE DTYPE. A real residual stream grows past
    /// |x| ≈ 256, where x² overflows f16 (361k > 65504 → inf → mean=inf → scale=0 → the layer
    /// dies). HBM and the tiles stay f16 — the residual itself fits — and only this reduction
    /// widens; the rescale narrows back, where the scale ≈ 1/rms is small and safe. A synthetic
    /// source never exercises this, because its values are tiny.
    ///
    /// ⭐ AND THE SCALE IS PER ROW. Each of the `r` rows has its own inverse rms, narrowed and
    /// broadcast across the columns — at r=1 that is one row, identical to a scalar splat, but at
    /// prefill each row is scaled by ITS OWN rms rather than row 0's.
    /// ⛔⛔⛔ THE WHOLE `[m, cols]` REGION, IN ONE PROGRAM. This used to loop
    /// `for row in 0..out.region.rows.len` and emit a `[1, cols]` sequence per row.
    ///
    /// That is PRE-TILING IN KTIR, and the consumer cannot undo it: `lower_ktir_to_superdsc::rmsnorm`
    /// reads its row count off the program's own output view, so an `m`-row rmsnorm arrived as `m`
    /// one-row rmsnorms. `assemble_rmsnorm` was then called with `rows = 1` — which picks main's
    /// DECODE algorithm for a PREFILL node — and declared its five scratch buffers at one row.
    /// MEASURED against main on the card, prefill rung at m=7: `Sq16=3x12288B` (4096 B = 1 row) where
    /// main has `Sq16=3x86016B` (28672 B = 7 rows), and the same for `Xn`, `Mean`, `Meps`, `Rinv`.
    /// `subtile→superdsc` calls `assemble_rmsnorm` ONCE per node with the node's real row count, and
    /// that choice — "the decode-vs-prefill ALGORITHM CHOICE it makes" — is the one its own comment
    /// says must stay unchanged.
    ///
    /// A row's rms still depends only on that row: the reduce below runs along the COLUMN axis and
    /// yields `[m]`, one value per row, which is what `KtirFunc::reduce`'s own doc describes as
    /// letting "a whole block of rows share one reduce instead of one reduce each".
    ///
    /// ⭐ NO ROW BLOCKING HERE EITHER, unlike [`KtirFunc::silu_mul`]. Blocking would hand the consumer
    /// the block height instead of `m` and declare the scratch at that height — the same divergence in
    /// a smaller denomination. An emulator-motivated tiling of this program belongs in
    /// `ktir-optimizer`, which is `spyre-emu`-gated; the producer states the shape the node has.
    fn rmsnorm(&mut self, x_r: &TensorRegion, gamma: &TensorRegion, out: &TensorRegion, eps: f32) {
        // The region's extents, read off the output — ONE spelling of each quantity. The width used to
        // arrive as a registry index beside it too, and two spellings of one number is what lets them
        // disagree.
        let c = out.region.cols.len;
        let m = out.region.rows.len;
        let x = self.load_region(x_r);
        let dims = vec![i64::from(m), i64::from(c)];
        let rows = vec![i64::from(m)];

        let xf = {
            let v = self.fresh();
            let op = Operation::new(self.a, Some(v), OpKind::ArithExtf, &[x]);
            let ty = self.f32_ty(dims.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        let x2 = self.f32_binop(OpKind::ArithMulf, xf, xf, dims.clone());
        let sinit = self.f32_splat(0.0, rows.clone());
        let ssum = {
            let a = self.a;
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::LinalgReduce, &[x2, sinit])
                .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![1])))
                .with_attr(a, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
            let ty = self.f32_ty(rows.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        let dts = self.f32_splat(f64::from(c), rows.clone());
        let mean = self.f32_binop(OpKind::ArithDivf, ssum, dts, rows.clone());
        let epst = self.f32_splat(f64::from(eps), rows.clone());
        let meps = self.f32_binop(OpKind::ArithAddf, mean, epst, rows.clone());
        let rms = {
            let v = self.fresh();
            let op = Operation::new(self.a, Some(v), OpKind::MathSqrt, &[meps]);
            let ty = self.f32_ty(rows.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        let onet = self.f32_splat(1.0, rows.clone());
        let inv = self.f32_binop(OpKind::ArithDivf, onet, rms, rows.clone());
        let inv_e = {
            let v = self.fresh();
            let op = Operation::new(self.a, Some(v), OpKind::ArithTruncf, &[inv]);
            let ty = self.tensor_ty(rows);
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        // `[m]` → `[m, c]` along the COLUMN axis: every column of a row shares that row's scale.
        let invb = self.broadcast(inv_e, dims.clone(), 1);
        let xs = self.binop(OpKind::ArithMulf, x, invb, dims.clone());
        // gamma is one row `[1, c]`, loaded RANK-1 so the broadcast produces `[r, c]` — a `[1, c]`
        // load would broadcast to `[1, 1, c]`, since the emulator does not rank-reduce.
        let gcol = self.load_1d(gamma.tensor, c, 0, c);
        let gb = self.broadcast(gcol, dims.clone(), 0);
        let y = self.binop(OpKind::ArithMulf, xs, gb, dims);
        self.store_region(y, out);
    }

    /// SiluMul — `out[j] = (gate / (1 + exp(-gate))) · up`.
    fn silu_mul(&mut self, gate_r: &TensorRegion, up_r: &TensorRegion, out: &TensorRegion) {
        // ROW BLOCKS THAT FIT, not one row at a time. The gate and up tiles are the widest in the
        // model, so a whole `[mq, intermediate]` region does not fit a core's LX at prefill — but a
        // row apiece emits `mq` copies of eight ops, which at the m=96 rung was ~20 ms per layer,
        // the largest non-GEMM cost in the forward once the attention stopped unrolling. Eight tiles
        // are live here (gate, up, neg, exp, the splat, denom, silu, y), so the block height is what
        // keeps those inside the LX — the same accounting `lower_elementwise_node` uses.
        let cols = out.region.cols.len;
        let blk = rows_per_block(cols, 8);
        let mut off = 0u32;
        while off < out.region.rows.len {
            let h = blk.min(out.region.rows.len - off);
            let dims = vec![i64::from(h), i64::from(cols)];
            let gate = self.load_region(&sub_rows(gate_r, off, h));
            let up = self.load_region(&sub_rows(up_r, off, h));
            let neg = self.negate(gate, dims.clone());
            let e = self.unop(OpKind::MathExp, neg, dims.clone());
            let one = self.splat_one(dims.clone());
            let denom = self.binop(OpKind::ArithAddf, one, e, dims.clone());
            let silu = self.binop(OpKind::ArithDivf, gate, denom, dims.clone());
            let y = self.binop(OpKind::ArithMulf, silu, up, dims.clone());
            self.store_region(y, &sub_rows(out, off, h));
            off += h;
        }
    }

    /// A rank-1 load of `prefix_len` elements at `off` from a `full_len`-element buffer — the rope
    /// tables and the rmsnorm gain, whose broadcasts need rank-1 sources.
    fn load_1d(&mut self, t: TensorId, full_len: u32, off: u32, prefix_len: u32) -> Ssa {
        let ptr = self.arg_for(t);
        let view = self.view_of(ptr, vec![i64::from(full_len)], vec![1]);
        let a = self.a;
        let offc = self.idx(off);
        let acc = self.fresh();
        let dims = vec![i64::from(prefix_len)];
        let op = Operation::new(a, Some(acc), OpKind::KtdpConstructAccessTile, &[view, offc])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
        let op = self.typed(
            op,
            IrType::AccessTile {
                dims: a.ints(dims.clone()),
            },
        );
        self.push(op);
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::KtdpLoad, &[acc]).with_attr(
            a,
            AttrKey::Shape,
            Attr::IntList(a.ints(dims.clone())),
        );
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }
}

/// One K/V segment of an attention node, resolved to what its tiles need.
struct KtirSeg {
    k: TensorId,
    v: TensorId,
    row_start: u32,
    seq_len: u32,
    col_start: u32,
}

impl<'g, F: RopeForm> KtirFunc<'g, F> {
    /// ⭐⭐⭐ ATTENTION: HEADS ACROSS THE GRID, ONE QUERY ROW AT A TIME.
    ///
    /// Per (query row `qi`, head `h`): `scores = scale · (q_h · K_kvhᵀ)`, softmax over the key axis,
    /// `out_h = softmax · V_kvh`. GQA picks the kv-head as `h / gqa`, a column offset.
    ///
    /// ⛔ THE HEAD SPLIT IS THE TILING THAT MATTERS. For prefill (`mq > 1`) the func runs at
    /// `grid = [nqh, 1]` and each core computes ONE head across all query rows, so the per-core LX
    /// working set is one head's tiles — `1/nqh` of what a single-core unroll keeps live, which
    /// overflows. Decode (`mq == 1`) keeps `[1, 1]` with every head on one core.
    ///
    /// ⭐ CAUSALITY IS A SLICE, NOT A MASK. The new-token segment holds the query rows' own K/V, so
    /// row `qi` loads `qi + 1` rows. The query-row loop is emit-time, so that slice is a constant.
    ///
    /// ⭐⭐ AND `swept` IS THE RUNG. The prefix cache TENSOR spans the full structural capacity, but
    /// a decode rung sweeps only part of it — that bound is `ActiveCap::resolve`'s answer, computed
    /// by the walk that owns the ladder, and it is what makes a short context cheaper than the full
    /// cap instead of paying for every masked-out slot. The runtime mask then bounds the swept rows
    /// to the ones actually valid this step.
    #[allow(clippy::too_many_arguments)]
    fn attn(
        &mut self,
        q: &TensorRegion,
        segs_in: &[TensorRegion],
        out: &TensorRegion,
        nq: u32,
        hd: u32,
        gqa: u32,
        // ⛔ THE VALUE, NOT A REGISTRY INDEX. `subtile→superdsc` states the attention multiplier as an
        // immediate (`self.scalar(f64::from(scale))`), and this program must state it the same way:
        // the emulator interprets these ops, and a bound `[1,1]` load in its place both changes the
        // arithmetic and hides the constant from the three attention optimizers, which recognise the
        // scale by pattern and require an `arith.constant`. The DEVICE still reads the registry — the
        // ported body resolves the slot from the multiplier its caller states at the door, and PROVES
        // that value is this immediate (`attn_at`'s `program_score_scale` check), so the two consumers
        // cannot scale by different numbers.
        scale: f32,
        mut mask_prefix: bool,
        swept: u32,
    ) {
        let nseg = segs_in.len() / 2;
        let mq = out.region.rows.len;
        let q_view = self.view_shaped(q.tensor, mq, nq * hd);
        let out_view = self.view_shaped(out.tensor, mq, nq * hd);

        let mut segs: Vec<KtirSeg> = Vec::with_capacity(nseg);
        for i in 0..nseg {
            let kr = &segs_in[2 * i];
            let vr = &segs_in[1 + 2 * i];
            // The masked prefix segment is read to the SWEPT extent — the rung — and the host mask
            // bounds that to the rows valid this step.
            let (row_start, seq_len) = if mask_prefix && i == 0 {
                (0, swept.min(self.graph.shape(kr.tensor).rows))
            } else {
                (kr.region.rows.start, kr.region.rows.len)
            };
            // ⛔ A ZERO-LENGTH SEGMENT CONTRIBUTES NO KEY COLUMNS. `ActiveCap::NONE` — a prefill chunk
            // that starts at position 0 — sweeps no resident prefix. Emitting its compute anyway
            // builds a `[mq, 0]` score tile and asks the host GEMM to contract over zero columns,
            // which is where BLAS refuses (`cblas_sgemv` parameter 7, the leading dimension, cannot
            // be 0). So it emits NO access tile and NO matmul.
            //
            // ⭐ BUT THE TENSOR IS STILL NAMED. It used to be dropped outright, which also dropped
            // its VIEW — and a view is what calls `arg_for`, so the resident K/V cache stopped being a
            // parameter of the program at all. `ktir_to_superdsc` then had nothing to read `k_id` /
            // `v_id` / `cap` off, and the card's attention needs them: the hw path keeps the cache
            // RESIDENT and writes this step's K/V into it device-side (`KvShifts`, `superdsc_exec`),
            // where the emulator threads prefix-KV from the host — and that host binding is
            // `#[cfg(not(feature = "spyre-hw"))]`, emulator-only. So the identity has to survive.
            //
            // Keeping the segment with `seq_len == 0` is what does it: the view is built in its
            // ORIGINAL position (so `kviews[0]` is still the prefix), while `live` below is what the
            // compute walks. This is not a KTIR spec change — `ktdp.construct_memory_view` is in the
            // fixed vocabulary; which legal ops this construction chooses to emit is ours.
            //
            // ⛔ AND WITH IT GOES THE MASK. The mask exists to bound the PREFIX segment to the rows
            // valid this step; with no prefix segment there is nothing it could bound. Leaving
            // `mask_prefix` set would take a DEAD segment as the masked one.
            if seq_len == 0 && i == 0 {
                mask_prefix = false;
            }
            segs.push(KtirSeg {
                k: kr.tensor,
                v: vr.tensor,
                row_start,
                seq_len,
                col_start: kr.region.cols.start,
            });
        }
        // Every segment states a view; only these carry key columns to attend.
        let live: Vec<usize> = segs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.seq_len > 0)
            .map(|(i, _)| i)
            .collect();
        let nseg = segs.len();

        // The shared mask tile, loaded once: a synthetic HBM source whose id is the next free one,
        // the same for every attention node (all layers share one prefix capacity).
        let mask_tile = if mask_prefix {
            let cap = segs[0].seq_len;
            let mask_id = self.graph.tensors.len() as u32;
            self.mask = Some((mask_id, cap));
            let mview = self.view_shaped(TensorId::from_index(mask_id as usize), 1, cap);
            let zero = self.idx(0);
            let macc = self.tile(mview, zero, zero, 1, cap);
            Some(self.load_tile(macc, 1, cap))
        } else {
            None
        };

        let kviews: Vec<Ssa> = (0..nseg).map(|i| self.view(segs[i].k)).collect();
        let vviews: Vec<Ssa> = (0..nseg).map(|i| self.view(segs[i].v)).collect();
        let scale_c = self.scalar(f64::from(scale));
        let ninf = self.scalar(-1.0e38);
        let zc = self.scalar(0.0);

        let head_pid = if mq > 1 {
            let p = self.fresh();
            let op = Operation::new(self.a, Some(p), OpKind::KtdpGetComputeTileId, &[]);
            let op = self.typed(op, IrType::Index);
            self.push(op);
            self.grid = (nq, 1);
            Some(p)
        } else {
            None
        };
        let hd_c = self.idx(hd);
        let gqa_c = self.idx(gqa);
        // ⭐ THE REDUCE INITS ARE LOOP-INVARIANT, SO THEY ARE EMITTED ONCE. Both are `[1]` tiles of a
        // constant and both are consumed as a `linalg.reduce` `outs` init, which yields a NEW value
        // rather than writing through — so one tile seeds every row, head and segment. Emitting them
        // inside the row loop cost `mq` copies each: at the m=96 prefill rung that is 192 of the 672
        // splats a granite attention segment runs.
        let mi = self.splat_of(ninf, vec![1]);
        let zit = self.splat_of(zc, vec![1]);

        // ⭐⭐⭐ ONE CAUSAL SEGMENT, mq > 1: THE WHOLE CHUNK IN ONE PASS.
        //
        // The row loop below is emit-time because causality is a SLICE — row `qi` reads `qi + 1`
        // keys — so a chunk emits `mq` copies of every op. MEASURED on granite at the m=96 prefill
        // rung: ~2,400 interpreted ops per layer, ~55 ms x 40 layers = ~2.2 s of a 5.4 s TTFT, and
        // the emulator charges ~25 us PER OP regardless of how small the tile is.
        //
        // When a chunk has NO resident prefix (`start == 0`, the bundle baked `ActiveCap::NONE`) the
        // segment list is exactly the causal one, and then the slice can become a MASK: scores for
        // every row at once, `[mq, hd] · [hd, mq]`, plus a `[mq, mq]` additive triangle. The mask is
        // the attention mask source this arm already registers — `col <= row` is
        // `sdsc_abstract::prefill_causal_col_valid`, and the worker fills it because the bundle's
        // `m_cap > 1`.
        //
        // ⛔ ONLY THIS SHAPE. Any chunk WITH a prefix segment keeps the row loop: its segments have
        // different key extents and the online-softmax combine across them is what the loop below
        // implements. Decode (`mq == 1`) is untouched — the loop runs once and emits what it always
        // did.
        // ⭐ ONE **LIVE** SEGMENT, not one segment. A dead prefix now stays in `segs` so its cache
        // tensor keeps a view (see the loop above), so the count that means "there is only the causal
        // block to attend" is the live one.
        if live.len() == 1 && mq > 1 && !mask_prefix && segs[live[0]].seq_len == mq {
            let s0 = live[0];
            let cap = mq;
            let mask_id = self.graph.tensors.len() as u32;
            self.mask = Some((mask_id, cap));
            let mview = self.view_shaped(TensorId::from_index(mask_id as usize), mq, cap);
            let zero = self.idx(0);
            let macc = self.tile(mview, zero, zero, mq, cap);
            let cmask = self.load_tile(macc, mq, cap);

            let sq = vec![i64::from(mq), i64::from(mq)];
            let od = vec![i64::from(mq), i64::from(hd)];
            let rowv = vec![i64::from(mq)];
            let mi_r = self.splat_of(ninf, rowv.clone());
            let z_r = self.splat_of(zc, rowv.clone());

            let (row_start, col_start) = (segs[s0].row_start, segs[s0].col_start);
            let crow = self.idx(row_start);
            let nh = if head_pid.is_some() { 1 } else { nq };
            for h in 0..nh {
                let (ch, kvcol) = match head_pid {
                    Some(pid) => {
                        let ch = self.index_op(OpKind::ArithMuli, pid, hd_c);
                        let kvh = self.index_op(OpKind::ArithDivui, pid, gqa_c);
                        let kvcol = self.index_op(OpKind::ArithMuli, kvh, hd_c);
                        (ch, kvcol)
                    }
                    None => (self.idx(h * hd), self.idx((h / gqa) * hd)),
                };
                let kcs = self.idx(col_start);
                let ccol = self.index_op(OpKind::ArithAddi, kcs, kvcol);

                let q_acc = self.tile(q_view, zero, ch, mq, hd);
                let qh = self.load_tile(q_acc, mq, hd);
                // ⛔ `s0`, NOT 0 — THE LIVE ORDINAL. This branch reads the ONE live segment, and its
                // index in `segs` is `live[0]`, which is 0 only while a dead prefix is dropped from
                // `segs` entirely. Keeping the dead prefix (so its view names the resident cache for
                // `ktir_to_superdsc`) makes index 0 the DEAD PREFIX — and reading the empty resident
                // cache instead of this chunk's own keys is silent wrong output. MEASURED: it turned
                // the emulator's "The capital of France is Paris…" into "A function that takes a list
                // of numbers…". `row_start`/`col_start` above were already remapped to `segs[s0]`;
                // these two loads were the sites the remap missed.
                let k_acc = self.tile(kviews[s0], crow, ccol, mq, hd);
                let kk = self.load_tile(k_acc, mq, hd);
                let kt = self.transpose(kk, mq, hd);

                let scr = self.matmul_plain(qh, kt, mq, mq);
                let sclt = self.splat_of(scale_c, sq.clone());
                let sc = self.binop(OpKind::ArithMulf, scr, sclt, sq.clone());
                let sc = self.binop(OpKind::ArithAddf, sc, cmask, sq.clone());

                // Row-wise softmax: one reduce per BLOCK, not per row.
                let mx = self.reduce(sc, mi_r, OpKind::ArithMaximumf, 1, rowv.clone());
                let mxb = self.broadcast(mx, sq.clone(), 1);
                let sh = self.binop(OpKind::ArithSubf, sc, mxb, sq.clone());
                let ex = self.unop(OpKind::MathExp, sh, sq.clone());
                let su = self.reduce(ex, z_r, OpKind::ArithAddf, 1, rowv.clone());

                // `s0`, not 0 — the live ordinal, for the reason the K load above states.
                let v_acc = self.tile(vviews[s0], crow, ccol, mq, hd);
                let vv = self.load_tile(v_acc, mq, hd);
                let o = self.matmul_plain(ex, vv, mq, hd);
                let sub = self.broadcast(su, od.clone(), 1);
                let o = self.binop(OpKind::ArithDivf, o, sub, od.clone());
                self.store_tile(out_view, zero, ch, mq, hd, o);
            }
            if head_pid.is_some() {
                self.grid = (nq, 1);
            }
            return;
        }

        for qi in 0..mq {
            let cqi = self.idx(qi);
            let nh = if head_pid.is_some() { 1 } else { nq };
            for h in 0..nh {
                let (ch, kvcol) = match head_pid {
                    Some(pid) => {
                        let ch = self.index_op(OpKind::ArithMuli, pid, hd_c);
                        let kvh = self.index_op(OpKind::ArithDivui, pid, gqa_c);
                        let kvcol = self.index_op(OpKind::ArithMuli, kvh, hd_c);
                        (ch, kvcol)
                    }
                    None => (self.idx(h * hd), self.idx((h / gqa) * hd)),
                };
                let qh_acc = self.tile(q_view, cqi, ch, 1, hd);
                let qh = self.load_tile(qh_acc, 1, hd);

                // Pass 1: per-segment scores and the running global max.
                let mut scores: Vec<(Ssa, u32)> = Vec::with_capacity(live.len());
                let mut gmax: Option<Ssa> = None;
                // ⭐ LIVE SEGMENTS ONLY. A dead prefix keeps its view (so the cache tensor stays a
                // parameter) but contributes no key columns — attending it would build the `[mq, 0]`
                // score tile the zero-length case exists to avoid.
                for &i in &live {
                    let (row_start, seq_len, col_start) =
                        (segs[i].row_start, segs[i].seq_len, segs[i].col_start);
                    let is_new = !(mask_prefix && i == 0);
                    let slen = if is_new && seq_len == mq {
                        qi + 1
                    } else {
                        seq_len
                    };
                    let crow = self.idx(row_start);
                    let kcs = self.idx(col_start);
                    let ccol = self.index_op(OpKind::ArithAddi, kcs, kvcol);
                    let kacc = self.tile(kviews[i], crow, ccol, slen, hd);
                    let kk = self.load_tile(kacc, slen, hd);
                    let kt = self.transpose(kk, slen, hd);
                    let scr = self.matmul_plain(qh, kt, 1, slen);
                    let sdims = vec![1, i64::from(slen)];
                    let sclt = self.splat_of(scale_c, sdims.clone());
                    let sc = self.binop(OpKind::ArithMulf, scr, sclt, sdims.clone());
                    let sc = match (mask_prefix && i == 0, mask_tile) {
                        (true, Some(m)) => self.binop(OpKind::ArithAddf, sc, m, sdims.clone()),
                        _ => sc,
                    };
                    let mx = self.reduce(sc, mi, OpKind::ArithMaximumf, 1, vec![1]);
                    let mxs = self.extract0(mx);
                    scores.push((sc, slen));
                    gmax = Some(match gmax {
                        None => mxs,
                        Some(g) => {
                            let v = self.fresh();
                            let op =
                                Operation::new(self.a, Some(v), OpKind::ArithMaximumf, &[g, mxs]);
                            let op = self.typed(op, IrType::Scalar(KTIR_ELEM));
                            self.push(op);
                            v
                        }
                    });
                }
                let gmax = gmax.expect("at least one segment");

                // Pass 2: exp(score - gmax), and the running global sum.
                let mut es: Vec<(Ssa, u32)> = Vec::with_capacity(nseg);
                let mut gsum: Option<Ssa> = None;
                for (sc, slen) in scores {
                    let sdims = vec![1, i64::from(slen)];
                    let gmb = self.splat_of(gmax, sdims.clone());
                    let sh = self.binop(OpKind::ArithSubf, sc, gmb, sdims.clone());
                    let ex = self.unop(OpKind::MathExp, sh, sdims);
                    let su = self.reduce(ex, zit, OpKind::ArithAddf, 1, vec![1]);
                    let sus = self.extract0(su);
                    es.push((ex, slen));
                    gsum = Some(match gsum {
                        None => sus,
                        Some(g) => {
                            let v = self.fresh();
                            let op = Operation::new(self.a, Some(v), OpKind::ArithAddf, &[g, sus]);
                            let op = self.typed(op, IrType::Scalar(KTIR_ELEM));
                            self.push(op);
                            v
                        }
                    });
                }
                let gsum = gsum.expect("at least one segment");

                // Pass 3: weighted V, accumulated across segments.
                let mut out_acc: Option<Ssa> = None;
                // ⛔ `es` IS INDEXED BY LIVE POSITION, `segs` BY SEGMENT. They coincided only while
                // dead segments were dropped from `segs` entirely; now that a dead prefix stays (to
                // keep its cache tensor named) the enumerate index is the LIVE ordinal and has to be
                // mapped back through `live` — otherwise a prefill chunk reads segment 0's row/col
                // start (the dead prefix's) for the new-token block.
                for (n, (ex, slen)) in es.into_iter().enumerate() {
                    let i = live[n];
                    let (row_start, col_start) = (segs[i].row_start, segs[i].col_start);
                    let sdims = vec![1, i64::from(slen)];
                    let gsb = self.splat_of(gsum, sdims.clone());
                    let w = self.binop(OpKind::ArithDivf, ex, gsb, sdims);
                    let vrw = self.idx(row_start);
                    let vcs = self.idx(col_start);
                    let vcl = self.index_op(OpKind::ArithAddi, vcs, kvcol);
                    let vacc = self.tile(vviews[i], vrw, vcl, slen, hd);
                    let vv = self.load_tile(vacc, slen, hd);
                    let ov = self.matmul_plain(w, vv, 1, hd);
                    out_acc = Some(match out_acc {
                        None => ov,
                        Some(o) => self.binop(OpKind::ArithAddf, o, ov, vec![1, i64::from(hd)]),
                    });
                }
                let oh = out_acc.expect("at least one segment");
                self.store_tile(out_view, cqi, ch, 1, hd, oh);
            }
        }
    }
}

impl<'g, F: RopeForm> KtirFunc<'g, F> {
    /// ⭐⭐⭐ THE MATMUL: ONE WHOLE CONTRACTION, `out[m,n] = A[m,k] @ W[k,n]`.
    ///
    /// ⛔⛔⛔ AND NOTHING ABOUT ITS TILING, BECAUSE TILING IS THE DEVICE'S. This used to build the
    /// schedule here — M across the grid via `ktdp.get_compute_tile_id`, N in column blocks, K as an
    /// accumulating `scf.for` over `iter_args` — sized by `ktir_k_block`/`ktir_n_block` against a
    /// 2 MB LX. That is a real and necessary decision, but it is the EMULATOR's: it executes the ops
    /// and cannot hold `W[2048, 2048]` fp16 (8 MB) as one tile. The card's path never wanted it. It
    /// hands `assemble_matmul` a whole GEMM and DECLARES the division instead — `WorkPlan::divide`
    /// fills `numWkSlicesPerDim_` for dxp's scheduler, `WorkPlan::time_tile_for_lx` fills
    /// `OpSpec.time_tile`, and `render_dxp_input` expands it into trips. `lower_matmul_node` states
    /// it: "the Spyre tape does NOT K-chunk (each `MatmulTile` is a whole GEMM; SuperDSC owns the
    /// K-split via the cost model)".
    ///
    /// So a pre-tiled KTIR was wrong for one of its two consumers: `KTIR → SuperDSC` received a
    /// 64-trip `scf.for` where the proven emitter wanted one contraction, and there is no SuperDSC
    /// op that means "loop". The nest moved VERBATIM to
    /// `ktir_optimizer::matmul_tile::apply_matmul_tiling`, which the emulator runs — same extents,
    /// same block sizes, same op order, so its measured rate is unchanged.
    ///
    /// ⭐ THE ACTIVATION IS VIEWED AS THE ROWS THIS MATMUL READS. A region is `start + len`; the
    /// prefill lm-head tail is one row at `mq-1` of a `[mq, hidden]` tensor, and viewing that slice
    /// (rather than indexing a full-height view) is what keeps it a ONE-row matmul — see `view_rows`.
    ///
    /// ⭐ W BINDS VERBATIM as its on-disk `[out, in]` = `[n, k]` buffer: the matmul reads it with
    /// transpose-B `indexing_maps` (B's map ends in the reduction dim, so the contraction reduces
    /// over k in place), so there is no transpose and no strided gather.
    fn matmul(&mut self, a: &TensorRegion, w: &TensorRegion, out: &TensorRegion) {
        let m = out.region.rows.len;
        let n = out.region.cols.len;
        let kdim = a.region.cols.len;
        let (a_view, a_row) = self.view_rows(a.tensor, a.region.rows.start, a.region.rows.len);
        let w_view = self.view_shaped(w.tensor, n, kdim);
        let out_view = self.view(out.tensor);
        let zero = self.idx(0);

        let a_val = {
            let acc = self.tile(a_view, a_row, zero, m, kdim);
            self.load_tile(acc, m, kdim)
        };
        let w_val = {
            let acc = self.tile(w_view, zero, zero, n, kdim);
            self.load_tile(acc, n, kdim)
        };

        let dims = vec![i64::from(m), i64::from(n)];
        let init = self.splat_zero(dims.clone());
        let arena = self.a;
        let maps: Vec<ktir_core::affine::AffineMap<'static>> = [[0i64, 2], [1, 2], [0, 1]]
            .iter()
            .map(|mm| ktir_core::affine::AffineMap {
                num_dims: 3,
                num_syms: 0,
                exprs: arena.exprs(
                    mm.iter()
                        .map(|d| ktir_core::affine::AffineExpr::Dim(*d as usize))
                        .collect(),
                ),
            })
            .collect();
        let res = self.fresh();
        let op = Operation::new(
            arena,
            Some(res),
            OpKind::LinalgMatmul,
            &[a_val, w_val, init],
        )
        .with_attr(
            arena,
            AttrKey::Shape,
            Attr::IntList(arena.ints(dims.clone())),
        )
        .with_attr(
            arena,
            AttrKey::IndexingMaps,
            Attr::AffineMapList(arena.maps(maps)),
        );
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);

        self.store_tile(out_view, a_row, zero, m, n, res);
    }
}

impl<'g, F: RopeForm> KtirFunc<'g, F> {
    /// ⭐ ROPE over `rows` token positions (1 at decode, m at prefill).
    ///
    /// `x` is `[rows, heads·hd]`, viewed as `[rows·heads, hd]` so each (token, head) is a row.
    /// cos/sin are `[rows, tbl_cols]` — one position per token row — and each is broadcast across
    /// the heads. At rows=1 this is exactly the single-token path, so decode is unchanged; at
    /// rows>1 every token row rotates by ITS OWN position's table.
    ///
    /// ⛔ PER ROW, NOT ONE COLLAPSED TILE. Emitting it per token row is what lets each row use its
    /// own table, and it avoids `tensor.collapse_shape`, which the emulator does not rank-reduce.
    /// W8A8: the same contraction as [`KtirFunc::matmul`], with the weight read as PACKED e4m3fn
    /// and the checkpoint's per-output-column scale applied to the result.
    ///
    /// ⭐ THE WEIGHT IS FP8 IN THE VIEW, NOT IN A SEPARATE PATH. `ktdp.construct_memory_view` names
    /// its element type, so a `[n, k]` view whose elem is `DType::Fp8E4m3` is one byte per element
    /// and `ktdp.load` widens on read (`ktir-core/src/tile.rs`'s `TileStorage::Fp8E4m3`). The
    /// contraction, the K-block loop and the N-column blocking are the fp16 ones — quantization
    /// changes what a weight byte MEANS, not how the matmul is tiled.
    ///
    /// ⛔ WHAT THIS DOES NOT DO: quantize the ACTIVATION. The activation is contracted at
    /// [`KTIR_ELEM`], so this is the weight-quantized half of W8A8 — the per-token activation
    /// quantize a card performs would change the arithmetic, and emitting one here would be
    /// inventing a chain no part of this lowering carries.
    fn matmul_fp8(
        &mut self,
        a: &TensorRegion,
        w: &TensorRegion,
        wscale: &TensorRegion,
        out: &TensorRegion,
    ) {
        let m = out.region.rows.len;
        let n = out.region.cols.len;
        let kdim = a.region.cols.len;
        let (a_view, a_row) = self.view_rows(a.tensor, a.region.rows.start, a.region.rows.len);
        let w_ptr = self.arg_for(w.tensor);
        let w_view = self.view_fp8(w_ptr, n, kdim);
        let out_view = self.view(out.tensor);
        let zero = self.idx(0);
        // The per-output-column scale the checkpoint ships, as a `[1, n]` row.
        let scale_view = self.view(wscale.tensor);

        // ⛔ UNTILED, for the reason [`KtirFunc::matmul`] states at length: the grid/N-block/K-loop
        // schedule is the EMULATOR's and lives in `ktir_optimizer::matmul_tile`. Quantization
        // changes what a weight byte MEANS, not who decides the tiling.
        let a_val = {
            let acc = self.tile(a_view, a_row, zero, m, kdim);
            self.load_tile(acc, m, kdim)
        };
        let w_val = {
            let acc = self.tile(w_view, zero, zero, n, kdim);
            self.load_tile(acc, n, kdim)
        };

        let dims = vec![i64::from(m), i64::from(n)];
        let init = self.splat_zero(dims.clone());
        let arena = self.a;
        let maps: Vec<ktir_core::affine::AffineMap<'static>> = [[0i64, 2], [1, 2], [0, 1]]
            .iter()
            .map(|mm| ktir_core::affine::AffineMap {
                num_dims: 3,
                num_syms: 0,
                exprs: arena.exprs(
                    mm.iter()
                        .map(|d| ktir_core::affine::AffineExpr::Dim(*d as usize))
                        .collect(),
                ),
            })
            .collect();
        let part = self.fresh();
        let op = Operation::new(
            arena,
            Some(part),
            OpKind::LinalgMatmul,
            &[a_val, w_val, init],
        )
        .with_attr(
            arena,
            AttrKey::Shape,
            Attr::IntList(arena.ints(dims.clone())),
        )
        .with_attr(
            arena,
            AttrKey::IndexingMaps,
            Attr::AffineMapList(arena.maps(maps)),
        );
        let ty = self.tensor_ty(dims.clone());
        let op = self.typed(op, ty);
        self.push(op);

        // ⭐ DEQUANT IS A COLUMN-WISE MULTIPLY — arithmetic the contraction owes, not a tiling
        // decision, so it stays here. `wscale` is the checkpoint's `[1, n]` per-output-column row and
        // scales every column this contraction computed.
        let s_val = {
            let acc = self.tile(scale_view, zero, zero, 1, n);
            self.load_tile(acc, 1, n)
        };
        let scaled = self.binop(OpKind::ArithMulf, part, s_val, dims);

        self.store_tile(out_view, a_row, zero, m, n, scaled);
    }

    fn rope(
        &mut self,
        x_t: TensorId,
        cos_t: TensorId,
        sin_t: TensorId,
        out_t: TensorId,
        cols: u32,
        hd: u32,
        rows: u32,
        // The cos/sin tables are pre-tiled to the rope's full width by the host, so position `ri`'s
        // row starts at `ri · tbl_cols`; the first `half` of each row holds the rotary values, and
        // every head's slice is identical.
        tbl_cols: u32,
    ) {
        let heads = cols / hd;
        let half = hd / 2;
        let mh = rows * heads;
        let zero = self.idx(0);
        let half_c = self.idx(half);
        let x_view = self.view_shaped(x_t, mh, hd);
        let out_view = self.view_shaped(out_t, mh, hd);
        let hh = vec![i64::from(heads), i64::from(half)];

        for ri in 0..rows {
            let rbase = self.idx(ri * heads);
            let xf_acc = self.tile(x_view, rbase, zero, heads, half);
            let xf = self.load_tile(xf_acc, heads, half);
            let xs_acc = self.tile(x_view, rbase, half_c, heads, half);
            let xs = self.load_tile(xs_acc, heads, half);

            let cos1 = self.load_1d(cos_t, rows * tbl_cols, ri * tbl_cols, half);
            let cosb = self.broadcast(cos1, hh.clone(), 0);
            let sin1 = self.load_1d(sin_t, rows * tbl_cols, ri * tbl_cols, half);
            let sinb = self.broadcast(sin1, hh.clone(), 0);

            let a1 = self.binop(OpKind::ArithMulf, xf, cosb, hh.clone());
            let a2 = self.binop(OpKind::ArithMulf, xs, sinb, hh.clone());
            let of = self.binop(OpKind::ArithSubf, a1, a2, hh.clone());
            let b1 = self.binop(OpKind::ArithMulf, xf, sinb, hh.clone());
            let b2 = self.binop(OpKind::ArithMulf, xs, cosb, hh.clone());
            let os = self.binop(OpKind::ArithAddf, b1, b2, hh.clone());

            self.store_tile(out_view, rbase, zero, heads, half, of);
            self.store_tile(out_view, rbase, half_c, heads, half, os);
        }
    }
}

impl<'g, F: RopeForm> KtirFunc<'g, F> {
    /// Close the function: its parameters are `index` START ADDRESSES, in first-use order, and its
    /// grid is whatever the node's tiling raised it to.
    /// Close the function.
    ///
    /// ⛔ IT TOOK THE NODE'S DECLARED `[rows, cols]` AND THE CONSUMER NO LONGER NEEDS IT. Every body
    /// that read it now reads the same fact off the program: the row count from the output's own store
    /// windows (`node_rows`), the query rows from `q`'s view (`attn_operands`), and rope's rows from
    /// the cos/sin views, which are bound per position. A record stating what the IR already states is
    /// how the emulator and the card came to be able to disagree.
    /// Close the function, stating WHAT IT COMPUTES.
    ///
    /// ⛔ THE KIND IS A PARAMETER SO NO SITE CAN FORGET IT. It used to be encoded in `name`'s stem and
    /// parsed back out by the consumer's dispatch; see [`Program`](ktir_superdsc::ktir_node::Program).
    fn finish_shaped(
        mut self,
        name: &'static str,
        program: ktir_superdsc::ktir_node::Program,
    ) -> KtirNode {
        let a = self.a;
        self.ops
            .push(Operation::new(a, None, OpKind::FuncReturn, &[]));
        // ⭐⭐⭐ THE PARAMETERS ARE `%0 .. %{n-1}`, AND THAT IS NOT COSMETIC.
        //
        // `arg_for` mints a parameter at its first USE, so a function that stores its result last
        // gets an output pointer numbered in the MIDDLE of its values. Fusion mints one canonical
        // id per tensor BEFORE it renames anything (`ktir-optimizer`'s `fusion.rs`, the
        // `canon_arg … or_insert_with(rename.mint())` loop), i.e. it takes a function's parameters
        // to be its lowest ids. Handing it interleaved ones produced a fused function where the
        // same id was both a parameter and the result of an `arith.mulf`, and a
        // `construct_memory_view` read a pointer nothing defined — MEASURED on granite-3.1-2b as
        // `undefined SSA value: %10`.
        //
        // So the renumber is a PERMUTATION applied at the end: parameters take `0..n` in parameter
        // order, every other value follows in first-definition order. Nothing about the program
        // changes except the names.
        let params: Vec<Ssa> = self
            .arg_order
            .iter()
            .map(|t| self.arg_of_tensor[t])
            .collect();
        let mut remap: std::collections::HashMap<u32, u32> = params
            .iter()
            .enumerate()
            .map(|(i, s)| (s.0, i as u32))
            .collect();
        let mut next = params.len() as u32;
        fn number(
            ops: &[Operation<'static>],
            remap: &mut std::collections::HashMap<u32, u32>,
            next: &mut u32,
        ) {
            for op in ops {
                for s in ssa_attrs(op) {
                    remap.entry(s.0).or_insert_with(|| {
                        let v = *next;
                        *next += 1;
                        v
                    });
                }
                if let Some(r) = op.result {
                    remap.entry(r.0).or_insert_with(|| {
                        let v = *next;
                        *next += 1;
                        v
                    });
                }
                for rg in op.regions {
                    number(rg, remap, next);
                }
            }
        }
        number(&self.ops, &mut remap, &mut next);
        let ops: Vec<Operation<'static>> =
            self.ops.iter().map(|op| rename_op(a, op, &remap)).collect();
        self.ops = ops;
        let args: Vec<(Ssa, IrType<'static>)> = params
            .iter()
            .map(|s| (Ssa(remap[&s.0]), IrType::Index))
            .collect();
        let (gx, gy) = self.grid;
        KtirNode {
            func: ktir_core::ir::IRFunction {
                name,
                arguments: a.args(args),
                operations: a.ops(self.ops),
                grid: (gx as usize, gy as usize, 1),
                return_type: None,
            },
            program,
            // The caller's numbering, as the crate's own opaque key type: this producer numbers a
            // buffer by its SubtileIR tensor index, which is a fact of THIS side of the door.
            bindings: self
                .arg_order
                .iter()
                .map(|&t| ktir_superdsc::ktir_node::BufferId::new(t as u32))
                .collect(),
            // Only WHICH buffer: the capacity this builder also knows is stated by that parameter's own
            // view, so the lowering reads it there rather than being told twice.
            mask: self.mask.map(|(t, _)| ktir_superdsc::ktir_node::BufferId::new(t)),
            // A program writes its own node's output unless its builder says otherwise, and only the
            // prefill lm-head extraction does (see `lower_prefill_lm_head_at_m1`).
            node_out_tid: None,
        }
    }
}
/// The SSA values an operation names in its ATTRIBUTES — `scf.for`'s induction variable and its
/// loop-carried arguments. They are values like any operand, so a renaming that misses them
/// renames the uses of a loop's accumulator without renaming its declaration.
fn ssa_attrs(op: &Operation<'static>) -> Vec<Ssa> {
    op.attributes
        .iter()
        .filter_map(|(_, v)| match v {
            Attr::Ssas(ss) => Some(ss.iter().copied()),
            _ => None,
        })
        .flatten()
        .collect()
}

/// `op` with every SSA name replaced through `remap` — result, operands, `Ssas` attributes and
/// nested regions. Total: a name absent from the map is a bug in the numbering, not a value to
/// leave alone, so it panics rather than emitting a dangling reference.
fn rename_op(
    a: &'static Arena,
    op: &Operation<'static>,
    remap: &std::collections::HashMap<u32, u32>,
) -> Operation<'static> {
    let at = |s: Ssa| {
        Ssa(*remap.get(&s.0).unwrap_or_else(|| {
            panic!(
                "ktir renumber: %{} has no new name — it was used but never defined",
                s.0
            )
        }))
    };
    let mut out = *op;
    out.result = op.result.map(at);
    out.operands = a.ssa(op.operands.iter().copied().map(at).collect());
    out.attributes = a.attrs(
        op.attributes
            .iter()
            .map(|(k, v)| match v {
                Attr::Ssas(ss) => (*k, Attr::Ssas(a.ssa(ss.iter().copied().map(at).collect()))),
                other => (*k, other.clone()),
            })
            .collect(),
    );
    out.regions = a.regions(
        op.regions
            .iter()
            .map(|rg| a.ops(rg.iter().map(|o| rename_op(a, o, remap)).collect()))
            .collect(),
    );
    out
}
