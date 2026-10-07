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
use ktir_core::affine::{AffineExpr, AffineMap, AffineSet};
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
/// ⛔ GELU IS NOT THREE. Its tanh polynomial holds `x` to the very end plus the transient chain
/// AND the ±15 clamp's two splats — peak seven once the clamp landed, budgeted eight — so the `_`
/// arm's three would size a block far below what the chain holds at the MLP's
/// `[m, intermediate]` width. MEASURED at six: `TensorSplat: LX capacity exceeded on core 0:
/// 2027520 + 337920 > 2097152` (five resident, charging the sixth) on gemma-4-12b-it's fused
/// seg10 map window — the fused window's gelu neighbors keep operands live across the chain,
/// so the standalone chain's peak is not the window's peak.
fn ew_live_tiles(kind: EwKind) -> u32 {
    match kind {
        EwKind::Silu => 6,
        EwKind::Gelu => 8,
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

/// How many f16-tile-equivalents an rmsnorm body's variance chain holds at once.
///
/// ⛔ SIX, NOT THREE — THE CHAIN IS f32, DELIBERATELY (see `KtirFunc::rmsnorm`'s head
/// comment: an f16 square overflows at |x| > 256 and zeroes the row). At the square's
/// charge `x` (f16), the `ArithExtf`'d `xf` (f32) and the `xf·xf` being charged (f32)
/// are all live: `1 + 2 + 2` f16-tile equivalents, budgeted six for margin (the
/// gemma-4 error this constant exists for was `1523712 + 1015808 > 2097152` — one f16
/// `[496,512]` plus one f32 `[496,512]` resident, charging a SECOND f32). The
/// fused-segment planner cannot see these temporaries at all — it charges HBM tensor
/// ids at f16 bytes-per-elem — so the ONLY bound on the chain is the block width
/// chosen in `rmsnorm_inv`.
const RMS_LIVE_TILES: u32 = 6;

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

/// `tr` narrowed to `w` columns starting `off` columns into its own region.
fn sub_cols(tr: &TensorRegion, off: u32, w: u32) -> TensorRegion {
    TensorRegion {
        tensor: tr.tensor,
        region: scratchy_subtile::subtile_ir::Region {
            rows: tr.region.rows,
            cols: scratchy_subtile::subtile_ir::Range::new(tr.region.cols.start + off, w),
        },
    }
}

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
        EwKind::Add | EwKind::BiasAdd => E::Add,
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
        EwKind::Add | EwKind::BiasAdd => "add",
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
            EwKind::Add | EwKind::Mul | EwKind::Sub | EwKind::BiasAdd => {
                let a = st.load_region(&sub_rows(&node.inputs[0], off, h));
                let b = st.load_region(&sub_rows(&node.inputs[1], off, h));
                let op = match kind {
                    EwKind::Add | EwKind::BiasAdd => OpKind::ArithAddf,
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
            EwKind::Gelu => {
                let x = st.load_region(&sub_rows(&node.inputs[0], off, h));
                st.gelu(x, dims.clone())
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
        EwKind::Add | EwKind::BiasAdd => {
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
        // ⭐ GELU IS ONE DEVICE OP — `OpFunc::Gelu` is a real DDL primitive (the SFP
        // constant table ships the tanh polynomial). The program states the SAME function
        // longhand (`KtirFunc::gelu`) and stamps `Elementwise(Gelu)`, so the door emits the
        // single op and the emulator interprets the identical math — the silu pattern.
        EwKind::Gelu => {
            let x = st.load_region(&node.inputs[0]);
            st.gelu(x, dims)
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
    gain: scratchy_subtile::subtile_ir::GainConvention,
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
    //
    // ⭐ THE GAIN OFFSET IS AN IMMEDIATE TOO, for the same reason and with the same law: the (1 + w)
    // convention is the same chain with `gamma + 1` in the multiply, and the addition is computed in
    // the PROGRAM rather than folded into the loaded weight — folding would make the bound tensor
    // disagree with the checkpoint.
    let gain_offset = match gain {
        scratchy_subtile::subtile_ir::GainConvention::Scale => 0.0,
        scratchy_subtile::subtile_ir::GainConvention::OnePlusScale => 1.0,
    };
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("rmsnorm_s{}", node.id.index()));
    st.rmsnorm(&node.inputs[0], &node.inputs[1], &node.output, eps, gain_offset);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RmsNorm);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RmsNormUnit`] — the gainless twin of [`lower_rmsnorm_node`].
/// One input (x), no gamma, and the program kind the consumer door dispatches on is
/// `Program::RmsNormUnit`, whose assembler stops at the normalising multiply.
fn lower_rmsnorm_unit_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    eps: f32,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "RmsNormUnit t{} expects 1 input (x), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("rmsnormunit_s{}", node.id.index()));
    st.rmsnorm_unit(&node.inputs[0], &node.output, eps);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RmsNormUnit);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RouterNorm`] — the router's own pre-norm `rmsnorm(x, router.scale)`.
///
/// ⭐ THIS IS `lower_rmsnorm_node`'s COMPUTATION, VERBATIM. The router's gain is a
/// dense `[1, hidden]` row the host stages like every other norm gain (the loader
/// took it from `router.scale` and the front end bound it as operand 1), and the
/// op differs from a layer's post-attention norm in NOTHING the device sees: two
/// operands (`x`, `gain`), one shape-preserving output, an epsilon. Gemma's router
/// gain is the SCALE convention (the loader stores `router.scale` verbatim, no
/// `+1`), which is the one [`Program::RmsNorm`] multiplies by — so the program and
/// the door arm are that node's own, unmodified.
fn lower_router_norm_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    eps: f32,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "RouterNorm t{} expects 2 inputs (x, router), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("routernorm_s{}", node.id.index()));
    st.rmsnorm(&node.inputs[0], &node.inputs[1], &node.output, eps, 0.0);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RmsNorm);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RouterLogits`] — `x · W_router`, the router's dense `[m, experts]`
/// projection.
///
/// ⭐ THIS IS `lower_matmul_node`'s COMPUTATION, VERBATIM. The router gate is a dense
/// weight the loader staged `[hidden, experts]` (the FUF `[k, n]` convention, no
/// transpose), exactly the form every dense `MatmulTile` reads; the op differs from a
/// layer projection in nothing the device sees. Two operands — the fp16 contraction,
/// not W8A8: the router weights are staged dense bf16 (they are `[128, 2816]`, ~1% of
/// an expert bank's footprint, and quantizing them would change the routing decision
/// every expert downstream consumes).
fn lower_router_logits_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "RouterLogits t{} expects 2 inputs (x, router), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    // The REFUSAL half of `lower_matmul_node`, same reason: it is the one place a
    // malformed node is refused before a program is built, and this node's dims are
    // built from exactly the regions `KtirFunc::matmul` reads itself.
    crate::subtile_tape_to_tile_ir::node_to_single_tile_op(node).map_err(SuperDscError)?;
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("routerlogits_s{}", node.id.index()));
    st.matmul(&node.inputs[0], &node.inputs[1], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::Matmul);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::ExpertGatedAct`] — `act(gate) · up` over the MoE pair rows.
///
/// ⭐ SILUMUL'S COMPUTATION, with the act the block declares. The pair rows fold
/// into `[m, k·w]` columns, so the computation is POINTWISE over whatever layout
/// the sort produced — no pair structure is read here, which is what makes this
/// op lowerable before the sort/unsort pair is. The door is [`lk::gated_act`]:
/// silumul's two-op body with `silu`/`gelufwd` dispatched on the act.
fn lower_expert_gated_act_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    act: scratchy_subtile::subtile_ir::GatedAct,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "ExpertGatedAct t{} expects 2 inputs (gate, up), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let program_act = match act {
        scratchy_subtile::subtile_ir::GatedAct::Silu => ktir_superdsc::ktir_node::GatedAct::Silu,
        scratchy_subtile::subtile_ir::GatedAct::Gelu => ktir_superdsc::ktir_node::GatedAct::Gelu,
    };
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("gatedact_s{}", node.id.index()));
    st.gated_act(&node.inputs[0], &node.inputs[1], &node.output, act);
    let k = st.finish_shaped(
        name,
        ktir_superdsc::ktir_node::Program::ExpertGatedAct(program_act),
    );
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RouteSoftmax`] — `softmax(scores, dim=-1)` over `[m, experts]`,
/// the router's score softmax.
/// ⭐ THE RMSNORM'S STRUCTURE, with the softmax's primitives in it. The program
/// ([`KtirFunc::route_softmax`]) states the stability chain longhand; the door
/// ([`route_softmax`]) emits [`assemble_row_softmax`]'s five device ops — one
/// native `max` reduce, the broadcast `sub`, `exp`, one native `sum` reduce, the
/// broadcast `realdiv`. No registry const: the whole chain is data-driven.
fn lower_route_softmax_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "RouteSoftmax t{} expects 1 input (scores), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("routesoftmax_s{}", node.id.index()));
    st.route_softmax(&node.inputs[0], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RouteSoftmax);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RouteRenorm`] — `scores / rowsum(scores)`, mixtral's
/// renormalised top-k. The softmax chain minus the stability subtract and the
/// exp; the same one-stick reduce/broadcast structure.
fn lower_route_renorm_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "RouteRenorm t{} expects 1 input (scores), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("routerenorm_s{}", node.id.index()));
    st.route_renorm(&node.inputs[0], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RouteRenorm);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RouteScale`] — scores times the softmax temperature, a
/// shape-preserving constant multiply.
///
/// ⭐ THIS IS `lower_scalarmul_node`'s COMPUTATION, VERBATIM — the scale is a
/// compile-time constant the tape carries (gemma's `hidden^-0.5`), so the program
/// splats it and multiplies, and the door resolves the `[1,1]` registry slot by the
/// value the splat states. The scores are `[m, experts]`-wide, far inside the LX
/// bound, so the whole-region path always runs.
fn lower_route_scale_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    scale: f32,
    sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "RouteScale t{} expects 1 input (scores), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    lower_scalarmul_node(node, ir, scale, sym_id_base).map(|e| vec![e])
}

/// Lower a [`SubOp::RouteArgsort`] — each row's expert indices sorted by
/// ascending score, as the RANK VECTOR [`KtirFunc::route_argsort`] computes.
///
/// ⭐ NUMERICALLY IDENTICAL TO METAL'S `argpartition.metal` (MLX `block_sort`
/// ascending, NaN-as-greater, ties by index) — the count form
/// `rank[i,j] = |{h : x[i,h] < x[i,j]}| + |{h : x[i,h] == x[i,j] ∧ h < j}|` is
/// exactly that ordering, which is what makes the trailing top-k slice below
/// pick the same experts metal picks. The card's refusal arm stands: this is
/// the emulator's compare/reduce form, not the vendor `topkindex` op.
fn lower_route_argsort_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "RouteArgsort t{} expects 1 input (scores), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("routeargsort_s{}", node.id.index()));
    st.route_argsort(&node.inputs[0], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RouteArgsort);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RouteTopK`] — the last `k` sorted indices of each row,
/// metal's `slice_trailing_cols` (`src_col = axis_size - top_k + j`).
fn lower_route_topk_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    k: u32,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "RouteTopK t{} expects 1 input (sorted), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("routetopk_s{}", node.id.index()));
    st.route_topk(&node.inputs[0], &node.output, k);
    let k_node = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RouteTopK);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k_node);
    Ok(vec![e])
}

/// Lower a [`SubOp::RouteGatherScores`] — the scores at the chosen indices,
/// `out[n, j] = scores[n, idx[n, j]]`: metal's `take_along_axis` at axis -1.
fn lower_route_gather_scores_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "RouteGatherScores t{} expects 2 inputs (scores, indices), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("routegatherscores_s{}", node.id.index()));
    st.route_gather_scores(&node.inputs[0], &node.inputs[1], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RouteGatherScores);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::RouteExpertScale`] — each score times its expert's learned
/// scale: `out[m, j] = scores[m, j] · per_expert_scale[idx[m, j]]`, the f32
/// multiply metal's `moe_per_expert_scale` computes.
fn lower_route_expert_scale_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 3 {
        return Err(SuperDscError(format!(
            "RouteExpertScale t{} expects 3 inputs (scores, indices, router), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("routeexpertscale_s{}", node.id.index()));
    st.route_expert_scale(&node.inputs[0], &node.inputs[1], &node.inputs[2], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::RouteExpertScale);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::ExpertSort`] — the (token, expert) pair rows the expert
/// projections read, in the GATHERED (decode) semantics: the blockwise copy
/// `[m, w] → [m, k·w]` of [`KtirFunc::expert_sort`], numerically identical to
/// metal's gathered bake (which emits nothing and reads token rows `k` times).
fn lower_expert_sort_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    k: u32,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "ExpertSort t{} expects 2 inputs (x, indices), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("expertsort_s{}", node.id.index()));
    st.expert_sort(&node.inputs[0], &node.output, k);
    let k_node = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::ExpertSort);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k_node);
    Ok(vec![e])
}

/// Lower a [`SubOp::ExpertMatmul`] — one projection of each pair's expert
/// over the stacked `[E·out, in]` fp8 weight bank, the gathered form of
/// [`KtirFunc::matmul_fp8`]: metal's `affine_gather_qmv` semantics (the
/// expert's slab selected by the pair's index, the f32-accumulated
/// contraction, the per-channel dequant scale).
///
/// ⭐ THE INDICES ARE NOT THE MATMUL'S INPUT. The wavefront wires the node
/// as `[rows, pairs, w, s]`, where `pairs` is the SORT's `[m, k·w]` output —
/// the routing the projections read. The expert INDICES tensor is the sort
/// node's input 1, so this walks one producer up (`producer_of`) to bind it
/// as the program's fourth parameter. A pairs tensor whose producer is not
/// a two-input ExpertSort is refused here, naming the node.
fn lower_expert_matmul_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    k: u32,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 4 {
        return Err(SuperDscError(format!(
            "ExpertMatmul t{} expects 4 inputs (rows, routing, weights, scales) — the fp8 \
             expert-bank form the gemma-4 wavefront emits; found {}. A 3-input dense-bank form \
             is a different loader's work, and this lowering refuses it rather than guess",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let pairs = &node.inputs[1];
    let idx = match producer_of(ir, pairs.tensor) {
        Some(sort)
            if matches!(sort.op, SubOp::ExpertSort { .. }) && sort.inputs.len() == 2 =>
        {
            &sort.inputs[1]
        }
        _ => {
            return Err(SuperDscError(format!(
                "ExpertMatmul t{} reads its routing from t{} whose producer is not a \
                 two-input ExpertSort — the expert indices cannot be bound, and a projection \
                 without them would read slab 0 for every pair",
                node.output.tensor.index() as u32,
                pairs.tensor.index() as u32
            )))
        }
    };
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("expertmatmul_s{}", node.id.index()));
    st.expert_matmul(&node.inputs[0], idx, &node.inputs[2], &node.inputs[3], &node.output, k);
    let k_node = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::ExpertMatmul);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k_node);
    Ok(vec![e])
}

/// Lower a [`SubOp::ExpertUnsort`] — the pair rows back in token order: the
/// identity copy of [`KtirFunc::expert_unsort`] in the (token, slot) layout.
fn lower_expert_unsort_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "ExpertUnsort t{} expects 2 inputs (rows, routing), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("expertunsort_s{}", node.id.index()));
    st.expert_unsort(&node.inputs[0], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::ExpertUnsort);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::ExpertCombine`] — each token's pair rows summed by its
/// scores, the f32 fma chain of [`KtirFunc::expert_combine`]: metal's
/// `moe_weighted_sum` arithmetic exactly (`fma` accumulate in f32, one f16
/// narrowing at the end).
fn lower_expert_combine_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "ExpertCombine t{} expects 2 inputs (rows, scores), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    // ⛔ A SHARED EXPERT IS REFUSED HERE — the LAST place the bound is
    // visible before a program is minted. The chain this builder emits has
    // no term for a shared expert's contribution (the emu's own doc: "a
    // shared expert's contribution would be an extra `fma` term on the same
    // accumulator"), so a bundle that declares one cannot be lowered — the
    // refusal names the bound, per the no-refusals law that a missing
    // capability is a named error, not a silent drop. The loader-level gate
    // (to_wavefront's Qwen path) is the FIRST line; this is the second.
    let SubOp::ExpertCombine { shared, .. } = &node.op else {
        unreachable!("the caller matched this op");
    };
    if let Some(shared_inter) = shared.0 {
        return Err(SuperDscError(format!(
            "ExpertCombine t{} declares a shared expert (intermediate width {shared_inter}): the \
             KTIR combine is the k-slot fma chain over the routed pair rows, and a shared expert's \
             contribution is an extra term that chain does not state — shared-expert models are \
             the loader's worklist, not a silent drop",
            node.output.tensor.index() as u32
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("expertcombine_s{}", node.id.index()));
    st.expert_combine(&node.inputs[0], &node.inputs[1], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::ExpertCombine);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// final-logit soft cap. One input (the logits), shape-preserving, and the cap
/// is the model constant `final_logit_softcapping` the tape now carries.
///
/// The program is [`KtirFunc::tanhsoftcap`]'s longhand chain (divf by the
/// splatted cap → tanh → mulf by it), stamped `Program::TanhSoftCap`; the
/// emit side reads the SAME cap back structurally
/// ([`program_tanhsoftcap_cap`]) and resolves the `[1,1]` registry const by
/// value — the RmsNormUnit contract, verbatim.
fn lower_tanhsoftcap_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    cap: f32,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "TanhSoftCap t{} expects 1 input (x), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    if cap <= 0.0 {
        return Err(SuperDscError(format!(
            "TanhSoftCap t{} has a non-positive cap {cap} — the cap divides the logits, so a \
             model with `final_logit_softcapping` <= 0 must not emit a softcap tile at all",
            node.output.tensor.index() as u32
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("tanhsoftcap_s{}", node.id.index()));
    let (rows, cols) = (node.output.region.rows.len, node.output.region.cols.len);
    // Whole tiles where they fit, rows where they do not — the same bound, and
    // the same reason, as `lower_elementwise_node`: the softcap spans the
    // VOCAB-wide logits, wider than any activation.
    let by_row = rows > 1 && u64::from(rows) * u64::from(cols) * 3 > u64::from(EW_LX_ELEMS);
    if by_row {
        let blk = rows_per_block(cols, 3);
        let mut off = 0u32;
        while off < rows {
            let h = blk.min(rows - off);
            let dims = vec![i64::from(h), i64::from(cols)];
            let x = st.load_region(&sub_rows(&node.inputs[0], off, h));
            let y = st.tanhsoftcap(x, dims, f64::from(cap));
            st.store_region(y, &sub_rows(&node.output, off, h));
            off += h;
        }
    } else {
        let dims = vec![i64::from(rows), i64::from(cols)];
        let x = st.load_region(&node.inputs[0]);
        let y = st.tanhsoftcap(x, dims, f64::from(cap));
        st.store_region(y, &node.output);
    }
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::TanhSoftCap);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::ScalarWeightMul`] — `out = x · w`, gemma4's per-layer
/// `layer_scalar` multiply. Two inputs (x and the `[1]` weight), shape-
/// preserving. The weight is a HOST-STAGED weight-source (RmsNorm-kind
/// accessor, `.weight` bound by `superdsc_weights`), so the program LOADS it
/// and the descriptor reads the staged `[1,1]` buffer with the scalar
/// broadcast — the ATTN_SCALE operand mode over a worker-bound weight instead
/// of a registry const.
fn lower_scalar_weight_mul_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 2 {
        return Err(SuperDscError(format!(
            "ScalarWeightMul t{} expects 2 inputs (x, weight), found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("scalarwmul_s{}", node.id.index()));
    st.scalar_weight_mul(&node.inputs[0], &node.inputs[1], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::ScalarWeightMul);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// Lower a [`SubOp::Reshape`] — a re-laying copy `[r_in, c_in]` → `[r_out, c_out]` preserving
/// the flat element sequence (gemma4's per-head q/k/v-norm views and flatten-backs). ONE input.
///
/// ⛔ THE COUNT LAW IS CHECKED AT THE PRODUCER TOO: the emit door re-states it (against its own
/// descriptors), but the host oracle's `assert_eq` fires first at expansion — and this producer
/// refusing a malformed node here names the NODE, before any program is minted.
fn lower_reshape_node<F: RopeForm>(
    node: &SubtileNode<F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    if node.inputs.len() != 1 {
        return Err(SuperDscError(format!(
            "Reshape t{} expects 1 input, found {}",
            node.output.tensor.index() as u32,
            node.inputs.len()
        )));
    }
    let (r_in, c_in) = (
        node.inputs[0].region.rows.len,
        node.inputs[0].region.cols.len,
    );
    let (r_out, c_out) = (node.output.region.rows.len, node.output.region.cols.len);
    if u64::from(r_in) * u64::from(c_in) != u64::from(r_out) * u64::from(c_out) {
        return Err(SuperDscError(format!(
            "Reshape t{}: `[{r_in}, {c_in}]` → `[{r_out}, {c_out}]` does not preserve the element \
             count — a reshape moves the same elements to new coordinates",
            node.output.tensor.index() as u32,
        )));
    }
    let mut st = KtirFunc::new(ir);
    let name = Arena::global().str(format!("reshape_s{}", node.id.index()));
    st.reshape(&node.inputs[0], &node.output);
    let k = st.finish_shaped(name, ktir_superdsc::ktir_node::Program::Reshape);
    let mut e = EmittedOp::bare(name.to_string());
    e.ktir = Some(k);
    Ok(vec![e])
}

/// THE PER-HEAD NORM SANDWICH, MATCHED BY DATAFLOW — `Reshape{Times(H>1)} →
/// RmsNorm/RmsNormUnit → Reshape{Times(1)}`, the gemma-4 per-head q/k-norm shape
/// (qwen3's, too — the bridge mints it for every per-head DSL norm).
///
/// ⭐ WHY A DATAFLOW MATCH AND NOT ADJACENCY: the three nodes are INTERLEAVED with the
/// unrelated flat `v_norm` in walk order (the split of q and the split of k/v share a
/// producer), so "the next node is the norm" is false in every gemma-4 layer. What is
/// true — and all this matcher asks — is that each of the three tensors has EXACTLY ONE
/// consumer, identified by scanning `ir.nodes` for reads of it.
///
/// ⭐ THE FLATTEN-BACK PROOF the match demands: the split states `Times(H)` (a per-head
/// view multiplies rows), the flatten states `Times(1)`, both preserve the element count
/// and row-major order (the Reshape law), and the flatten's `[rows, cols]` must equal the
/// SPLIT'S INPUT's `[rows, cols]` — the same extents, not merely the same count, so the
/// flatten restores exactly the arrangement the split left and head window `h` of the
/// output is the split input's columns `h·D..(h+1)·D`.
///
/// ⛔ EVERY CHECK IS A REFUSAL BACK TO THE UN-FUSED PATH. Any extra consumer of the
/// split's output (a skip connection reading the flat q), any op between norm and
/// flatten, an in-place alias (split output == flatten output) — the matcher returns
/// `None` and all three nodes lower exactly as they did before this existed. The
/// byte-identity of the fallback is pinned by test.
struct HeadNormSandwich<'a, F: RopeForm> {
    /// The split `Reshape{Times(H)}` node.
    split: &'a SubtileNode<F>,
    /// The per-head norm node (gained or unit).
    norm: &'a SubtileNode<F>,
    /// The flatten-back `Reshape{Times(1)}` node.
    flatten: &'a SubtileNode<F>,
    /// The per-head width `D` (the split's output columns). The head COUNT is not
    /// carried: it is `full / D` of the split input's own extents, re-derived by the
    /// program builder and the door alike, so no second spelling can disagree.
    head_dim: u32,
}

/// The nodes reading `t` (any region of it) — the consumers a single-use match demands.
fn consumers_of<F: RopeForm>(
    ir: &SubtileIR<F>,
    t: scratchy_subtile::subtile_ir::TensorId,
) -> Vec<&SubtileNode<F>> {
    ir.nodes
        .iter()
        .filter(|n| n.inputs.iter().any(|r| r.tensor == t))
        .collect()
}

/// Whether `n`'s input region is the WHOLE of `t` — the fused norm must read every
/// element of the re-laid view (a partial read means a window this fusion cannot state).
fn reads_whole<F: RopeForm>(n: &SubtileNode<F>, t: u32) -> bool {
    let r = &n.inputs.iter().find(|r| r.tensor.index() as u32 == t);
    match r {
        Some(r) => {
            r.region.rows.start == 0
                && r.region.cols.start == 0
                && r.region.rows.len == n.output.region.rows.len
                && r.region.cols.len == n.output.region.cols.len
        }
        None => false,
    }
}

/// Match the sandwich AT THE SPLIT — the one position the fused program is emitted from.
fn sandwich_at_split<'a, F: RopeForm>(
    node: &'a SubtileNode<F>,
    ir: &'a SubtileIR<F>,
) -> Option<HeadNormSandwich<'a, F>> {
    let SubOp::Reshape {
        rows: scratchy_subtile::subtile_ir::RowScale::Times(heads),
        cols: head_dim,
    } = &node.op
    else {
        return None;
    };
    // A per-head view MULTIPLIES rows; `Times(1)` is a shape-preserving op's non-view.
    if heads.get() == 1 {
        return None;
    }
    if node.inputs.len() != 1 {
        return None;
    }
    let src = &node.inputs[0];
    let split_out = node.output.tensor;
    // The split's single consumer must be the per-head norm, reading the whole view.
    let [norm] = consumers_of(ir, split_out)[..] else {
        return None;
    };
    let (eps, gained) = match &norm.op {
        SubOp::RmsNorm {
            eps,
            gain: scratchy_subtile::subtile_ir::GainConvention::Scale,
        } => (*eps, true),
        // The (1+w) convention adds `+1` to the loaded gain row inside the program —
        // this producer carries it exactly as `lower_rmsnorm_node` does.
        SubOp::RmsNorm {
            eps,
            gain: scratchy_subtile::subtile_ir::GainConvention::OnePlusScale,
        } => (*eps, true),
        SubOp::RmsNormUnit { eps } => (*eps, false),
        _ => return None,
    };
    if !gained && norm.inputs.len() != 1 || gained && norm.inputs.len() != 2 {
        return None;
    }
    if !reads_whole(norm, split_out.index() as u32) {
        return None;
    }
    // The norm's single consumer must be the flatten-back, restoring the split's input.
    let norm_out = norm.output.tensor;
    let [flatten] = consumers_of(ir, norm_out)[..] else {
        return None;
    };
    let SubOp::Reshape {
        rows: scratchy_subtile::subtile_ir::RowScale::Times(one),
        cols: flat_cols,
    } = &flatten.op
    else {
        return None;
    };
    if one.get() != 1 {
        return None;
    }
    // THE FLATTEN-BACK PROOF: same extents as the split's input, so the flat arrangement
    // — and therefore each head window — is restored exactly. A same-count/different-shape
    // flatten (e.g. `[m, H·D] → [m·D, H]`) permutes elements and is NOT this fusion.
    if flatten.inputs.len() != 1
        || flatten.inputs[0].tensor != norm_out
        || flatten.output.region.rows.len != src.region.rows.len
        || *flat_cols != src.region.cols.len
        || flatten.output.tensor == split_out
    {
        return None;
    }
    let _ = eps; // carried by the program's own splat; read by the door's eps reader
    Some(HeadNormSandwich {
        split: node,
        norm,
        flatten,
        head_dim: *head_dim,
    })
}

/// The node whose OUTPUT is `t`, if exactly one — the producer a chain-walk follows
/// upward. `None` for a source (no producer) or a tensor two nodes write (not a chain).
fn producer_of<F: RopeForm>(
    ir: &SubtileIR<F>,
    t: scratchy_subtile::subtile_ir::TensorId,
) -> Option<&SubtileNode<F>> {
    let mut it = ir.nodes.iter().filter(|n| n.output.tensor == t);
    let p = it.next()?;
    it.next().is_none().then_some(p)
}

/// Whether `node` is the MIDDLE (norm) of a sandwich matched at its own split — the
/// node whose program is emitted there and whose own lowering must therefore be NOTHING.
fn is_sandwich_norm<F: RopeForm>(node: &SubtileNode<F>, ir: &SubtileIR<F>) -> bool {
    let Some(x) = node.inputs.first() else {
        return false;
    };
    producer_of(ir, x.tensor).is_some_and(|split| {
        sandwich_at_split(split, ir).is_some_and(|s| s.norm.id == node.id)
    })
}

/// Whether `node` is the TAIL (flatten-back) of a sandwich — same derivation, one hop up.
fn is_sandwich_flatten<F: RopeForm>(node: &SubtileNode<F>, ir: &SubtileIR<F>) -> bool {
    let Some(x) = node.inputs.first() else {
        return false;
    };
    producer_of(ir, x.tensor).is_some_and(|norm| is_sandwich_norm(norm, ir))
}

/// Lower the per-head norm SANDWICH to ONE program — the norm applied to head windows of
/// the ORIGINAL tensor, both reshapes deleted. [`HeadNormSandwich`]'s docs carry the
/// proof; this builds the program: parameters are the split's SOURCE `x`, the gamma row
/// (gained form only) and the flatten's OUTPUT. Per head `h`: load `x[:, h·D..(h+1)·D]`,
/// the f32 mean-of-squares chain over `[m, D]`, and the terminal multiply writing the
/// output's head window. The door (`windowed_rmsnorm` in `lower_ktir_to_superdsc`)
/// derives every extent from these views.
fn lower_sandwich_node<F: RopeForm>(
    sandwich: &HeadNormSandwich<'_, F>,
    ir: &SubtileIR<F>,
    _sym_id_base: &mut i64,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let HeadNormSandwich {
        split,
        norm,
        flatten,
        head_dim,
    } = sandwich;
    let x_r = &split.inputs[0];
    let out_r = &flatten.output;
    let gained = matches!(norm.op, SubOp::RmsNorm { .. });
    let eps = match &norm.op {
        SubOp::RmsNorm { eps, .. } | SubOp::RmsNormUnit { eps } => *eps,
        _ => unreachable!("the matcher already proved which norm this is"),
    };
    let gain_offset = match &norm.op {
        SubOp::RmsNorm {
            gain: scratchy_subtile::subtile_ir::GainConvention::Scale,
            ..
        } => 0.0,
        SubOp::RmsNorm {
            gain: scratchy_subtile::subtile_ir::GainConvention::OnePlusScale,
            ..
        } => 1.0,
        SubOp::RmsNormUnit { .. } => 0.0,
        _ => unreachable!(),
    };
    let mut st = KtirFunc::new(ir);
    st.windowed_rmsnorm(x_r, &norm.inputs.get(1), out_r, *head_dim, eps, gain_offset);
    let name = Arena::global().str(format!("rmsnorm_s{}", split.id.index()));
    let k = st.finish_shaped(
        name,
        ktir_superdsc::ktir_node::Program::WindowedRmsNorm { unit: !gained },
    );
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
        RopeTensors {
            x_t: node.inputs[0].tensor,
            cos_t: node.inputs[1].tensor,
            sin_t: node.inputs[2].tensor,
            out_t: node.output.tensor,
        },
        RopeGeometry {
            cols: total,
            hd: HD,
            rows: node.output.region.rows.len,
            // The cos/sin tables are pre-tiled to the rope's full width by the host, so position
            // `ri`'s row starts at `ri · tbl_cols` — the table's OWN column count, not the head dim.
            tbl_cols: node.inputs[1].region.cols.len,
        },
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
    attn: LowerAttn<'_, F>,
    // The geometry witness the door minted, carrying the GQA divisibility proof. Every head count
    // and head dim this function uses is read off it, so none of them is a value the node handed
    // over and there is nothing here to check against the consts.
    geom: scratchy_subtile::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
) -> Result<Vec<EmittedOp>, SuperDscError> {
    let LowerAttn {
        node,
        ir,
        cap,
        active_cap,
        // TRUE when this bundle's query rows are separate requests (a batched decode) rather than
        // consecutive positions of one sequence (a prefill chunk). Only the KV writes care: a
        // prompt's rows share a page and differ in slot, requests differ in both.
        rows_are_requests,
        sym_id_base: _sym_id_base,
        layout,
    } = attn;
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

/// The ROW-LOCAL op set: ops whose output row `r` depends only on input row `r` (plus weights and
/// per-row tables). The prefill lm-head fold is sound only inside this set — see
/// [`tensor_feeds_result_through_row_local_ops`].
fn row_local_tail_op<F: RopeForm>(op: &SubOp<F>) -> bool {
    use SubOp::*;
    matches!(
        op,
        MatmulTile { .. }
            | SumReduce { .. }
            | Elementwise(_)
            | SiluMul
            | ScalarMul { .. }
            | RmsNorm { .. }
            | RmsNormReduce { .. }
            | RmsNormApply { .. }
            | RmsNormUnit { .. }
            | TanhSoftCap { .. }
            | ScalarWeightMul
    )
}

/// Does tensor `t` feed `ir.result` through ROW-LOCAL consumers only (or is it the result)?
///
/// The prefill lm-head fold must fire ONLY on the result's own tail. The width test alone
/// (`cols == result_cols`) was falsified by the gemma-4 parity fixture (vocab 512): its
/// GLOBAL-layer q projection is also 512 cols (4 heads × hd 128), so the fold fired there too —
/// the q matmul ran at m=1, only row 0 of the roped Q was ever real (the last prompt token's,
/// rotated at position 0 = identity), attention saw 30 zero rows, and the first sampled token was
/// garbage while every width-128 tensor stayed fp16-clean. A width COLLISION is not an identity.
///
/// What separates the real tail: its rows die at the result (the host reads row 0 only), and
/// everything between it and the result is ROW-LOCAL — so computing one row and placing it at
/// row 0 loses nothing. The q projection fails that: its output passes through a norm, a rope,
/// and an ATTENTION (row-mixing — output row `i` reads K/V rows `0..=i`), so every one of its
/// rows is live and folding to one row corrupts the layer. Rope is excluded from the row-local
/// set deliberately: `RopeAppend` also writes the paged KV cache, a consumer the prefill graph
/// cannot see.
fn tensor_feeds_result_through_row_local_ops<F: RopeForm>(
    ir: &SubtileIR<F>,
    t: scratchy_subtile::subtile_ir::TensorId,
) -> bool {
    if t == ir.result {
        return true;
    }
    // Every consumer of `t` must be a row-local op whose own output keeps the property — one
    // row-mixing reader anywhere in the chain makes `t`'s other rows live and the fold unsound.
    let mut any_reader = false;
    for n in &ir.nodes {
        if !n.inputs.iter().any(|i| i.tensor == t) {
            continue;
        }
        any_reader = true;
        if !row_local_tail_op(&n.op)
            || !tensor_feeds_result_through_row_local_ops(ir, n.output.tensor)
        {
            return false;
        }
    }
    any_reader
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

/// ⭐⭐⭐ THE BUNDLE'S ATTENTION PARAMETER, READ WHERE MAIN READ IT.
///
/// `ibm/main`'s `lower_one_node` lowered each node during the tape walk, so `rows_are_requests` was
/// its own walk parameter. This split lowers KTIR → SuperDSC one pass later, per BUNDLE, so the fact
/// is read HERE — off this walk's parameters — and travels to the door as
/// [`crate::ktir_superdsc_door::BundleAttnParams`], an argument of the call.
///
/// ⛔ THE GEOMETRY IS NO LONGER A BUNDLE FACT, AND GEMMA-4 IS WHY. It used to be read off the
/// graph's first `AttnDecode` with a refusal if two disagreed — sound only while one expansion had
/// one attention class. Gemma-4's layers alternate a sliding class (nqh=16, nkvh=8, hd=256) with a
/// global one (nqh=16, nkvh=1, hd=512), so one graph carries TWO; the door now mints each attention
/// program's own geometry off the program's views (`ktir_superdsc_door::attn`), which is where
/// every other program-stated fact (scale, `mq`, the swept extent) was already read.
///
/// `None` when the graph has no attention node: there is then nothing for the row kind to be a fact
/// of, and an attention program arriving at the door without it is that door's own build error.
pub(crate) fn attn_bundle_params<F: RopeForm>(
    ir: &SubtileIR<F>,
    rows_are_requests: bool,
) -> Result<Option<crate::ktir_superdsc_door::BundleAttnParams>, SuperDscError> {
    let has_attn = ir
        .nodes
        .iter()
        .any(|n| matches!(n.op, SubOp::AttnDecode { .. }));
    Ok(has_attn.then_some(crate::ktir_superdsc_door::BundleAttnParams { rows_are_requests }))
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
        lower_attn_node::<F, NQH, NKVH, HD>(self, geom)
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
    //
    // ⛔ AND BY DATAFLOW: the node must be an ANCESTOR of `ir.result`. The width test alone was
    // falsified by the gemma-4 parity fixture (vocab 512), whose GLOBAL-layer q projection is ALSO
    // 512 cols (4 heads × hd 128) — the fold fired on every global layer's Q (extracted the last
    // row, ran it at m=1, roped at position 0), and the model's first sampled token was garbage
    // while every width-128 tensor stayed fp16-clean. A width COLLISION is not an identity: only
    // the tail's own row is dead, and only the result's ancestors are the tail.
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
        && !rows_are_requests
        && tensor_feeds_result_through_row_local_ops(ir, node.output.tensor);
    match &node.op {
        // The rest of the arch vocabulary. It reaches this emitter because the SHARED
        // front end expresses every op instead of asserting the unsupported ones away
        // in `lower_region` — which is the point: the IR carries the fact and the
        // TARGET says whether it has a kernel. Enumerated, never `_`, so adding a
        // SubOp is E0004 here rather than a surprise at emission.
        SubOp::TanhSoftCap { cap } => match lower_tanhsoftcap_node(node, ir, *cap, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::ScalarWeightMul => match lower_scalar_weight_mul_node(node, ir, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        // The GAINLESS form — gemma's per-head V/K norms. The same chain as `RmsNorm`
        // minus the gamma multiply, so it is the same 6-op decomposition with the
        // normalising multiply terminal. Unit gain means there is no GainConvention to
        // refuse here: nothing is applied.
        //
        // ⭐ A SANDWICH'S MIDDLE LOWERS TO NOTHING: its computation is the fused program
        // the split's arm already emitted (one windowed norm over all heads). The
        // self-check re-runs the same matcher from the split, so it agrees with the head's
        // decision in every walk — and any check that fails there falls through to the
        // ordinary un-fused lowering here, byte-identical to before this fusion existed.
        SubOp::RmsNormUnit { eps } => {
            if is_sandwich_norm(node, ir) {
                Ops(Vec::new())
            } else {
                match lower_rmsnorm_unit_node(node, ir, *eps, sym_id_base) {
                    Ok(v) => Ops(v),
                    Err(e) => Unhandled(e.0),
                }
            }
        }
        SubOp::GateSplit { .. }
        | SubOp::GateApply
        | SubOp::GateScale
        | SubOp::LoadPixels { .. }
        | SubOp::LoadPosEmbeds { .. }
        | SubOp::EmbeddingGather { .. }
        | SubOp::VisionRope
        | SubOp::VarlenAttention { .. }
        | SubOp::EncoderAttn { .. }
        | SubOp::GatedDeltaNet
        // The KV codec and sampled-rows expansions — still the blanket refusal; the
        // MoE rows above were split out only because the router side is now lowered.
        | SubOp::KvEncode { .. }
        | SubOp::KvStage { .. }
        | SubOp::RotateRows { .. }
        | SubOp::AttnPackedKv
        | SubOp::SampleRowsGather
        | SubOp::SampleRowsScatter
        | SubOp::AllRowsMatmul
        | SubOp::Mean => Unhandled(format!("{:?} has no SuperDSC kernel", node.op)),
        // ── THE ROUTER SIDE OF A MoE BLOCK, as decompositions of the programs this
        // emitter already ships. Every op here is a DENSE computation over `[m, ·]`
        // activation tiles — nothing about the router is expert-conditional — so each
        // one is exactly the rmsnorm / matmul / scalar-multiply / row-softmax chain
        // some dense model already lowers, and the producers below say which. The
        // EXPERT side (Sort/Matmul/GatedAct/Unsort/Combine) stays `Unhandled`: those
        // ops are pair-row permutations no dense program states, and binding them is
        // the topk/gather vocabulary the vendor `topk.ddl` ops carry — a separate
        // worklist item, refused by name until then.
        SubOp::RouterNorm { eps, .. } => {
            match lower_router_norm_node(node, ir, *eps, sym_id_base) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::RouterLogits { .. } => match lower_router_logits_node(node, ir, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::RouteScale { scale } => match lower_route_scale_node(node, ir, *scale, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        // The router's row softmax/renorm — the rmsnorm's reduce/broadcast structure
        // with the softmax's primitives in it (`max`/`exp`/`sum`/`realdiv`), all of
        // them DDL primitives already live on card.
        SubOp::RouteSoftmax => match lower_route_softmax_node(node, ir, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::RouteRenorm => match lower_route_renorm_node(node, ir, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        // ── THE ROUTER'S INDEX CHAIN — argsort/topk/gather/scale, the compare/
        // reduce/gather programs above. Each is numerically identical to its
        // metal counterpart (see the `lower_route_*` docs); the CARD track still
        // refuses them by name in `ktir_superdsc_door` until the vendor `topk.ddl`
        // vocabulary is bound, which is that track's work, not this one's.
        SubOp::RouteArgsort => match lower_route_argsort_node(node, ir, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::RouteTopK { k } => match lower_route_topk_node(node, ir, k.get(), sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::RouteGatherScores => {
            match lower_route_gather_scores_node(node, ir, sym_id_base) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::RouteExpertScale { .. } => {
            match lower_route_expert_scale_node(node, ir, sym_id_base) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        // ── THE EXPERT LAYOUT CHAIN (gathered/decode semantics) — the sort's
        // materialized copy, the unsort's identity, the combine's f32 fma sum.
        // Numerically identical to metal's GATHERED bake (which emits nothing
        // for sort/unsort and `moe_weighted_sum` for combine); the card track
        // refuses them by name until the vendor sort/mask vocabulary is bound.
        SubOp::ExpertSort { k, .. } => {
            match lower_expert_sort_node(node, ir, k.get(), sym_id_base) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::ExpertMatmul { k, .. } => {
            match lower_expert_matmul_node(node, ir, k.get(), sym_id_base) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::ExpertGatedAct { act } => {
            match lower_expert_gated_act_node(node, ir, *act, sym_id_base) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        SubOp::ExpertUnsort => match lower_expert_unsort_node(node, ir, sym_id_base) {
            Ok(v) => Ops(v),
            Err(e) => Unhandled(e.0),
        },
        SubOp::ExpertCombine { .. } => {
            match lower_expert_combine_node(node, ir, sym_id_base) {
                Ok(v) => Ops(v),
                Err(e) => Unhandled(e.0),
            }
        }
        // ⭐ THE SANDWICH'S TWO ENDS — the split (head) is where the fused program is
        // emitted; the flatten-back (tail) self-identifies as fused-away and emits NOTHING.
        // The middle (norm) arms do the same. The match is dataflow-based (see
        // `HeadNormSandwich`), so it is a pure function of the graph — the unrolled walk,
        // `graph_wiring` and the re-rolled tape walk all reach the same answer at the same
        // nodes, and any failed check falls back to the byte-identical un-fused lowering.
        SubOp::Reshape { .. } => {
            if is_sandwich_flatten(node, ir) {
                Ops(Vec::new())
            } else {
                match sandwich_at_split(node, ir) {
                    Some(sandwich) => match lower_sandwich_node(&sandwich, ir, sym_id_base) {
                        Ok(v) => Ops(v),
                        Err(e) => Unhandled(e.0),
                    },
                    None => match lower_reshape_node(node, ir, sym_id_base) {
                        Ok(v) => Ops(v),
                        Err(e) => Unhandled(e.0),
                    },
                }
            }
        }
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
        SubOp::SumReduce { .. } => Unhandled(format!(
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
        // ⭐ THE GAIN CONVENTION IS THE OFFSET THE CHAIN ADDS TO THE LOADED ROW — one kernel, two
        // offsets. `Scale` is the stored-gain form llama/granite use (`+0.0`); the gemma-class
        // (1 + w) convention is the SAME chain with `+1.0` applied to the gamma row inside the
        // program, never folded into the bound weight (folding would make the loaded gain disagree
        // with the checkpoint — the same reason the bridge CARRIES the convention instead of
        // folding it at `to_wavefront`). Before this, the (1 + w) form was refused BY NAME because
        // the program had no way to say the offset; now it does.
        SubOp::RmsNorm { eps, gain } => {
            // ⭐ A SANDWICH'S MIDDLE LOWERS TO NOTHING — same law as `RmsNormUnit`'s arm.
            if is_sandwich_norm(node, ir) {
                Ops(Vec::new())
            } else {
                match lower_rmsnorm_node(node, ir, *eps, *gain, sym_id_base) {
                    Ok(v) => Ops(v),
                    Err(e) => Unhandled(e.0),
                }
            }
        }
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
        SubOp::ScalarMul { scale } => match lower_scalarmul_node(node, ir, *scale, sym_id_base) {
            Ok(o) => Ops(vec![o]),
            Err(e) => Unhandled(e.0),
        },
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
    let bundle_layout = compute_bundle_layout(ir, weight_ids, rows_are_requests, &[])?;
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
    /// ⭐ THE ROPE-P CLASS SET — this tape's DISTINCT rope head dims, sorted descending; index `i`
    /// ↔ `rope_p_class_tid(i)`. The macro bakes this into the `Wiring` so the worker's load-time
    /// bind builds one `[hd,hd]` P table PER CLASS (a hybrid model's single composite P is
    /// impossible). Computed by the shared `rope_p_class_hds` helper off the same `ir` the
    /// placement pass reads, so the two lists are one list.
    pub rope_class_hds: Vec<u32>,
    /// ⭐ THE IDENTITY CLASS SET — this tape's DISTINCT `AttnDecode` head dims, sorted descending;
    /// index `i` ↔ `identity_class_tid(i)`. Same law as [`Self::rope_class_hds`], for the GQA
    /// krep/vrep and cachewr matmul-by-identity tables.
    pub attn_class_hds: Vec<u32>,
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
    // ⛔ A REGISTRY, NOT A PLACEMENT — the wiring reads ONE thing off the layout: the scalar
    // registry [`lower_attn_node`]'s desync check resolves its constants through. The placement
    // pass [`compute_bundle_layout`] computes here would be a THROWAWAY (the shipped layout is
    // the re-rolled bake's), and it is the CARD's budget pass: it refuses a weight segment past
    // one 16 GiB region with no layer classes to bank on — MEASURED, the 26b MoE under the fp8
    // preset wired fine through the banked re-rolled bake and then panicked HERE on a layout
    // nobody reads. The registry is a pure function of `ir.nodes`, so that is what this builds.
    let _ = weight_ids;
    let layout = BundleLayout {
        scalarmul_scales: scalar_registry(ir),
        // ⭐ THE CLASS REGISTRIES, here too — the door resolves a rope/attn program's P/identity
        // class through these, and this pass lowers through the door. Same shared helper, same
        // tape, so the wiring's answer and the placement pass's cannot disagree.
        rope_class_hds: ktir_superdsc::reserved_tids::rope_p_class_hds(
            ir.nodes.iter().filter_map(|n| match &n.op {
                SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
                    Some(head_dim.get())
                }
                _ => None,
            }),
        ),
        attn_class_hds: ktir_superdsc::reserved_tids::rope_p_class_hds(
            ir.nodes.iter().filter_map(|n| match &n.op {
                SubOp::AttnDecode { geom, .. } => Some(geom.hd().get()),
                _ => None,
            }),
        ),
        ..Default::default()
    };
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
                let regs = ktir_superdsc::emit::lower_ktir_to_superdsc::regions(&k)
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
        rope_class_hds: layout.rope_class_hds.clone(),
        attn_class_hds: layout.attn_class_hds.clone(),
    })
}
/// RE-ROLLED tape-driven SuperDSC lowering — the mirror of `lower_subtile_tape_to_tk_tape`
/// for SuperDSC (the fix for the 40-min unrolled-bundle compile). Walks the rerolled
/// `tape` (`subtile_tape::reroll_subtile_tape`): pre-loop Computes → `prefix`; the
/// `OpenLoop(Const(iters))`..`CloseLoop` body → `body` (lowered ONCE via the shared
/// `lower_one_node`); post-loop Computes → `suffix`. Collects the per-layer tid map
/// (`Compute::per_layer_out` for outputs + `ComputeInput::External::per_layer` for
/// weights) for the executor's per-iteration address advance.
/// Collect a matmul KERNEL weight (layer-0 tid, in=k, out=n) for the device re-tile manifest.
/// w = inputs[1] is `[k,n]=[in,out]` row-major; the PT array needs the device tile layout
/// `[out/64, in, 64]`. Arity-3 fp8 W8A8 matmuls MUST be collected too: their weight (inputs[1];
/// inputs[2] is w_scale) needs the 1-byte packed RetileDescriptor from the `fp8_weight_tids`
/// branch. in_k/out derive identically (inputs[0].cols=k, output.cols=n); the KSPLIT branch
/// (out>16384) never fires for fp8 projs. Excluding arity-3 staged them FLAT → scrambled weights
/// → garbage (`scr chat` incoherent); root-caused + on-card confirmed 2026-07-16.
///
/// Per segment now: the walk calls this once per node it emits, and the per-layer copies are
/// attached later by the manifest loop (`per_layer.get(w_tid)`).
fn collect_kernel0<F: RopeForm>(n: &SubtileNode<F>, out: &mut Vec<(u32, u32, u32)>) {
    if matches!(n.op, SubOp::MatmulTile { .. }) && (n.inputs.len() == 2 || n.inputs.len() == 3) {
        out.push((
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
    }
    // ⭐ THE EXPERT BANKS — the MoE projections' stacked `[E·n, in]` fp8 codes. The wavefront
    // wires `ExpertMatmul` as `[rows, pairs, w, s]`, so the bank is `inputs[2]` and its OWN
    // region already carries the full stacked extents (`[E·n, in]`, the launch source's declared
    // shape) — no `n`/`E` arithmetic here, the region IS the truth. The scale `inputs[3]`
    // `[E·n, 1]` stays FLAT: a bf16 scale bank is read by the dequant's per-channel row, not by
    // a matmul kernel, so no retile manifest entry belongs to it.
    if matches!(n.op, SubOp::ExpertMatmul { .. }) && n.inputs.len() == 4 {
        out.push((
            n.inputs[2].tensor.index() as u32,
            n.inputs[2].region.cols.len,
            n.inputs[2].region.rows.len,
        ));
    }
}

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
    // ONE MAP PER ATTENTION CLASS: a class-split roll's loops carry different-length per-layer
    // tables, and the layout needs to know which class a tensor belongs to before it can pack.
    let per_layer_ext = per_layer_external_tids(tape);
    // The layout's budget passes (spill/bank) take the per-loop maps directly: the spill reasons
    // over WHICH tids are per-layer at all (flattened internally), and the BANK needs the class
    // boundaries to know which (class, iteration) launches must stay in one bank.
    let mut bundle_layout =
        compute_bundle_layout(ir, weight_ids, rows_are_requests, &per_layer_ext)?;
    // ── THE TAPE'S SEGMENTS: prefix, ONE RANGE PER LOOP (an attention class's body), suffix ──
    // The rolled tape is `prefix; [OpenLoop(body₀) CloseLoop]…[OpenLoop(bodyₙ) CloseLoop]; suffix`
    // — sibling loops, never nested (the TapeBuilder typestate forbids nesting). One `(range,
    // iters)` per loop, in tape order; the walk below lowers each separately.
    let instrs = tape.instrs();
    let mut loops: Vec<(std::ops::Range<usize>, u32)> = Vec::new();
    {
        let mut open: Option<(usize, u32)> = None;
        for (i, instr) in instrs.iter().enumerate() {
            match instr {
                Instr::OpenLoop { bound, .. } => {
                    if open.is_some() {
                        return Err(SuperDscError(
                            "reroll-superdsc: a nested layer loop — the TapeBuilder typestate \
                             forbids nesting, so the tape is not one this walk built"
                                .into(),
                        ));
                    }
                    let it = match bound {
                        LoopBound::Const(it) => *it,
                        LoopBound::Runtime(_) => {
                            return Err(SuperDscError(
                                "reroll-superdsc: a runtime-bounded layer loop is unsupported \
                                 (the decode layer count is a compile-time Const)"
                                    .into(),
                            ));
                        }
                    };
                    if it == 0 {
                        return Err(SuperDscError(
                            "reroll-superdsc: a zero-iteration layer loop — reroll found a \
                             repeating body of no layers"
                                .into(),
                        ));
                    }
                    open = Some((i, it));
                }
                Instr::CloseLoop { .. } => {
                    let Some((o, it)) = open.take() else {
                        return Err(SuperDscError(
                            "reroll-superdsc: a CloseLoop with no OpenLoop — unbalanced brackets"
                                .into(),
                        ));
                    };
                    loops.push((o + 1..i, it));
                }
                _ => {}
            }
        }
        if open.is_some() {
            return Err(SuperDscError(
                "reroll-superdsc: an OpenLoop with no CloseLoop — unbalanced brackets".into(),
            ));
        }
    }
    if loops.is_empty() {
        return Err(SuperDscError(
            "reroll-superdsc: no layer loop in the rerolled tape (no OpenLoop/CloseLoop) — \
             reroll_subtile_tape found no repeating body"
                .into(),
        ));
    }
    let prefix_end = loops[0].0.start - 1;
    let suffix_start = loops[loops.len() - 1].0.end + 1;
    // ⛔ `iters` per body, from the loop bounds — NOT the count of instrs in the range.
    debug_assert_eq!(loops.len(), per_layer_ext.len()); // one map per sibling loop, by construction
    // ── The walk: prefix, each body, suffix, with per-segment lowering context ──
    // The residual-stream aliasing pre-pass first: thread the loop-carried hidden IN-PLACE, no copy.
    // Pre-scan EVERY body for hidden_in (its first node's input[0]) + hidden_out (its last node's
    // output) and alias them ALL onto the FIRST body's hidden_in placement → every launch of every
    // class reads+writes ONE resident buffer, so the residual threads with NO host round-trip and NO
    // device copy. WAR-safe: within a layer hidden_in's last read (computing h1 = h_in+attn)
    // precedes hidden_out's write (the final h_out = h1+mlp add); across launches the shared buffer
    // carries the residual. The suffix's input aliases the same buffer (the body→suffix seam).
    {
        let mut hidden_placement: Option<crate::lower_subtile_tape_to_superdsc::TensorPlacement> =
            None;
        let mut first_suffix_node: Option<u32> = None;
        for instr in &instrs[suffix_start..] {
            if let Instr::Compute { node, .. } = instr {
                first_suffix_node = Some(node.index() as u32);
                break;
            }
        }
        for (range, _) in &loops {
            let first = instrs[range.clone()]
                .iter()
                .find_map(|i| match i {
                    Instr::Compute { node, .. } => Some(*node),
                    _ => None,
                })
                .expect("a loop body holds at least one Compute");
            let last = instrs[range.clone()]
                .iter()
                .rev()
                .find_map(|i| match i {
                    Instr::Compute { node, .. } => Some(*node),
                    _ => None,
                })
                .expect("a loop body holds at least one Compute");
            let hin = ir.nodes[first.index()].inputs[0].tensor.index() as u32;
            let hout = ir.nodes[last.index()].output.tensor.index() as u32;
            let p = hidden_placement
                .take()
                .or_else(|| bundle_layout.placements.get(&hin).cloned())
                .expect("the first body's hidden-in has a layout placement");
            // hidden_out (last body op) → the shared hidden buffer: per-launch residual threads in
            // place; the NEXT body's hidden_in aliases the same buffer, so the cross-class seam
            // threads in place too.
            bundle_layout.placements.insert(hout, p);
            bundle_layout.placements.insert(hin, p);
            hidden_placement = Some(p);
        }
        // suffix_in (first suffix op's input[0]) → the same buffer.
        if let Some(s) = first_suffix_node {
            let sin = ir.nodes[s as usize].inputs[0].tensor.index() as u32;
            if let Some(p) = hidden_placement {
                bundle_layout.placements.insert(sin, p);
            }
        }
    }
    let layout = Some(&bundle_layout);
    let mut sym_id_base: i64 = 0;
    let mut prefix: Vec<EmittedOp> = Vec::new();
    let mut suffix: Vec<EmittedOp> = Vec::new();
    // fp8 activation-quantize dedup (see `lower_matmul_node`): keyed on the activation `t{id}`. Within
    // a once-walked body, q/k/v (same rms₁ tid) share ONE quant; distinct-tid activations never
    // false-share, and the reusing matmul lands in the SAME segment as the quant it reuses (shared
    // tid ⇒ same segment). RESET PER BODY: two classes' bodies are different programs over different
    // representative tids, so a stale entry from class 0 could only suppress a legitimate quant in
    // class 1 by tid collision — there is none, but the reset states the scope.
    let mut fp8_quantized: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut unhandled: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    // ⭐ SEEDED FROM THE PRE-PASS RATHER THAN REBUILT. `per_layer_external_tids` already read every
    // per-layer WEIGHT/KV class off this same tape — it had to, because `compute_bundle_layout` needs
    // the layer boundaries to decide weight BANKS, and that runs before any op is emitted. The walk
    // below then adds only what the pre-pass cannot see (node outputs, the resident Kᵀ), and its own
    // `or_insert_with` is a no-op for anything already here.
    // ⭐ SEEDED FROM THE PRE-PASS RATHER THAN REBUILT (flattened — the walk's `or_insert`s below
    // are no-ops for anything already here). A table's KEY is its class's layer-0 tid; a table's
    // LENGTH is its class's `iters`.
    let mut per_layer: std::collections::BTreeMap<u32, Vec<u32>> = per_layer_ext
        .iter()
        .flat_map(|m| m.iter())
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    // Which weight BANKS each launch group addresses — one set per GROUP, where a group is now the
    // prefix (index 0), each body (1 + body index), or the suffix (last). A launch has ONE base per
    // segment, so a group may address exactly one; checked after the walk.
    let n_groups = loops.len() + 2;
    let mut group_weight_banks: Vec<std::collections::BTreeSet<u32>> =
        vec![Default::default(); n_groups];
    // First SUFFIX Compute node — its input[0] is the residual the suffix reads.
    let mut first_suffix_node: Option<u32> = None;
    // Matmul KERNEL weights (layer-0 tid, in=k, out=n) for the device re-tile manifest.
    let mut kernel0: Vec<(u32, u32, u32)> = Vec::new();
    // ⭐ ONE `group` PER SEGMENT: 0 = prefix, 1..=n = body i, n+1 = suffix. The closure takes the
    // fp8-quantize dedup set BY PARAMETER (not by capture) so the walk can CLEAR it between bodies
    // — the reset is the scope statement, and a captured set would forbid the clear outright.
    let mut emit_seg = |group: usize,
                        n: &SubtileNode<F>,
                        fp8_quantized: &mut std::collections::HashSet<String>,
                        ops_out: &mut Vec<EmittedOp>|
     -> Result<(), SuperDscError> {
        for r in &n.inputs {
            let tid = r.tensor.index() as u32;
            if let Some(p) = bundle_layout.placements.get(&tid)
                && matches!(p.role, SegRole::Weight)
            {
                group_weight_banks[group].insert(p.bank);
            }
        }
        match lower_one_node(
            n,
            ir,
            active_cap,
            rows_are_requests,
            &mut sym_id_base,
            layout,
            fp8_quantized,
        ) {
            NodeLowering::Ops(v) => {
                ops_out.extend(v);
                Ok(())
            }
            NodeLowering::Unhandled(s) => {
                if matches!(n.op, SubOp::MatmulTile { .. }) {
                    Err(SuperDscError(s))
                } else {
                    unhandled.insert(s);
                    Ok(())
                }
            }
            NodeLowering::HostRouted(_) => Ok(()),
        }
    };
    // ── Prefix ──
    for instr in &instrs[..prefix_end] {
        if let Instr::Compute { node, .. } = instr {
            let n = &ir.nodes[node.index()];
            collect_kernel0(n, &mut kernel0);
            emit_seg(0, n, &mut fp8_quantized, &mut prefix)?;
        }
    }
    // ── Bodies: one RolledBody per loop, its iters, its layer ids, its hidden tids ──
    let mut bodies: Vec<crate::lower_subtile_tape_to_superdsc::RolledBody> = Vec::new();
    // Each body's PER-LAYER REPRESENTATIVE tids — every External table + per-layer node output
    // this body's own instrs name, plus its kct. `body_of_tid` (the launch-table guard) uses this
    // to know which class a `per_layer` table belongs to; a table's KEY is the class's layer-0 tid,
    // and only the body that READS (or writes) that tid can own it.
    let mut body_representatives: Vec<Vec<u32>> = Vec::with_capacity(loops.len());
    for (bi, (range, iters)) in loops.iter().enumerate() {
        let mut ops: Vec<EmittedOp> = Vec::new();
        let mut first_node: Option<u32> = None;
        let mut last_node: Option<u32> = None;
        let mut representatives: Vec<u32> = Vec::new();
        for instr in &instrs[range.clone()] {
            let Instr::Compute {
                node,
                inputs,
                per_layer_out,
                ..
            } = instr
            else {
                continue;
            };
            let n = &ir.nodes[node.index()];
            collect_kernel0(n, &mut kernel0);
            if first_node.is_none() {
                first_node = Some(node.index() as u32);
            }
            last_node = Some(node.index() as u32);
            // Per-layer OUTPUT tids (the node's output tensor in each layer copy of THIS class).
            if *iters > 1 && per_layer_out.len() as u32 == *iters {
                let outs: Vec<u32> = per_layer_out
                    .iter()
                    .map(|nid| ir.nodes[nid.index()].output.tensor.index() as u32)
                    .collect();
                representatives.push(n.output.tensor.index() as u32);
                per_layer
                    .entry(n.output.tensor.index() as u32)
                    .or_insert(outs);
            }
            // Per-layer WEIGHT/external tids of THIS class.
            for ci in inputs.iter() {
                if let ComputeInput::External {
                    tensor,
                    per_layer: pl,
                    ..
                } = ci
                    && *iters > 1
                    && pl.len() as u32 == *iters
                {
                    representatives.push(tensor.index() as u32);
                    per_layer
                        .entry(tensor.index() as u32)
                        .or_insert_with(|| pl.iter().map(|t| t.index() as u32).collect());
                }
            }
            // RESIDENT kct per-layer registration (kill-the-restickify residency): the score reads a
            // per-layer RESIDENT Kᵀ kernel `kct_resident_tid(k_id)`. It's neither an External nor a
            // node output, so the loops above never add it — insert it manually: map each layer's
            // K-cache source tid k_Lv → kct_resident_tid(k_Lv), keyed under the layer-0 kct tid.
            // REQUIRED so the launch-table guard validates kct's per-layer offset matches k/v — the
            // executor shifts seg2 by the launch's kv_off for every op, so a kct packed out of step
            // with k/v would SILENTLY read the wrong layer (runtime-errors-need-compile-time-checks).
            if let SubOp::AttnDecode { layout: kv, .. } = &n.op
                && *iters > 1
                && let Some(k_layers) = per_layer.get(&(kv.cache_tensor().index() as u32)).cloned()
                && k_layers.len() as u32 == *iters
            {
                let kct_tids: Vec<u32> = k_layers
                    .iter()
                    .map(|&k_lv| kct_resident_tid(k_lv))
                    .collect();
                representatives.push(kct_resident_tid(kv.cache_tensor().index() as u32));
                per_layer
                    .entry(kct_resident_tid(kv.cache_tensor().index() as u32))
                    .or_insert(kct_tids);
            }
            emit_seg(1 + bi, n, &mut fp8_quantized, &mut ops)?;
        }
        let first = first_node.unwrap_or(u32::MAX);
        let last = last_node.unwrap_or(u32::MAX);
        // The body's hidden in/out tids.
        let hidden_in_tid = if first == u32::MAX {
            u32::MAX
        } else {
            ir.nodes[first as usize].inputs[0].tensor.index() as u32
        };
        let hidden_out_tid = if last == u32::MAX {
            u32::MAX
        } else {
            ir.nodes[last as usize].output.tensor.index() as u32
        };
        // ⭐ THE CLASS'S LAYER IDS, from its OWN rope table: the body's RopeAppend node (the KV
        // writer — "a layer is named by its KV writer") carries `per_layer_out` naming this class's
        // rope nodes; `layer_index()` on each gives the ABSOLUTE layer id per class-relative
        // iteration. This is the only source of the true order, and it is the tape's own.
        let mut layer_ids: Vec<u32> = Vec::with_capacity(*iters as usize);
        for instr in &instrs[range.clone()] {
            if let Instr::Compute {
                node,
                per_layer_out,
                ..
            } = instr
                && matches!(ir.nodes[node.index()].op, SubOp::RopeAppend { .. })
                && per_layer_out.len() as u32 == *iters
            {
                layer_ids = per_layer_out
                    .iter()
                    .map(|nid| {
                        ir.nodes[nid.index()]
                            .op
                            .layer_index()
                            .expect("a RopeAppend names a layer")
                    })
                    .collect();
                break;
            }
        }
        if layer_ids.len() != *iters as usize {
            return Err(SuperDscError(format!(
                "reroll-superdsc: body {bi} has {} RopeAppend layer ids but its loop runs {} \
                 iterations — the layer cannot be named for every launch, so the true layer order \
                 is unknown and the launch table cannot be built",
                layer_ids.len(),
                iters
            )));
        }
        bodies.push(crate::lower_subtile_tape_to_superdsc::RolledBody {
            ops,
            iters: *iters,
            layer_ids,
            hidden_in_tid,
            hidden_out_tid,
        });
        body_representatives.push(representatives);
        fp8_quantized.clear();
    }
    // ── Suffix ──
    for instr in &instrs[suffix_start..] {
        if let Instr::Compute { node, .. } = instr {
            let n = &ir.nodes[node.index()];
            if first_suffix_node.is_none() {
                first_suffix_node = Some(node.index() as u32);
            }
            collect_kernel0(n, &mut kernel0);
            emit_seg(n_groups - 1, n, &mut fp8_quantized, &mut suffix)?;
        }
    }
    if !unhandled.is_empty() {
        return Err(SuperDscError(format!(
            "reroll-superdsc: {} unhandled op kind(s): [{}]",
            unhandled.len(),
            unhandled.into_iter().collect::<Vec<_>>().join(", "),
        )));
    }
    // ⭐ THE LAUNCH SEQUENCE, in TRUE LAYER ORDER. Each body names its layers (absolute ids); the
    // model's layers run in ascending id order, so one sort of `(body, iter, layer_id)` triples is
    // the whole sequence. A single-class body has `layer_ids == [0..iters)`, which yields exactly
    // the `for v in 0..iters` sequence the executor ran before classes existed.
    let mut order: Vec<(u8, u32, u32)> = bodies
        .iter()
        .enumerate()
        .flat_map(|(bi, b)| {
            b.layer_ids
                .iter()
                .enumerate()
                .map(move |(it, &id)| (bi as u8, it as u32, id))
        })
        .collect();
    let total_layers = order.len();
    order.sort_by_key(|&(_, _, id)| id);
    if let Some(w) = order.windows(2).find(|w| w[0].2 == w[1].2) {
        return Err(SuperDscError(format!(
            "reroll-superdsc: two launches name layer {} (bodies {} and {}) — a layer id appears \
             in two classes' rope tables, so neither owns it",
            w[0].2, w[0].0, w[1].0
        )));
    }
    let launches: Vec<(u8, u32)> = order.iter().map(|&(b, i, _)| (b, i)).collect();
    let launch_layer: Vec<u32> = order.iter().map(|&(_, _, l)| l).collect();
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
    //
    // ⛔ CARD ONLY, for exactly the reason its own doc gives: the one consumer that reads those
    // synthetics is `ktir_groups_via_superdsc`, which `emit_bundle_inner` calls only under
    // `spyre-hw`. The emulator executes the programs' SSA directly and addresses every REAL tensor
    // through the placements `compute_bundle_layout` already made — running the door here for it
    // would demand a SuperDSC descriptor body for every program kind before the emulator could run
    // ANY of them, which is the card's admission predicate applied to a device that has no
    // descriptors. The same device split codegen's own bake makes (`let card = cfg!(feature =
    // "spyre-hw")`).
    let attn_params = attn_bundle_params(ir, rows_are_requests)?;
    let body_op_lists: Vec<&Vec<EmittedOp>> = bodies.iter().map(|b| &b.ops).collect();
    // ⛔ AND THE GATE IS THE DEVICE, NOT A CONVENTION. A program the door
    // refuses BY NAME (the MoE expansion ops, whose SuperDSC bodies are the
    // card track's later work) is legal KTIR the emulator runs fine — an
    // unconditional declare pass would demand a SuperDSC body for every
    // program kind before `-Fspyre-emu` could run ANY bundle containing one,
    // which is the card's admission predicate applied to a device that has
    // no descriptors. MEASURED: the gemma-4-26b parity-tiny build panicked
    // here at `routeargsort_s24` under `-Fspyre-emu` while the same program
    // passed its emulator unit test one crate away.
    if cfg!(feature = "spyre-hw") {
        for ops in std::iter::once(&prefix)
            .chain(body_op_lists.iter().copied())
            .chain(std::iter::once(&suffix))
        {
            let mut declare_syms: i64 = 0;
            let mut declare_fp8: std::collections::HashSet<String> =
                std::collections::HashSet::new();
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
        // ⭐ THE EXPERT BANKS ARE fp8 TOO — every MoE projection's stacked `[E·n, in]` codes.
        // UNCONDITIONALLY: the spyre `SwitchGluExpertsLayer` loader covers the
        // fp8-dynamic-per-channel checkpoint ONLY (a dense/bf16 checkpoint bails at load, before
        // any staging), and the wavefront's `ExpertQuant` is declared for every ExpertMatmul —
        // so on this target an expert bank reaching the manifest is an fp8 bank by construction.
        // The input position differs per op: `inputs[1]` for a dense MatmulTile, `inputs[2]` for
        // the `[rows, pairs, w, s]` ExpertMatmul wiring.
        .filter(|n| {
            matches!(n.op, SubOp::MatmulTile { .. }) && n.inputs.len() == 3
                || matches!(n.op, SubOp::ExpertMatmul { .. }) && n.inputs.len() == 4
        })
        .map(|n| {
            let i = if matches!(n.op, SubOp::MatmulTile { .. }) { 1 } else { 2 };
            n.inputs[i].tensor.index() as u32
        })
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
            //
            // ⭐ THE EXPERT BANKS ARE THE SAME MAP AT `n = E·out`. The bank's host form is the
            // expert-major stack `[E·out, in]` — expert e's slab is rows `[e·out, (e+1)·out)`, i.e.
            // host elements `[e·out·in, (e+1)·out·in)` — and the device form this map resolves
            // (`host[o·k + i]`, `o ∈ [0, E·out)`) makes rows `o` and `o+1` DEVICE-ADJACENT within a
            // 64-row n-stick block and every slab a whole number of such blocks (E·out ≡ 0 mod 64
            // whenever out ≡ 0 mod 64, which the guard below states). So expert e's packed slab is
            // CONTIGUOUS at device elements `[e·out·in, (e+1)·out·in)` — no E-outermost axis is
            // needed and none is declared: the slab boundary the MoE gather copy reads (one entry =
            // one expert's whole packed slab) is a row boundary of this very map.
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
    // ══════════════════════════════════════════════════════════════════════════════════════════
    //  ⭐ THE LAUNCH TABLE — one `(w_bank, w_off, kv_off)` per layer, GENERATED from the
    //  placements and PROVEN against them, so the executor never re-derives an address from a
    //  stride.
    //
    //  The old contract was arithmetic: layer `v` at `(v / layers_per_bank, (v % lpb)·weight_stride)`
    //  and `v·kv_stride`. That formula is only correct while every layer has the SAME geometry —
    //  gemma-4's two attention classes do not (sliding 8×256 KV vs global 1×512, and weight blocks
    //  that differ by class), so the formula would become three formulae (one per class, plus a
    //  class base) that could disagree with the packing that actually happened. The table replaces
    //  all of them: for the launch of (body c, iteration j), the shift is read straight off class
    //  c's own placements, and the guard below holds EVERY per-layer tensor of EVERY class to the
    //  table — strictly stronger than the pairwise stride check it replaces, because it also
    //  validates the bank and the class base, per layer, not per consecutive pair.
    // ══════════════════════════════════════════════════════════════════════════════════════════
    let w_seg = SegRole::Weight.segment();
    let kv_seg = SegRole::Kv.segment();
    // Which body owns a `per_layer` table: its key (the class's layer-0 tid) was collected from
    // that body's own instrs — see `body_representatives`. Only that body's launches shift for it.
    let body_of_tid = |tid: u32| -> Option<usize> {
        body_representatives
            .iter()
            .position(|reps| reps.contains(&tid))
    };
    // ⛔⛔⛔ A PER-LAYER **WEIGHT** OUTSIDE THE STRIDED SEGMENT IS SILENT GARBAGE: a launch shifts
    // only seg{w_seg}'s base, so layer v would read layer 0's copy of this tensor for every layer —
    // fluent output, wrong model. [`spill_weight_tail`] may only move the NON-per-layer tail, and
    // this is what holds it to that.
    //
    // ⛔ KEYED ON THE **ROLE**, NOT THE SEGMENT. A per-layer INTERMEDIATE colored into a spill slot
    // is legitimate and must keep falling through — the `WeightOverflow = 5` attempt refused
    // exactly that case (per-layer intermediate t448 in seg5) and read it as proof no segment was
    // available, when the real defect was taking a COLOR.
    //
    // A per-layer tensor must also not change SEGMENT across its class's layers: the launch shift
    // is one segment-wide byte offset, so a tensor that moved would be read at a wrong address on
    // one side of the move.
    for (t0, tids) in &per_layer {
        if tids.len() < 2 {
            continue;
        }
        let Some(p0) = bundle_layout.placements.get(t0) else {
            continue;
        };
        if p0.segment != w_seg && p0.segment != kv_seg && matches!(p0.role, SegRole::Weight) {
            return Err(SuperDscError(format!(
                "reroll-superdsc: per-layer WEIGHT t{t0} is placed in seg{}, which no launch \
                 shifts — every one of the {} layers would read layer 0's copy. Only NON-per-layer \
                 weights (the final norm, the lm_head / tied embedding) may spill; see \
                 `spill_weight_tail`.",
                p0.segment,
                tids.len(),
            )));
        }
        // Segment stability is required ONLY of the tensors a launch SHIFTS — the weight and KV
        // classes. A per-layer INTERMEDIATE may be colored differently per layer (a seg5 spill
        // slot for one, seg3 for another): intermediates are re-bound per launch from the launch's
        // own placement, so no shift ever crosses the move. Refusing those is what broke the first
        // attempt at this guard — the old pairwise check `continue`d them, and so does this one.
        let shifted =
            (p0.segment == w_seg && matches!(p0.role, SegRole::Weight)) || p0.segment == kv_seg;
        if shifted {
            for t in tids {
                if let Some(p) = bundle_layout.placements.get(t)
                    && p.segment != p0.segment
                {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: per-layer tensor t{t0} changes segment ({} → {}) across \
                         its class's layers — the launch shift is one segment-wide offset, so one \
                         side of the move would be read at a wrong address",
                        p0.segment, p.segment
                    )));
                }
            }
        }
    }
    // ── GENERATE: for each launch (body c, iter j), the weight shift from ANY of c's per-layer
    //    weight tables and the KV shift from c's per-layer KV tables. A body with no per-layer
    //    weight/KV gets (0, 0, 0) — nothing to shift. The proof below then holds EVERY tensor of
    //    the class to the entry, so which table generated it cannot matter; a class whose tables
    //    DISAGREE is refused here with both readings named.
    let mut launch_w: Vec<(u8, u64)> = Vec::with_capacity(total_layers);
    let mut launch_kv: Vec<u64> = Vec::with_capacity(total_layers);
    for &(body, iter) in &launches {
        let mut w_entry: Option<(u8, u64)> = None;
        let mut kv_entry: Option<u64> = None;
        for (t0, tids) in &per_layer {
            if body_of_tid(*t0) != Some(body as usize) {
                continue;
            }
            let (Some(p0), Some(pj)) = (
                bundle_layout.placements.get(t0),
                bundle_layout.placements.get(&tids[iter as usize]),
            ) else {
                continue;
            };
            if p0.segment == w_seg && matches!(p0.role, SegRole::Weight) {
                let entry = (pj.bank as u8, pj.offset.wrapping_sub(p0.offset));
                if let Some(prev) = w_entry
                    && prev != entry
                {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: body {body}'s per-layer weights DISAGREE about layer \
                         {iter}'s launch shift ({prev:?} vs {entry:?}, generated from t{t0}) — the \
                         packing did not advance the class's weights together, so no one shift \
                         addresses them all"
                    )));
                }
                w_entry = Some(entry);
            } else if p0.segment == kv_seg {
                let entry = pj.offset.wrapping_sub(p0.offset);
                if let Some(prev) = kv_entry
                    && prev != entry
                {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: body {body}'s per-layer KV tensors DISAGREE about layer \
                         {iter}'s launch shift ({prev} vs {entry} B, generated from t{t0}) — k, v \
                         and Kᵀ must advance together or the score reads one layer's keys against \
                         another's values"
                    )));
                }
                kv_entry = Some(entry);
            }
        }
        launch_w.push(w_entry.unwrap_or((0, 0)));
        launch_kv.push(kv_entry.unwrap_or(0));
    }
    // ── PROVE, EVERY TENSOR, EVERY LAYER. The launch addresses
    //    `(bank: L.w_bank, offset: L.w_off + placement(t₀).offset)` — hold each per-layer weight
    //    to the placement its layer actually got, and each per-layer KV tensor likewise. This is
    //    the check the generator cannot make: generation read ONE tensor per segment per launch;
    //    this holds ALL of them.
    for (li, &(body, iter)) in launches.iter().enumerate() {
        let (wb, wo) = launch_w[li];
        let ko = launch_kv[li];
        for (t0, tids) in &per_layer {
            if body_of_tid(*t0) != Some(body as usize) {
                continue;
            }
            let (Some(p0), Some(pj)) = (
                bundle_layout.placements.get(t0),
                bundle_layout.placements.get(&tids[iter as usize]),
            ) else {
                continue;
            };
            if p0.segment == w_seg && matches!(p0.role, SegRole::Weight) {
                if pj.bank as u8 != wb || pj.offset != wo.wrapping_add(p0.offset) {
                    return Err(SuperDscError(format!(
                        "reroll-superdsc: per-layer weight t{tid} (layer {layer} of body {body}, \
                         class-relative {iter}) is placed at bank {bank} offset {off}, but the \
                         launch table addresses it at bank {wb} offset {want} (shift {wo} + \
                         layer-0 offset {base}). Every layer must sit where the launch looks, or \
                         that layer reads another layer's weights.",
                        tid = tids[iter as usize],
                        layer = launch_layer[li],
                        bank = pj.bank,
                        off = pj.offset,
                        want = wo.wrapping_add(p0.offset),
                        wo = wo,
                        base = p0.offset,
                        body = body,
                        iter = iter,
                        wb = wb,
                    )));
                }
            } else if p0.segment == kv_seg && pj.offset != ko.wrapping_add(p0.offset) {
                return Err(SuperDscError(format!(
                    "reroll-superdsc: per-layer KV tensor t{tid} (layer {layer} of body {body}) is \
                     placed at offset {off}, but the launch table addresses it at {want} (shift \
                     {ko} + layer-0 offset {base}). The score would read one layer's keys against \
                     another's values.",
                    tid = tids[iter as usize],
                    layer = launch_layer[li],
                    off = pj.offset,
                    want = ko.wrapping_add(p0.offset),
                    ko = ko,
                    base = p0.offset,
                    body = body,
                )));
            }
        }
    }
    // The PAGE stride: one pool page covers every layer of every class, so the stride is the
    // extent of the packed per-layer KV — the max end over every class's LAST layer placement
    // (classes are packed contiguously; per-class strides differ, which is exactly why this is a
    // max and not a count × stride). 0 when there is no paged KV.
    let kv_page_stride = per_layer
        .values()
        .filter_map(|tids| {
            let last = tids.last()?;
            let p = bundle_layout.placements.get(last)?;
            (p.segment == kv_seg).then(|| p.offset + p.size)
        })
        .max()
        .unwrap_or(0);
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
    let suffix_weight_bank = group_bank(n_groups - 1, "suffix")?;
    // Each BODY's weight reads must all sit in the bank its OWN launches bind — the launch-table
    // proof above already held its per-layer weights there, so a disagreement here means the body
    // also reads a NON-per-layer weight from another bank: the replication case, refused with its
    // own name rather than as a launch-table mismatch.
    for bi in 0..bodies.len() {
        let launch_bank = launches
            .iter()
            .zip(&launch_w)
            .find(|(l, _)| l.0 as usize == bi)
            .map(|(_, w)| w.0 as u32)
            .unwrap_or(0);
        if group_bank(1 + bi, "body")? != launch_bank {
            return Err(SuperDscError(format!(
                "reroll-superdsc: body {bi}'s program addresses weight bank(s) {:?}, but its \
                 launches bind bank {launch_bank} — a NON-per-layer weight read inside the layer \
                 loop would need replicating into every bank.",
                group_weight_banks[1 + bi],
            )));
        }
    }
    // The suffix's residual input (first suffix node's input[0]) — the executor threads the last
    // launch's hidden_out → suffix_in once after the launch sequence (the body→suffix seam). Same
    // input-ordering convention as a body's hidden_in (rmsnorm input[0] = x = the residual).
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
            "[SEGDUMP] layers={total_layers} total={:.3}GB segs(GB)=[{}]",
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
    Ok(RolledSuperDsc {
        prefix,
        bodies,
        suffix,
        layout: bundle_layout,
        per_layer,
        launches,
        launch_w,
        launch_kv,
        kv_page_stride,
        prefix_weight_bank,
        suffix_weight_bank,
        kv_request_stride,
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

/// The four tensor identities [`KtirFunc::rope`] reads and writes.
struct RopeTensors {
    x_t: TensorId,
    cos_t: TensorId,
    sin_t: TensorId,
    out_t: TensorId,
}

/// The shape facts [`KtirFunc::rope`] tiles over. See that method's doc for `tbl_cols`.
struct RopeGeometry {
    cols: u32,
    hd: u32,
    rows: u32,
    tbl_cols: u32,
}

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

    /// `tanh(x)`, the whole-tensor unary — `MathTanh` is a legal KTIR kind and
    /// `OpFunc::Tanh` a real DDL primitive (see `elementwise_op_func`), so unlike
    /// silu it needs no longhand decomposition. Used by the gelu polynomial below.
    fn tanh(&mut self, x: Ssa, dims: Vec<i64>) -> Ssa {
        self.unop(OpKind::MathTanh, x, dims)
    }

    /// `cap · tanh(x / cap)` — Gemma's final-logit soft cap, written longhand as
    /// ONE `arith.divf` by the splatted cap, ONE `math.tanh`, ONE `arith.mulf`
    /// by the same splat. `MathTanh` is a legal KTIR kind and `OpFunc::Tanh` a
    /// real DDL primitive, so the tanh needs no decomposition; the divide and
    /// the multiply are the device's `realdiv` and `multiply`.
    ///
    /// ⭐ THE CAP IS AN IMMEDIATE (`splat(cap)`), WHICH IS WHAT THE EMIT SIDE
    /// READS BACK — [`program_tanhsoftcap_cap`] resolves the `[1,1]` registry
    /// const from the DIVF's splatted operand, so the value this program
    /// interprets and the value the descriptor binds are one fact, the same
    /// contract [`Self::rmsnorm_unit`]'s epsilon holds through
    /// [`program_rmsnorm_eps`].
    fn tanhsoftcap(&mut self, x: Ssa, dims: Vec<i64>, cap: f64) -> Ssa {
        let capt = self.splat(cap, dims.clone());
        let scaled = self.binop(OpKind::ArithDivf, x, capt, dims.clone());
        let t = self.tanh(scaled, dims.clone());
        let capt2 = self.splat(cap, dims.clone());
        self.binop(OpKind::ArithMulf, t, capt2, dims)
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

    /// The tanh-approximation gelu `OpFunc::Gelu`'s DDL polynomial implements:
    /// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, spelled longhand for the interpreter the way
    /// silu is — KTIR has no gelu op, and the constants are immediates (`splat`), never registry
    /// slots.
    ///
    /// ⛔ THE TANH ARGUMENT IS CLAMPED TO ±15, the same clamp the metal target's own
    /// `gelu_tanh_f16` applies (activation.metal): `tanh` evaluates via `exp(2·inner)`,
    /// which overflows to Inf (→ NaN through the `1 + tanh` add) for large `inner`.
    /// `tanh` is already saturated to ±1 well before ±15, so the clamp is bit-exact
    /// there while killing the overflow. MEASURED on gemma-4-12b-it: the layer-0 MLP
    /// drove `inner` past f16's `exp` range, and the metal-offloaded map window
    /// NaN'd every gelu output — layer 1's k/v were all-NaN from the first forward
    /// on (nan=38912 of 38912) while the pure interpreter path stayed clean.
    ///
    /// ⛔ F16 OPS, NOT `f32_*`. The elementwise LX budget ([`EW_LX_ELEMS`], [`ew_live_tiles`]) is
    /// calibrated in f16 tiles — an `f32_splat` here made every constant a 4-byte-per-element tile,
    /// and gelma-4's `[22, 15360]` block died with `TensorSplat: LX capacity exceeded: 2027520 +
    /// 1351680 > 2097152` (one f16 operand + ONE f32 splat already over the pad). Silu's chain is
    /// f16 end to end for the same reason; f32 is for the variance reduction ([`Self::rmsnorm`]),
    /// whose [m, hidden] tiles are far narrower than the MLP's [m, intermediate].
    fn gelu(&mut self, x: Ssa, dims: Vec<i64>) -> Ssa {
        let k = self.splat((2.0f32 / std::f32::consts::PI).sqrt() as f64, dims.clone());
        let c = self.splat(0.044_715, dims.clone());
        let x2 = self.binop(OpKind::ArithMulf, x, x, dims.clone());
        let x3 = self.binop(OpKind::ArithMulf, x2, x, dims.clone());
        let cx3 = self.binop(OpKind::ArithMulf, c, x3, dims.clone());
        let inner0 = self.binop(OpKind::ArithAddf, x, cx3, dims.clone());
        let inner = self.binop(OpKind::ArithMulf, k, inner0, dims.clone());
        // `min(max(inner, -15), 15)` — one splat each, dying at its own use.
        let lo = self.splat(-15.0, dims.clone());
        let hi = self.splat(15.0, dims.clone());
        let clamped_lo = self.binop(OpKind::ArithMaximumf, inner, lo, dims.clone());
        let inner = self.binop(OpKind::ArithMinimumf, clamped_lo, hi, dims.clone());
        let t = self.unop(OpKind::MathTanh, inner, dims.clone());
        // `0.5·x·(1+t) = 0.5·x + 0.5·x·t`, with the `0.5` splat DEFERRED to here and
        // the `1` splat GONE — the polynomial's transients are dead by now, so the
        // clamp's two splats never overlap them. MEASURED: with all constants splatted
        // up front the fused map window held 7 tiles and died at
        // `TensorSplat: LX capacity exceeded: 2027520 + 337920 > 2097152`.
        let half = self.splat(0.5, dims.clone());
        let hx = self.binop(OpKind::ArithMulf, half, x, dims.clone());
        let ht = self.binop(OpKind::ArithMulf, hx, t, dims.clone());
        self.binop(OpKind::ArithAddf, hx, ht, dims)
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
        // ⭐ `outs` NAMES THE INIT OPERAND — MLIR's `linalg.reduce ins(...) outs(%init)`
        // spells the accumulator seed as the `outs` value, and the interpreter folds
        // `combiner(reduced, outs)` UNCONDITIONALLY when the attr is present
        // (`linalg.rs`'s `test_reduce_folds_outs_init` pins it: sum with `outs` 100
        // is 110, not 10). Without the attr the init operand is silently dropped on
        // every path — harmless while every reduce seeded an identity splat, and
        // WRONG the moment `rmsnorm_inv`'s column-blocked variance chains a second
        // reduce seeded with the first block's partial sum: the sum then covered
        // only the LAST block's columns (MEASURED at m=64, hidden=2816: the
        // sum-of-squares was ~35× too small, `inv` ~5.9× too large, and every
        // router logit wrong by that factor).
        let op = Operation::new(a, Some(v), OpKind::LinalgReduce, &[x, init])
            .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![dim])))
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(out.clone())))
            .with_attr(a, AttrKey::ReduceFn, Attr::Op(f))
            .with_attr(a, AttrKey::OutsVar, Attr::Ssas(a.ssa(vec![init])));
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
    /// source never exercises this, because its values are tiny. MEASURED on gemma-4-12b-it:
    /// layer 11's `down_proj` output reaches |x| = 651, whose f16 square is inf — every row
    /// carrying such an element was ZEROED by the scale=0, and the dead rows then re-exploded
    /// at the next norm (1/sqrt(eps) × gamma), NaN-ing the whole forward from layer 12 on.
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
    /// ⭐ NO ROW BLOCKING HERE, unlike [`KtirFunc::silu_mul`]. Row blocking would hand the consumer
    /// the block height instead of `m` and declare the scratch at that height — the same divergence
    /// in a smaller denomination — AND it would give one program N `math.sqrt` roots where the
    /// consumer's `program_rmsnorm_eps` reads exactly one. An emulator-motivated tiling belongs in
    /// `ktir-optimizer`, which is `spyre-emu`-gated; the producer states the shape the node has.
    /// ⭐⭐ BUT THE VARIANCE PHASE IS **COLUMN**-BLOCKED when the region does not fit a core's LX —
    /// see [`Self::rmsnorm_inv`] for why that is the one legal blocking axis.
    fn rmsnorm(
        &mut self,
        x_r: &TensorRegion,
        gamma: &TensorRegion,
        out: &TensorRegion,
        eps: f32,
        gain_offset: f32,
    ) {
        // The region's extents, read off the output — ONE spelling of each quantity. The width used to
        // arrive as a registry index beside it too, and two spellings of one number is what lets them
        // disagree.
        let c = out.region.cols.len;
        let m = out.region.rows.len;
        let dims = vec![i64::from(m), i64::from(c)];

        let (x, inv_e) = self.rmsnorm_inv(x_r, eps);
        // `[m]` → `[m, c]` along the COLUMN axis: every column of a row shares that row's scale.
        let invb = self.broadcast(inv_e, dims.clone(), 1);
        let xs = self.binop(OpKind::ArithMulf, x, invb, dims.clone());
        // gamma is one row `[1, c]`, loaded RANK-1 so the broadcast produces `[r, c]` — a `[1, c]`
        // load would broadcast to `[1, 1, c]`, since the emulator does not rank-reduce.
        let gcol = self.load_1d(gamma.tensor, c, 0, c);
        // The (1 + w) convention: offset the loaded gain row BEFORE the broadcast. Computed here,
        // never folded into the weight — see `lower_rmsnorm_node`'s gain_offset note.
        let gcol = if gain_offset != 0.0 {
            let off = self.f32_splat(f64::from(gain_offset), vec![i64::from(c)]);
            self.f32_binop(OpKind::ArithAddf, gcol, off, vec![i64::from(c)])
        } else {
            gcol
        };
        let gb = self.broadcast(gcol, dims.clone(), 0);
        let y = self.binop(OpKind::ArithMulf, xs, gb, dims);
        self.store_region(y, out);
    }

    /// THE VARIANCE PHASE OF EVERY RMSNORM — load `x` whole, compute the per-row
    /// `inv = 1/sqrt(mean(x²) + eps)` in f32, narrow it to f16, and return
    /// `(the whole-region x tile, inv_e)` for the caller's own tail multiply.
    ///
    /// ⛔ THE SQUARE IS f32, NOT f16. An f16 square overflows at |x| > 256 — see
    /// [`Self::rmsnorm`]'s head comment for the measured gemma-4 failure — so the
    /// chain widens `x` BEFORE squaring, exactly as `assemble_rmsnorm`'s f32 mean
    /// reduce widens its operands on the card.
    ///
    /// ⭐⭐ AND THE SQUARE IS **COLUMN**-BLOCKED WHEN THE REGION DOES NOT FIT LX. The
    /// f32 chain holds `x`'s block (f16), its widening and its square (f32 each) at
    /// once — 5 f16-tile-equivalents per column — which on gemma-4's `[496, 512]`
    /// global-layer Q norm is 2,539,520 B > 2 MiB (MEASURED as `ArithMulf: LX
    /// capacity exceeded ... charging %75 tile [496, 512]`). COLUMN blocking is the
    /// one legal axis because everything after the reduce is `[m]`-shaped: the
    /// partial sums concatenate by ADDING (an `[m]` `addf` per block), the ONE
    /// `math.sqrt` root survives (a row block would mint one root per block and
    /// break `program_rmsnorm_eps`'s single-root reading), and the final multiply
    /// and store stay whole-region so `node_rows`/`r_cover` still see one node.
    /// Every block's access tiles keep row corner 0 — `base_addressed`'s law.
    fn rmsnorm_inv(&mut self, x_r: &TensorRegion, eps: f32) -> (Ssa, Ssa) {
        let c = x_r.region.cols.len;
        let m = x_r.region.rows.len;
        let rows = vec![i64::from(m)];
        let x = self.load_region(x_r);

        // The f32 square chain holds `RMS_LIVE_TILES` f16-tile-equivalents of a
        // column block plus two `[m]` f32 accumulators (see that constant's comment).
        let live = u64::from(RMS_LIVE_TILES);
        let w: u32 = if u64::from(m) * u64::from(c) * live > u64::from(EW_LX_ELEMS) {
            ((u64::from(EW_LX_ELEMS) / live / u64::from(m.max(1))) as u32)
                .max(64)
                .min(c)
        } else {
            c
        };

        // The per-row sum of squares, accumulated across column blocks.
        let mut ssum = self.f32_splat(0.0, rows.clone());
        let mut off = 0u32;
        while off < c {
            let h = w.min(c - off);
            let blk_dims = vec![i64::from(m), i64::from(h)];
            let xb = if w >= c {
                x
            } else {
                self.load_region(&sub_cols(x_r, off, h))
            };
            let xbf = {
                let v = self.fresh();
                let op = Operation::new(self.a, Some(v), OpKind::ArithExtf, &[xb]);
                let ty = self.f32_ty(blk_dims.clone());
                let op = self.typed(op, ty);
                self.push(op);
                v
            };
            let x2 = self.f32_binop(OpKind::ArithMulf, xbf, xbf, blk_dims);
            let part = self.reduce(x2, ssum, OpKind::ArithAddf, 1, rows.clone());
            ssum = part;
            off += h;
        }

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
        (x, inv_e)
    }

    /// Unit-gain RmsNorm — [`Self::rmsnorm`]'s chain WITHOUT the gamma multiply: the same f32
    /// mean-of-squares / `1/rms` / broadcast, stopping at `x · inv_rms` (Gemma4 `v_norm`, no
    /// learnable scale). A separate entry point rather than a `gamma: Option` on [`Self::rmsnorm`],
    /// because the gain operand changes the PARAMETER set (the emulator binds one address per
    /// argument), and an op that never had a gamma must not mint a parameter for one.
    ///
    /// ⛔ NOT ROW-BLOCKED, for the same reason [`Self::rmsnorm`] is not: the consumer reads the
    /// program's ONE `math.sqrt` root structurally (`program_rmsnorm_eps`), and N blocks would be
    /// N roots. An emulator-motivated tiling belongs in `ktir-optimizer` (`spyre-emu`-gated).
    fn rmsnorm_unit(&mut self, x_r: &TensorRegion, out: &TensorRegion, eps: f32) {
        let c = out.region.cols.len;
        let m = out.region.rows.len;
        let dims = vec![i64::from(m), i64::from(c)];
        let (x, inv_e) = self.rmsnorm_inv(x_r, eps);
        let invb = self.broadcast(inv_e, dims.clone(), 1);
        let y = self.binop(OpKind::ArithMulf, x, invb, dims);
        self.store_region(y, out);
    }

    /// The FUSED per-head norm sandwich — [`Self::rmsnorm`]'s chain applied to head
    /// windows of the ORIGINAL `[m, H·D]` tensor, writing the flatten-back output's
    /// head windows, both reshapes deleted (`lower_sandwich_node` carries the proof).
    ///
    /// ⭐ THE VIEWS STATE THE DOOR'S FACTS: `x` is viewed `[m, H·D]` (the arrangement the
    /// flatten restores), `gamma` is loaded rank-1 `[D]` (one head's width, shared across
    /// heads — the on-disk q/k-norm rows), and the OUTPUT's first access tile is ONE
    /// HEAD'S WINDOW `[m, D]` at column `0` — the tile the door reads the head width
    /// from, exactly the way `Program::Attn`'s door reads its geometry off program views.
    ///
    /// ⛔ ONE `math.sqrt` ROOT PER HEAD, NOT ONE PER PROGRAM: `rmsnorm_inv`'s
    /// single-root law exists for `program_rmsnorm_eps`'s reading of a WHOLE-REGION norm;
    /// this program has `H` independent chains, and the door's reader
    /// (`program_windowed_eps`) is the read-all-roots-require-agreement form (the
    /// `program_scalarmul_scale` precedent) — so a per-head column-blocked chain would
    /// ALSO be legal here, but this body keeps each head's chain whole-D because a head
    /// (`D` = 256/512) fits the LX budget `rmsnorm_inv` already accounts for.
    fn windowed_rmsnorm(
        &mut self,
        x_r: &TensorRegion,
        gamma: &Option<&TensorRegion>,
        out: &TensorRegion,
        head_dim: u32,
        eps: f32,
        gain_offset: f32,
    ) {
        // The whole geometry is derivable from the two regions: `m` and `full` are
        // `x`'s own extents (the matcher proved the flatten restores this arrangement),
        // and `heads` is their quotient — the door re-derives the same three off the
        // output's view, so taking them as arguments would be a second spelling that
        // could disagree.
        let (m, full) = (x_r.region.rows.len, x_r.region.cols.len);
        debug_assert_eq!(full % head_dim, 0);
        let heads = full / head_dim;
        let dims = vec![i64::from(m), i64::from(head_dim)];
        let rows = vec![i64::from(m)];
        for h in 0..heads {
            let off = h * head_dim;
            // Head `h`'s window of `x` — a column slice of the ORIGINAL tensor, which is
            // exactly what the deleted split+norm+flatten computed.
            let xh = self.load_region(&sub_cols(x_r, off, head_dim));
            // The f32 mean-of-squares chain over the head window: `rmsnorm_inv`'s
            // arithmetic on a `[m, D]` slice, ending at `inv_e` (`[m]` f16).
            let mut ssum = self.f32_splat(0.0, rows.clone());
            let xf = {
                let v = self.fresh();
                let op = Operation::new(self.a, Some(v), OpKind::ArithExtf, &[xh]);
                let ty = self.f32_ty(dims.clone());
                let op = self.typed(op, ty);
                self.push(op);
                v
            };
            let x2 = self.f32_binop(OpKind::ArithMulf, xf, xf, dims.clone());
            let part = self.reduce(x2, ssum, OpKind::ArithAddf, 1, rows.clone());
            ssum = part;
            let dts = self.f32_splat(f64::from(head_dim), rows.clone());
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
                let ty = self.tensor_ty(rows.clone());
                let op = self.typed(op, ty);
                self.push(op);
                v
            };
            let invb = self.broadcast(inv_e, dims.clone(), 1);
            let xs = self.binop(OpKind::ArithMulf, xh, invb, dims.clone());
            let y = match gamma {
                Some(g) => {
                    let gcol = self.load_1d(g.tensor, head_dim, 0, head_dim);
                    let gcol = if gain_offset != 0.0 {
                        let off = self.f32_splat(f64::from(gain_offset), vec![i64::from(head_dim)]);
                        self.f32_binop(OpKind::ArithAddf, gcol, off, vec![i64::from(head_dim)])
                    } else {
                        gcol
                    };
                    let gb = self.broadcast(gcol, dims.clone(), 0);
                    self.binop(OpKind::ArithMulf, xs, gb, dims.clone())
                }
                None => xs,
            };
            self.store_region(y, &sub_cols(out, off, head_dim));
        }
    }

    /// SiluMul — `out[j] = (gate / (1 + exp(-gate))) · up`.
    /// ScalarWeightMul — `out = x · w`, `w` a LOADED `[1]`-shaped weight
    /// (gemma4 `layer_scalar[layer]`). The weight is loaded RANK-1 (`load_1d`,
    /// one element) and broadcast along the ROW axis to `[m, c]` — the same
    /// construction `rmsnorm` uses for its `[1, c]` gamma, so the broadcast
    /// produces a 2-D tile and never a `[1, 1, c]` the emulator cannot
    /// rank-reduce.
    ///
    /// ⭐ THE WEIGHT IS A REAL PARAMETER, not a splat: the value is staged by
    /// the host from the checkpoint (`superdsc_weights` binds the
    /// `RmsNorm`-kind accessor's `.weight`), so the program must LOAD it —
    /// unlike [`Self::tanhsoftcap`]'s cap, which is a config constant the
    /// program states inline.
    fn scalar_weight_mul(&mut self, x_r: &TensorRegion, w_r: &TensorRegion, out: &TensorRegion) {
        let c = out.region.cols.len;
        let m = out.region.rows.len;
        let x = self.load_region(x_r);
        let dims = vec![i64::from(m), i64::from(c)];
        // The `[1]` weight, loaded rank-1 and sprayed over every row.
        let w = self.load_1d(w_r.tensor, 1, 0, 1);
        let wb = self.broadcast(w, dims.clone(), 0);
        let y = self.binop(OpKind::ArithMulf, x, wb, dims);
        self.store_region(y, out);
    }

    /// A ROW SOFTMAX over `[m, c]` — the MoE router's `RouteSoftmax`, written
    /// longhand as the stability chain `exp(x − rowmax) / rowsum(exp(x − rowmax))`.
    ///
    /// The structure is [`Self::rmsnorm_unit`]'s — a per-row reduce, a broadcast
    /// back over the row, a terminal divide — with `max`/`exp`/`sum`/`divf` in
    /// place of the norm's primitives, because a row softmax and a row rms are
    /// the same shape of computation: one scalar per row, sprayed back over it.
    /// All in the EMULATOR's f32 (`f32_splat`/`f32_binop`), the same arithmetic
    /// the rmsnorm chain uses; the descriptor side is
    /// [`assemble_row_softmax`]'s f16 primitives.
    fn route_softmax(&mut self, x_r: &TensorRegion, out: &TensorRegion) {
        let c = out.region.cols.len;
        let m = out.region.rows.len;
        let x = self.load_region(x_r);
        let dims = vec![i64::from(m), i64::from(c)];
        let rows = vec![i64::from(m)];
        // 1. rowmax(x) — the stability subtractend, `[m]` f32.
        let minit = self.f32_splat(f64::from(f32::NEG_INFINITY), rows.clone());
        let xmax = {
            let a = self.a;
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::LinalgReduce, &[x, minit])
                .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![1])))
                .with_attr(a, AttrKey::ReduceFn, Attr::Op(OpKind::ArithMaxnumf));
            let ty = self.f32_ty(rows.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        // 2. x − rowmax. ⛔ `x` IS ALREADY `[m, c]` — ONLY the `[m]` rowmax needs
        //    the broadcast (axis 1 sprays each row's scalar back over its columns);
        //    broadcasting `x` too inserts an axis on a FULL-RANK tile, and the
        //    emulator's `broadcast_to` refuses `[m, 1, c] → [m, c]`. MEASURED as
        //    `LinalgBroadcast: cannot broadcast [2, 1, 8] to [2, 8]` the first time
        //    the whole router chain ran fused — RouteSoftmax had no single-op test,
        //    and the (also untested) `route_renorm` below had the same latent
        //    shape error in its `xb` broadcast, which this fix removes too.
        let maxb = self.broadcast(xmax, dims.clone(), 1);
        let shifted = self.f32_binop(OpKind::ArithSubf, x, maxb, dims.clone());
        // 3. exp(shifted).
        let e = self.unop(OpKind::MathExp, shifted, dims.clone());
        // 4. rowsum(exp) — the denominator, `[m]` f32.
        let sinit = self.f32_splat(0.0, rows.clone());
        let sum = {
            let a = self.a;
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::LinalgReduce, &[e, sinit])
                .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![1])))
                .with_attr(a, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
            let ty = self.f32_ty(rows.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        // 5. exp / rowsum, the denominator broadcast back over the row.
        let sumb = self.broadcast(sum, dims.clone(), 1);
        let y = self.f32_binop(OpKind::ArithDivf, e, sumb, dims);
        self.store_region(y, out);
    }

    /// A ROW RENORM over `[m, c]` — mixtral's `RouteRenorm`, `x / rowsum(x)`:
    /// the softmax chain minus the stability subtract and the exp.
    fn route_renorm(&mut self, x_r: &TensorRegion, out: &TensorRegion) {
        let c = out.region.cols.len;
        let m = out.region.rows.len;
        let x = self.load_region(x_r);
        let dims = vec![i64::from(m), i64::from(c)];
        let rows = vec![i64::from(m)];
        let sinit = self.f32_splat(0.0, rows.clone());
        let sum = {
            let a = self.a;
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::LinalgReduce, &[x, sinit])
                .with_attr(a, AttrKey::Dimensions, Attr::IntList(a.ints(vec![1])))
                .with_attr(a, AttrKey::ReduceFn, Attr::Op(OpKind::ArithAddf));
            let ty = self.f32_ty(rows.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        let sumb = self.broadcast(sum, dims.clone(), 1);
        let y = self.f32_binop(OpKind::ArithDivf, x, sumb, dims);
        self.store_region(y, out);
    }

    /// ⭐ ARGMAX-STYLE ROW GATHER — `out[i, j] = src[i, idx[i, j]]`, the shape both
    /// [`Self::route_gather_scores`] and [`Self::route_expert_scale`] share.
    ///
    /// The gather is an `ktdp.construct_indirect_access_tile` over the parent
    /// `[m, W]` view with a DIRECT row dim and an INDIRECT column dim read
    /// through the indices view `[m, k]` (flattened, so enumeration point
    /// `(d0, d1)` addresses index element `d0·k + d1` — the `dim_subs` subscript
    /// `Dim(0)·k + Dim(1)`, dotted with the view's rank-1 stride). The loaded
    /// index is the PARENT COLUMN for that output element, exactly
    /// `take_along_axis`'s `out[n, k] = src[n, indices[n, k]]`.
    ///
    /// `W` is the parent's width (`E` for the scores, `E` for the per-expert
    /// scale row); `idx_cols` is the gather's width (`k`).
    fn gather_rows(&mut self, parent: &TensorRegion, idx_r: &TensorRegion, out: &TensorRegion) {
        let m = out.region.rows.len;
        let idx_cols = out.region.cols.len;
        let a = self.a;
        // The parent view, whole — the gather's coords name absolute rows/cols.
        let parent_view = self.view(parent.tensor);
        // The indices as ONE FLAT rank-1 view: element (n, j) sits at `n·k + j`,
        // which is the subscript the indirect dim's `dim_subs` entry states.
        let idx_ptr = self.arg_for(idx_r.tensor);
        let idx_view = self.view_of(
            idx_ptr,
            vec![i64::from(m) * i64::from(idx_cols)],
            vec![1],
        );
        // THE INDIRECT TILE: dim 0 direct (the row), dim 1 indirect (the column,
        // read from the indices view at this output element's flat position).
        let iat = self.fresh();
        // `Dim(0)·k + Dim(1)` — the enumeration point's flat index into the
        // indices view. One expression per INDEX-VIEW AXIS (rank 1 here), so the
        // map's single result is that flat subscript.
        let sub = a.expr(AffineExpr::Add(
            a.expr(AffineExpr::Mul(
                a.expr(AffineExpr::Const(i64::from(idx_cols))),
                a.expr(AffineExpr::Dim(0)),
            )),
            a.expr(AffineExpr::Dim(1)),
        ));
        // The enumeration space is the full output box `[m, k]` — an
        // UNCONSTRAINED set (no constraints) enumerates exactly that.
        let vss = AffineSet {
            num_dims: 2,
            num_syms: 0,
            constraints: a.constraints(vec![]),
        };
        let dims = vec![i64::from(m), i64::from(idx_cols)];
        let op = Operation::new(
            a,
            Some(iat),
            OpKind::KtdpConstructIndirectAccessTile,
            &[parent_view, idx_view],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
        .with_attr(
            a,
            AttrKey::DimKinds,
            Attr::StrList(a.names(vec!["direct", "indirect"])),
        )
        .with_attr(a, AttrKey::DimData, Attr::IntList(a.ints(vec![0, 0])))
        .with_attr(a, AttrKey::IntermediateVars, Attr::Ssas(a.ssa(vec![])))
        .with_attr(
            a,
            AttrKey::DimSubs,
            // One map per output dim: dim 0 is direct (its map is unused but
            // must be present — `parse_dim_subscripts` refuses a short list),
            // dim 1's single expr is the flat index subscript above.
            Attr::AffineMapList(a.maps(vec![
                AffineMap {
                    num_dims: 2,
                    num_syms: 0,
                    exprs: a.exprs(vec![AffineExpr::Dim(0)]),
                },
                AffineMap {
                    num_dims: 2,
                    num_syms: 0,
                    exprs: std::slice::from_ref(sub),
                },
            ])),
        )
        .with_attr(a, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        let op = self.typed(
            op,
            IrType::AccessTile {
                dims: a.ints(dims.clone()),
            },
        );
        self.push(op);
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::KtdpLoad, &[iat])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        self.store_region(v, out);
    }

    /// ARGUMENT SORT by ascending row value — the MoE router's
    /// [`SubOp::RouteArgsort`], as the RANK VECTOR:
    /// `rank[i, j] = |{h : x[i,h] < x[i,j]}| + |{h : x[i,h] == x[i,j] ∧ h < j}|`.
    ///
    /// ⭐ A COMPARE/REDUCE, NOT A PERMUTATION. The count form computes each
    /// element's sorted position directly — a stable ascending argsort's rank —
    /// with no data-dependent memory traffic, which is what the emulator's
    /// fixed-shape tiles express. It is exactly MLX `block_sort`'s ordering
    /// (metal's `argpartition.metal`, LessThan with NaN-as-greater: ties by
    /// index), so the top-k slice downstream matches metal's bit for bit.
    ///
    /// ⛔ NaN LANDS AT THE END BECAUSE IT IS SANITIZED TO +INF FIRST, not
    /// because the raw compares would put it there — they would NOT: an
    /// unsanitized NaN wins neither `olt` nor `oeq` and its rank is 0, the
    /// FRONT of the sorted row, stealing a top-k slot from a real expert. The
    /// `cmpf uno`/`select` sanitize below maps NaN to +inf, and +inf then sorts
    /// greater than every finite exactly as MLX's NaN-as-greater convention
    /// demands.
    ///
    /// ⛔ ROW-BLOCKED IN `E²` PER ROW, NOT `E`. Every intermediate here is
    /// `[h, E, E]` — one full row-pair outer product per block row — and the
    /// peak live set is two Bool compares plus two f32 sums (~6 f16-tile
    /// equivalents of `E²` per row), so the block height that keeps that inside
    /// the LX is `EW_LX_ELEMS / (E² · 8)`, not `rows_per_block`'s own division.
    fn route_argsort(&mut self, x_r: &TensorRegion, out: &TensorRegion) {
        let e = out.region.cols.len;
        debug_assert_eq!(e, x_r.region.cols.len, "RouteArgsort is shape-preserving");
        let blk = (EW_LX_ELEMS / (e.max(1) * e.max(1)) / 8).max(1);
        // The tie-break's `h' < j` compare: two broadcasts of ONE `[1, E]` iota
        // row (a `tensor.generate` yielding its column index), giving `[1, 1, E]`
        // (the h' values) against `[1, E, 1]` (the j values) — the pair compares
        // `h' < j` at every `(j, h')`, block-invariant, so it is built ONCE.
        // `iota[h']` varies along AXIS 2 and `iota[j]` along AXIS 1 — the same
        // (j, h') placement the value compares use — so `cmpf olt` at (·, j, h')
        // reads `h' < j`, the stable-order tie-break's direction.
        let iota = self.index_grid(1, e, 1);
        let h_vec = self.broadcast(iota, vec![1, 1, i64::from(e)], 1);
        let j_vec = self.broadcast(iota, vec![1, i64::from(e), 1], 2);
        let h_lt_j = self.cmpf("olt", h_vec, j_vec, vec![1, i64::from(e), i64::from(e)]);
        let mut off = 0u32;
        while off < out.region.rows.len {
            let h = blk.min(out.region.rows.len - off);
            let x = self.load_region(&sub_rows(x_r, off, h));
            let dims = vec![i64::from(h), i64::from(e)];
            // ⛔ NaN → +inf FIRST, so the plain ordered compares below compute
            // MLX's NaN-as-greater total order exactly. WITHOUT this, a NaN
            // target wins neither `olt` nor `oeq` and its rank is 0 — it would
            // land at the FRONT of the sorted row and, worse, the +inf the
            // router's own softmax can never produce but a corrupted logits
            // row can, would steal a top-k slot from a real expert. The
            // sanitize is `cmpf uno (x, x)` (true exactly at NaN) selecting
            // +inf over the value; f16 holds +inf.
            let is_nan = self.cmpf("uno", x, x, dims.clone());
            let inf = self.splat(f64::INFINITY, dims.clone());
            let x = self.select(is_nan, inf, x, dims.clone());
            let wide = vec![i64::from(h), i64::from(e), i64::from(e)];
            // `x[i, j]` as a column vector `[h, E, 1]` and `x[i, h']` as a row
            // vector `[h, 1, E]` — the pair the compares broadcast to `[h, E, E]`.
            let col_vec = self.broadcast(x, vec![i64::from(h), i64::from(e), 1], 2);
            let row_vec = self.broadcast(x, vec![i64::from(h), 1, i64::from(e)], 1);
            // 1. strictly-less: `x[i, h'] < x[i, j]` contributes 1 to rank[i, j].
            let lt = self.cmpf("olt", row_vec, col_vec, wide.clone());
            // 2. tie-break: `x[i, h'] == x[i, j]` AND `h' < j` — the AND of two
            //    0/1 tiles is their product (`[h, E, E]` × `[1, E, E]` broadcasts).
            let eq = self.cmpf("oeq", row_vec, col_vec, wide.clone());
            let tie = self.binop(OpKind::ArithMulf, eq, h_lt_j, wide.clone());
            // 3. rank = Σ_h' (lt + tie), reduced over dim 2 (the h' axis), the
            //    init a zero of the RESULT shape (`[h, E]`) — the reduce folds
            //    `combiner(reduced, outs)` unconditionally, so a non-identity
            //    init would add itself to every row.
            let sum = self.binop(OpKind::ArithAddf, lt, tie, wide);
            let zero = self.splat_zero(dims.clone());
            let rank = self.reduce(sum, zero, OpKind::ArithAddf, 2, dims.clone());
            self.store_region(rank, &sub_rows(out, off, h));
            off += h;
        }
    }

    /// `tensor.generate {^bb0(%i, %j): yield %axis}` — one INDEX GRID over
    /// `[r, c]`, yielding axis `axis`'s index (i32) at each position: the iota
    /// the argsort tie-break compares. Three ops (bb0 args, yield, generate),
    /// the shape the `generate` handler's vectorized meshgrid execution runs.
    fn index_grid(&mut self, r: u32, c: u32, axis: usize) -> Ssa {
        let a = self.a;
        let (i, j, t) = (self.fresh(), self.fresh(), self.fresh());
        // `^bb0(%i, %j)` — the block arguments, named in the bb0 marker op.
        let bb0 = Operation::new(a, None, OpKind::RegionBb0Args, &[])
            .with_attr(a, AttrKey::Names, Attr::Ssas(a.ssa(vec![i, j])));
        // `tensor.yield %axis` — the body is a single yield of the wanted arg.
        let yielded = if axis == 0 { i } else { j };
        let y = Operation::new(a, None, OpKind::TensorYield, &[yielded]);
        // The generate op itself: rank-2 shape, i32 elements (meshgrid grids are
        // i32 tiles), the region = [bb0, yield].
        let shape = vec![i64::from(r), i64::from(c)];
        let mut generate = Operation::new(a, Some(t), OpKind::TensorGenerate, &[])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(shape.clone())))
            .with_attr(
                a,
                AttrKey::Dtype,
                Attr::Dtype(ktir_core::dtypes::DType::I32),
            );
        generate.regions = a.regions(vec![a.ops(vec![bb0, y])]);
        let ty = IrType::Tensor {
            dims: a.ints(shape),
            elem: ktir_core::dtypes::DType::I32,
        };
        let generate = self.typed(generate, ty);
        self.push(generate);
        t
    }

    /// `arith.cmpf` with predicate `pred` over two same-shape tiles — Bool 0/1
    /// result. A dedicated entry because [`Self::binop]` states only float
    /// kinds; the compare's operands and result all carry the shape attribute
    /// the map-window planner reads.
    fn cmpf(&mut self, pred: &'static str, l: Ssa, r: Ssa, dims: Vec<i64>) -> Ssa {
        let a = self.a;
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::ArithCmpf, &[l, r])
            .with_attr(a, AttrKey::Predicate, Attr::Str(pred))
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// `arith.select` — element-wise `cond ? t : f` over tiles of one shape.
    fn select(&mut self, cond: Ssa, t: Ssa, f: Ssa, dims: Vec<i64>) -> Ssa {
        let a = self.a;
        let v = self.fresh();
        let op = Operation::new(a, Some(v), OpKind::ArithSelect, &[cond, t, f])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
        let ty = self.tensor_ty(dims);
        let op = self.typed(op, ty);
        self.push(op);
        v
    }

    /// The top-k expert indices of each row — [`SubOp::RouteTopK`], metal's
    /// `sorted[..., -k:]` selection (`slice_trailing_cols`'s experts).
    ///
    /// ⭐ A ONE-HOT SELECTOR SUM OVER THE RANK VECTOR, NOT A COLUMN SLICE. The
    /// input here is [`Self::route_argsort`]'s RANK vector — `rank[n, h]` is
    /// expert `h`'s sorted position, a PERMUTATION of `0..E` (the stable
    /// tie-break makes every rank distinct). Metal's argpartition kernel
    /// returns the sorted-INDEX permutation, so ITS trailing slice is the top-k
    /// indices; slicing OUR rank vector's trailing columns would instead read
    /// "the ranks of the last k columns" — measured picking completely wrong
    /// experts (`[45,110,48,124,...]` against the golden `[73,59,111,44,...]`).
    /// The inversion is the selector `indices[n, j] = Σ_h h ·
    /// 𝟙[rank[n, h] == E − k + j]`: exactly one `h` matches each target rank,
    /// so the sum IS that `h` — the same expert metal's trailing slice names,
    /// in the same order (ascending score). Same compare/reduce machinery as
    /// the argsort itself, no data-dependent memory traffic.
    ///
    /// ⛔ THE IOTA IS GENERATED AT THE FULL `[m, E]` SHAPE, not broadcast from
    /// a `[1, E]` row. The selector needs `h` as a VALUE in the summand;
    /// `tensor.generate` yields i32 (the meshgrid convention), so one
    /// `arith.sitofp` widens it and the multiply/reduce stay float like every
    /// other reduce here. `linalg.broadcast` INSERTS an axis (see
    /// [`Self::route_softmax`]'s ⛔), so a `[1, E]` iota broadcasts to
    /// `[1, 1, E]`, not `[m, E]` — the meshgrid at the output's own shape
    /// avoids the broadcast entirely.
    fn route_topk(&mut self, sorted_r: &TensorRegion, out: &TensorRegion, k: u32) {
        let m = out.region.rows.len;
        let e = sorted_r.region.cols.len;
        let rank = self.load_region(sorted_r);
        let wide = vec![i64::from(m), i64::from(e)];
        // The iota grid `[m, E]` (expert id per column), widened to float.
        let iota = self.index_grid(m, e, 1);
        let iota_f = {
            let v = self.fresh();
            let op = Operation::new(self.a, Some(v), OpKind::ArithSitofp, &[iota]);
            let ty = self.f32_ty(wide.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        for j in 0..k {
            // The target rank: the `j`-th smallest of the top-k ranks.
            let target = self.splat(f64::from((e - k + j) as f32), wide.clone());
            // `𝟙[rank == target] · h`, reduced over the expert axis: exactly one
            // summand is nonzero, so the sum is the matching expert's index.
            let is_match = self.cmpf("oeq", rank, target, wide.clone());
            let sel = self.binop(OpKind::ArithMulf, is_match, iota_f, wide.clone());
            let zero = self.splat_zero(vec![i64::from(m)]);
            let idx = self.reduce(sel, zero, OpKind::ArithAddf, 1, vec![i64::from(m)]);
            self.store_region(idx, &sub_cols(out, j, 1));
        }
    }

    /// The scores at the chosen indices — [`SubOp::RouteGatherScores`]:
    /// `out[n, j] = logits[n, idx[n, j]]`, `take_along_axis` at axis -1.
    /// [`Self::gather_rows`] with the logits as the parent.
    fn route_gather_scores(&mut self, scores_r: &TensorRegion, idx_r: &TensorRegion, out: &TensorRegion) {
        self.gather_rows(scores_r, idx_r, out);
    }

    /// Each score times its expert's learned scale — [`SubOp::RouteExpertScale`]:
    /// `out[m, j] = scores[m, j] · per_expert_scale[idx[m, j]]`, with the
    /// multiply in f32 (metal's `moe_per_expert_scale` computes a float
    /// intermediate so the f16 result has ONE rounding, matching the host
    /// reference).
    ///
    /// The per-expert scale is a `[1, E]` row the loader stages; it is loaded
    /// RANK-1 (`load_1d`, `[E]`) so the gather's parent view is `[1, E]` —
    /// every row gathers from the same one-row parent, which the direct dim 0
    /// (the enumeration row) would misaddress. ⛔ SO THE GATHER IS RANK-1: the
    /// enumeration point is the FLAT `(n, j)` pair and the indirect dim reads
    /// `per_expert_scale[idx[n, j]]` — a one-dim gather whose parent is the `[E]`
    /// row and whose index view is the flattened `[m·k]` indices, with the
    /// OUTPUT reshaped `[m, k]` by the store's own region.
    fn route_expert_scale(
        &mut self,
        scores_r: &TensorRegion,
        idx_r: &TensorRegion,
        scale_r: &TensorRegion,
        out: &TensorRegion,
    ) {
        let m = out.region.rows.len;
        let k = out.region.cols.len;
        let a = self.a;
        // The scale row, rank-1 `[E]` — `load_1d`'s form.
        let e = scale_r.region.cols.len;
        let scale_ptr = self.arg_for(scale_r.tensor);
        let scale_view = self.view_of(scale_ptr, vec![i64::from(e)], vec![1]);
        // The indices, flattened rank-1 `[m·k]`.
        let idx_ptr = self.arg_for(idx_r.tensor);
        let idx_view = self.view_of(idx_ptr, vec![i64::from(m) * i64::from(k)], vec![1]);
        // The rank-1 gather: ONE indirect dim whose subscript reads
        // `idx[d0·k + d1]`... but a rank-1 enumeration point is just `d0`, so
        // the subscript is `Dim(0)` over the `[m·k]` indices view and the
        // output shape is `[m·k]` — flattened, with the multiply and the store
        // giving it back its `[m, k]` shape.
        let flat = m * k;
        let iat = self.fresh();
        let sub = a.expr(AffineExpr::Dim(0));
        let vss = AffineSet {
            num_dims: 1,
            num_syms: 0,
            constraints: a.constraints(vec![]),
        };
        let dims1 = vec![i64::from(flat)];
        let op = Operation::new(
            a,
            Some(iat),
            OpKind::KtdpConstructIndirectAccessTile,
            &[scale_view, idx_view],
        )
        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims1.clone())))
        .with_attr(
            a,
            AttrKey::DimKinds,
            Attr::StrList(a.names(vec!["indirect"])),
        )
        .with_attr(a, AttrKey::DimData, Attr::IntList(a.ints(vec![0])))
        .with_attr(a, AttrKey::IntermediateVars, Attr::Ssas(a.ssa(vec![])))
        .with_attr(
            a,
            AttrKey::DimSubs,
            Attr::AffineMapList(a.maps(vec![AffineMap {
                num_dims: 1,
                num_syms: 0,
                exprs: std::slice::from_ref(sub),
            }])),
        )
        .with_attr(a, AttrKey::VariablesSpaceSet, Attr::AffineSet(vss));
        let op = self.typed(
            op,
            IrType::AccessTile {
                dims: a.ints(dims1.clone()),
            },
        );
        self.push(op);
        let gathered = self.fresh();
        let op = Operation::new(a, Some(gathered), OpKind::KtdpLoad, &[iat])
            .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims1.clone())));
        let ty = self.tensor_ty(dims1.clone());
        let op = self.typed(op, ty);
        self.push(op);
        // The scores, flattened the same way, multiplied in f32.
        let scores = self.load_region(scores_r);
        let scores_flat = {
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::TensorReshape, &[scores])
                .with_attr(a, AttrKey::TargetShape, Attr::IntList(a.ints(dims1.clone())));
            let ty = self.tensor_ty(dims1.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        let g_f32 = {
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::ArithExtf, &[gathered]);
            let ty = self.f32_ty(dims1.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        let s_f32 = {
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::ArithExtf, &[scores_flat]);
            let ty = self.f32_ty(dims1.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        let prod = self.f32_binop(OpKind::ArithMulf, s_f32, g_f32, dims1.clone());
        let narrowed = {
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::ArithTruncf, &[prod]);
            let ty = self.tensor_ty(dims1.clone());
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        // Store through the output's own `[m, k]` region — the flat product
        // re-lays out by the store, the same identity `reshape` states.
        self.store_region(narrowed, out);
    }

    /// EXPERT SORT (the GATHERED/decode semantics) — [`SubOp::ExpertSort`] as
    /// a BLOCKWISE COPY `[m, w] → [m, k·w]`: output column block `j` is the
    /// input row again, i.e. the `[m, k, w]` (token, slot, width) layout.
    ///
    /// ⭐ METAL'S GATHERED BAKE EMITS NOTHING HERE (`S::Sort(_) => vec![]`):
    /// the gathered `affine_gather_qmv` reads the TOKEN rows directly, `k`
    /// pairs per row (`per_row = k`). The pair rows ARE the token rows, read
    /// `k` times — the sort's permutation is a no-op in this regime, and this
    /// copy is its materialized form: numerically identical (every element
    /// moved once, no arithmetic), no data-dependent permutation. The indices
    /// ride along because every expert projection reads them through this
    /// node's output routing.
    ///
    /// The `shared` bound is REFUSED by arity elsewhere; gemma-4 declares no
    /// shared expert, and a bundle that does is the fused-MoE loader's
    /// worklist, not this lowering's.
    fn expert_sort(&mut self, x_r: &TensorRegion, out: &TensorRegion, k: u32) {
        let w = x_r.region.cols.len;
        debug_assert_eq!(
            out.region.cols.len,
            w * k,
            "ExpertSort output is the [m, k·w] pair layout"
        );
        let x = self.load_region(x_r);
        // Column blocks: slot `j`'s block is the same `[m, w]` tile again. The
        // LX holds `w` elements per block row and the copy is pointwise, so
        // the whole `[m, w]` tile loads once and stores `k` times — the
        // natural blocking for a copy whose source is one tile.
        for j in 0..k {
            self.store_region(x, &sub_cols(out, j * w, w));
        }
    }

    /// EXPERT UNSORT — [`SubOp::ExpertUnsort`] in the same (token, slot)
    /// layout: the identity copy. Metal's gathered bake emits nothing here
    /// either (`S::Unsort => vec![]`) — the combine reads the pair rows where
    /// they lie — so this materializes the same non-move: all `m·k·w`
    /// elements moved once, no arithmetic, no permutation.
    fn expert_unsort(&mut self, rows_r: &TensorRegion, out: &TensorRegion) {
        let x = self.load_region(rows_r);
        self.store_region(x, out);
    }

    /// EXPERT COMBINE — [`SubOp::ExpertCombine`]:
    /// `out[n, d] = Σ_k rows[n, k·w + d] · scores[n, k]`, the f32-accumulated
    /// weighted sum metal's `moe_weighted_sum` computes (`acc = fma(row,
    /// score, acc)` in f32, one narrowing at the end).
    ///
    /// ⭐ THE SLOT SCORE BROADCAST, NOT A PER-SLOT SPLAT. Each slot `k`'s
    /// score is ONE `[m, 1]` column of the `[m, k]` scores tile; slicing it
    /// out (`tensor.extract_slice`) and broadcasting it across the `[m, w]`
    /// block multiplies the whole block by its slot's score in one op, which
    /// is what the kernel's inner loop does per (n, d) pair.
    ///
    /// ⛔ THE ACCUMULATOR IS f32, LIKE THE KERNEL'S `float acc`. A shared
    /// expert's contribution would be an extra `fma` term on the same
    /// accumulator; gemma-4 declares none (`shared = None`), and a bundle that
    /// declares one is refused at the loader, before a program is minted.
    fn expert_combine(
        &mut self,
        rows_r: &TensorRegion,
        scores_r: &TensorRegion,
        out: &TensorRegion,
    ) {
        let m = out.region.rows.len;
        let w = out.region.cols.len;
        let k = scores_r.region.cols.len;
        debug_assert_eq!(
            rows_r.region.cols.len,
            w * k,
            "ExpertCombine reads the [m, k·hidden] pair rows"
        );
        let a = self.a;
        let scores = self.load_region(scores_r);
        let mut acc: Option<Ssa> = None;
        for j in 0..k {
            // Slot `j`'s `[m, w]` block of the pair rows, widened to f32.
            let block = self.load_region(&sub_cols(rows_r, j * w, w));
            let block_f32 = {
                let v = self.fresh();
                let op = Operation::new(a, Some(v), OpKind::ArithExtf, &[block]);
                let ty = self.f32_ty(vec![i64::from(m), i64::from(w)]);
                let op = self.typed(op, ty);
                self.push(op);
                v
            };
            // Slot `j`'s score column `[m, 1]`, sliced out of `[m, k]` and
            // broadcast across the block's width.
            let col = {
                let v = self.fresh();
                let dims = vec![i64::from(m), 1];
                let op = Operation::new(a, Some(v), OpKind::TensorExtractSlice, &[scores])
                    .with_attr(
                        a,
                        AttrKey::SliceOffsets,
                        Attr::IntList(a.ints(vec![0, i64::from(j)])),
                    )
                    .with_attr(a, AttrKey::SliceSizes, Attr::IntList(a.ints(dims.clone())))
                    .with_attr(a, AttrKey::SliceStrides, Attr::IntList(a.ints(vec![1, 1])));
                let ty = self.tensor_ty(dims);
                let op = self.typed(op, ty);
                self.push(op);
                v
            };
            // Broadcast the `[m, 1]` column across the block's width — an
            // EXPANSION of its existing size-1 axis, not an insertion, so the
            // `Dimensions` list is EMPTY (`linalg.broadcast`'s other form; the
            // handler defaults an absent list to exactly this).
            let col_b = {
                let v = self.fresh();
                let dims = vec![i64::from(m), i64::from(w)];
                let init = self.fresh();
                let empty = Operation::new(a, Some(init), OpKind::TensorEmpty, &[])
                    .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
                    .with_attr(a, AttrKey::Dtype, Attr::Dtype(KTIR_ELEM));
                let ty = self.tensor_ty(dims.clone());
                let empty = self.typed(empty, ty);
                self.push(empty);
                let op = Operation::new(a, Some(v), OpKind::LinalgBroadcast, &[col, init])
                    .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
                let ty = self.tensor_ty(dims);
                let op = self.typed(op, ty);
                self.push(op);
                v
            };
            let col_f32 = {
                let v = self.fresh();
                let op = Operation::new(a, Some(v), OpKind::ArithExtf, &[col_b]);
                let ty = self.f32_ty(vec![i64::from(m), i64::from(w)]);
                let op = self.typed(op, ty);
                self.push(op);
                v
            };
            // `acc = fma(block, score, acc)` in f32 — the kernel's exact chain,
            // one narrowing at the very end.
            acc = Some(match acc {
                None => self.f32_binop(
                    OpKind::ArithMulf,
                    block_f32,
                    col_f32,
                    vec![i64::from(m), i64::from(w)],
                ),
                Some(prev) => {
                    let v = self.fresh();
                    let dims = vec![i64::from(m), i64::from(w)];
                    let op = Operation::new(a, Some(v), OpKind::MathFma, &[block_f32, col_f32, prev])
                        .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())));
                    let ty = self.f32_ty(dims);
                    let op = self.typed(op, ty);
                    self.push(op);
                    v
                }
            });
        }
        let acc = acc.expect("ExpertCombine has at least one slot (k ≥ 1)");
        let narrowed = {
            let v = self.fresh();
            let op = Operation::new(a, Some(v), OpKind::ArithTruncf, &[acc]);
            let ty = self.tensor_ty(vec![i64::from(m), i64::from(w)]);
            let op = self.typed(op, ty);
            self.push(op);
            v
        };
        self.store_region(narrowed, out);
    }

    /// ⛔ W-BLOCK BUDGET: the largest power-of-two divisor of `n` whose fp8→f16
    /// widened weight block `nb · in` keeps the per-core LX live set inside the
    /// 512 KB the emulator's own `n_block` charges an accumulator
    /// (`matmul_tile.rs`'s `BLOCK_MN_BUDGET`, the same number for the same
    /// reason — a few `[nb, in]`-scale siblings resident while the block is
    /// worked). The block is a divisor of `n` so every expert's slab tiles
    /// identically; a non-divisor would ragged the last block per expert.
    fn expert_w_block(n: u32, in_dim: u32) -> u32 {
        const W_BLOCK_BUDGET_BYTES: u64 = 512 * 1024;
        let mut nb = 1u32;
        let mut cand = 1u32;
        while cand <= n / 2 && n.is_multiple_of(cand) {
            if u64::from(cand) * u64::from(in_dim) * 2 <= W_BLOCK_BUDGET_BYTES {
                nb = cand;
            }
            cand *= 2;
        }
        if u64::from(n) * u64::from(in_dim) * 2 <= W_BLOCK_BUDGET_BYTES {
            n
        } else {
            nb
        }
    }

    /// EXPERT MATMUL — [`SubOp::ExpertMatmul`], the gathered form of
    /// [`Self::matmul_fp8`]: one projection of each (token, slot) pair, the
    /// weight slab selected by the pair's expert index out of the stacked
    /// `[E·out, in]` bank.
    ///
    /// `out[n, slot·n + c] = Σ_k rows[n, k] · W[e(n,slot), c, k] ·
    /// s[e(n,slot)·n + c]`, with `e(n, slot) = indices[n, slot]` — metal's
    /// `affine_gather_qmv` exactly: the kernel's `expert_idx =
    /// rhs_indices[n·top_k + slot]` slab base, its f32-accumulated contraction
    /// (one narrowing to `T_act` at the end of the accumulation,
    /// `qmv_impl`'s `static_cast`), and its per-output-channel scale
    /// multiply on the narrowed result. The rows this reads are the SORT's
    /// `[m, k·w]` pair rows — token row `n`'s activation in slot `j`'s
    /// column block, which is what the gathered kernel's `per_row = k`
    /// token-row re-read means once the sort is materialized.
    ///
    /// ⭐ THE EXPERT INDEX IS A RUNTIME VALUE AND THE EMISSION IS PER PAIR.
    /// Each (token, slot) pair's expert id is read OUT of the indices tensor
    /// (`tensor.extract` → `arith.fptosi` → an `index` multiply for the slab
    /// row offset), and the pair's whole projection is emitted against that
    /// dynamic corner — `construct_access_tile` takes SSA corner operands,
    /// so the block window names `e·out` in the bank without any
    /// data-dependent permutation. `m·k` pairs at `k ≤ 8` is the decode/pair
    /// regime the gathered bake covers; the count is a compile-time constant
    /// of the graph, so the per-pair unroll is too.
    ///
    /// ⛔ SELF-BLOCKED OVER OUTPUT COLUMNS, AND INVISIBLY TO
    /// `ktir_optimizer::matmul_tile`. The 26b bank's gate/up slab is
    /// `[128·704, 2816]` fp8 — half a gigabyte widened, never one tile — so
    /// each pair's contraction walks `nb`-wide column blocks of its expert's
    /// slab ([`Self::expert_w_block`]'s budget). The optimizer's `recognize`
    /// would re-tile a recognized contraction and DROP the W corner
    /// (`matmul_tile.rs` keeps `a_row`, discards the W one — every expert
    /// would read slab 0), so this emission must not match its forms: the
    /// plain form's store must drain the matmul result and it does not (the
    /// scale multiply intervenes), and the fp8 form requires the scale
    /// operand to be a `[1, n]`-shaped access-tile LOAD — the scale here is
    /// a rank-1 `[nb]` slice of the expert's scale block, broadcast to
    /// `[1, nb]` by an `linalg.broadcast` the recognizer never looks
    /// through. Both arms refuse; the blocking below is the only one.
    ///
    /// ⛔ AND THE PER-TOKEN LOOP IS OVER PAIRS, NOT TOKENS. Pair `p`'s token
    /// row is `p / k` and its slot is `p % k` — the flat `(n, slot)`
    /// enumeration the kernel's `nk = tid.z` is, stated as emit-time
    /// constants because `m` and `k` are graph constants.
    fn expert_matmul(
        &mut self,
        rows_r: &TensorRegion,
        idx_r: &TensorRegion,
        w_r: &TensorRegion,
        s_r: &TensorRegion,
        out: &TensorRegion,
        k: u32,
    ) {
        let m = out.region.rows.len;
        let n = out.region.cols.len / k;
        let in_dim = rows_r.region.cols.len / k;
        debug_assert_eq!(
            out.region.cols.len,
            n * k,
            "ExpertMatmul output is the [m, k·n] pair layout"
        );
        debug_assert_eq!(
            rows_r.region.cols.len,
            in_dim * k,
            "ExpertMatmul rows are the [m, k·in] pair rows"
        );
        let a = self.a;
        // The stacked banks, viewed once: fp8 codes `[E·n, in]` (1 byte per
        // element — `ktdp.load` widens on read), scales RANK-1 `[E·n]` (bf16
        // staged, f16 in HBM — one scale per bank row, read as the flat
        // sequence so a `[bw]` slice at the block's rows is one window).
        // The bank extents come from the weight REGIONS' own shapes — the
        // launch source's declared `[E·out, in]`.
        let bank_rows = w_r.region.rows.len;
        let w_ptr = self.arg_for(w_r.tensor);
        let w_view = self.view_fp8(w_ptr, bank_rows, in_dim);
        let s_ptr = self.arg_for(s_r.tensor);
        let s_view = self.view_of(s_ptr, vec![i64::from(bank_rows)], vec![1]);
        // The expert indices, loaded whole as their `[m, k]` tile.
        let idx = self.load_region(idx_r);
        let nb = Self::expert_w_block(n, in_dim);
        for p in 0..m * k {
            let (n_tok, slot) = (p / k, p % k);
            // ⭐ THE PAIR'S EXPERT ID — a scalar read out of the indices
            // tile, integerized, and multiplied by the slab height: the
            // dynamic row corner of every window this pair opens into the
            // bank. `tensor.extract` of an f16 tile yields an f32 scalar;
            // `arith.fptosi` narrows it to the i64 the index arithmetic
            // takes (the tile path's i32 is for whole tiles — the SCALAR
            // path is i64, which `construct_access_tile`'s corner reader
            // accepts).
            let e_f = {
                let (ri, ci) = (self.idx(n_tok), self.idx(slot));
                let v = self.fresh();
                let op = Operation::new(a, Some(v), OpKind::TensorExtract, &[idx, ri, ci]);
                let op = self.typed(op, IrType::Scalar(KTIR_ELEM));
                self.push(op);
                v
            };
            let e_i = {
                let v = self.fresh();
                let op = Operation::new(a, Some(v), OpKind::ArithFptosi, &[e_f]);
                let op = self.typed(op, IrType::Scalar(ktir_core::dtypes::DType::I64));
                self.push(op);
                v
            };
            let slab_stride = self.idx(n);
            let row_off = self.index_op(OpKind::ArithMuli, e_i, slab_stride);
            // The pair's activation row: slot `slot`'s `[in]` block of the
            // token's pair rows, rank-1 (the contraction's A is a vector —
            // the kernel's matrix-VECTOR product at one token).
            let x = self.load_region(&sub_cols(&sub_rows(rows_r, n_tok, 1), slot * in_dim, in_dim));
            // The output column blocks of this pair's slot.
            let mut c_off = 0u32;
            while c_off < n {
                let bw = nb.min(n - c_off);
                // The expert's weight block `[bw, in]` — rows `row_off +
                // c_off ..+bw` of the stacked `[E·n, in]` bank, ALL its
                // columns (the output-column block is a ROW block of the
                // on-disk `[out, in]` slab), fp8 widened to f16 on load.
                let w_coff = self.idx(c_off);
                let w_row = self.index_op(OpKind::ArithAddi, row_off, w_coff);
                let w_zero = self.idx(0);
                let w_acc = self.tile(w_view, w_row, w_zero, bw, in_dim);
                let w_val = self.load_tile(w_acc, bw, in_dim);
                // Transpose-B contraction `[1, in] · [bw, in]ᵀ` — the bank's
                // on-disk `[out, in]` read in place, the same maps the
                // dense [`Self::matmul_fp8`] states.
                let dims = vec![1i64, i64::from(bw)];
                // F32 seed, for the reason [`Self::matmul_fp8`]'s init states:
                // `accumulate_outs` rounds to the SEED's dtype, and an f16 seed
                // rounds the UN-scaled code-dot to f16 (inf).
                let init = self.f32_splat(0.0, dims.clone());
                let maps: Vec<AffineMap<'static>> = [[0i64, 2], [1, 2], [0, 1]]
                    .iter()
                    .map(|mm| AffineMap {
                        num_dims: 3,
                        num_syms: 0,
                        exprs: a.exprs(
                            mm.iter()
                                .map(|d| AffineExpr::Dim(*d as usize))
                                .collect(),
                        ),
                    })
                    .collect();
                let part = self.fresh();
                let op = Operation::new(a, Some(part), OpKind::LinalgMatmul, &[x, w_val, init])
                    .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims.clone())))
                    .with_attr(a, AttrKey::IndexingMaps, Attr::AffineMapList(a.maps(maps)));
                let ty = self.tensor_ty(dims.clone());
                let op = self.typed(op, ty);
                self.push(op);
                // ⭐ THE PER-CHANNEL DEQUANT — rank-1 `[bw]` slice of the
                // expert's scale block (rows `[row_off + c_off, +bw)` of the
                // stacked `[E·n, 1]` bank, read as the bank's flat `[E·n]`
                // sequence), inserted an axis to the result's `[1, bw]`. The
                // rank-1 load + broadcast shape is what keeps
                // `matmul_tile::recognize`'s fp8 arm — which requires a
                // directly-loaded `[1, n]` scale tile — off this contraction;
                // see the method doc.
                let s_rank1 = {
                    // The scale rows are `row_off + c_off ..+bw` — the same
                    // rows the weight block read — as a RANK-1 `[bw]` slice
                    // of the bank's flat `[E·n]` scale sequence (the
                    // `[E·n, 1]` bank read as one column).
                    let s_coff = self.idx(c_off);
                    let s_row = self.index_op(OpKind::ArithAddi, row_off, s_coff);
                    let acc = self.fresh();
                    let dims1 = vec![i64::from(bw)];
                    let op = Operation::new(
                        a,
                        Some(acc),
                        OpKind::KtdpConstructAccessTile,
                        &[s_view, s_row],
                    )
                    .with_attr(a, AttrKey::Shape, Attr::IntList(a.ints(dims1.clone())));
                    let op = self.typed(
                        op,
                        IrType::AccessTile {
                            dims: a.ints(dims1.clone()),
                        },
                    );
                    self.push(op);
                    let v = self.fresh();
                    let op = Operation::new(a, Some(v), OpKind::KtdpLoad, &[acc]).with_attr(
                        a,
                        AttrKey::Shape,
                        Attr::IntList(a.ints(dims1.clone())),
                    );
                    let ty = self.tensor_ty(dims1);
                    let op = self.typed(op, ty);
                    self.push(op);
                    v
                };
                // Insert axis 0: `[bw] → [1, bw]` — the broadcast's
                // insertion form (`self.broadcast` with `dim = 0`).
                let s_val = self.broadcast(s_rank1, dims.clone(), 0);
                let scaled = self.binop(OpKind::ArithMulf, part, s_val, dims);
                // The slot's `c_off` column block of this token's output row.
                self.store_region(
                    scaled,
                    &sub_cols(&sub_rows(out, n_tok, 1), slot * n + c_off, bw),
                );
                c_off += bw;
            }
        }
    }

    /// A reshape — the WHOLE source region loaded through its own view, stored through the
    /// output's. The load/store views carry each side's `[rows, cols]`, so the emit door reads
    /// both extents off the program's own parameters and decomposes into single-stick copies;
    /// this body's job is only to move all `r·c` elements once, which a whole-tensor load +
    /// store does exactly.
    ///
    /// ⛔ NO `tensor.reshape` OP EXISTS HERE, DELIBERATELY: KTIR's op set has none, and the
    /// emulator's `eval_dag` oracle already treats a reshape as the identity over the flat
    /// sequence — a load/store pair through two differently-shaped views is exactly that, with
    /// the views stating both extents.
    fn reshape(&mut self, x_r: &TensorRegion, out: &TensorRegion) {
        let x = self.load_region(x_r);
        self.store_region(x, out);
    }

    /// The expert MLP's gated activation — `act(gate) · up` over the pair rows,
    /// [`Self::silu_mul`]'s computation with the act the block declares: silu
    /// longhand for [`GatedAct::Silu`], the gelu tanh POLYNOMIAL for
    /// [`GatedAct::Gelu`] (the function the DDL's `gelufwd` primitive computes,
    /// the same contract [`Self::gelu`] states for the dense elementwise).
    fn gated_act(
        &mut self,
        gate_r: &TensorRegion,
        up_r: &TensorRegion,
        out: &TensorRegion,
        act: scratchy_subtile::subtile_ir::GatedAct,
    ) {
        // Row blocks that fit, the same budget and the same reason as `silu_mul`:
        // eight tiles live at once here in the worst case (gate, up, the act's
        // temporaries, the result).
        let cols = out.region.cols.len;
        let blk = rows_per_block(cols, 8);
        let mut off = 0u32;
        while off < out.region.rows.len {
            let h = blk.min(out.region.rows.len - off);
            let dims = vec![i64::from(h), i64::from(cols)];
            let gate = self.load_region(&sub_rows(gate_r, off, h));
            let up = self.load_region(&sub_rows(up_r, off, h));
            let a = match act {
                scratchy_subtile::subtile_ir::GatedAct::Silu => {
                    let neg = self.negate(gate, dims.clone());
                    let e = self.unop(OpKind::MathExp, neg, dims.clone());
                    let one = self.splat_one(dims.clone());
                    let denom = self.binop(OpKind::ArithAddf, one, e, dims.clone());
                    self.binop(OpKind::ArithDivf, gate, denom, dims.clone())
                }
                scratchy_subtile::subtile_ir::GatedAct::Gelu => self.gelu(gate, dims.clone()),
            };
            let y = self.binop(OpKind::ArithMulf, a, up, dims);
            self.store_region(y, &sub_rows(out, off, h));
            off += h;
        }
    }

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

        // ⛔ THE PRE-SCALE CODE-DOT IS COMPUTED IN f32, NOT f16. The contraction
        // runs on the RAW e4m3 codes (the hardware `matmulfp8` in-fold applies
        // the scale INSIDE the accumulate); the emulator lane spells it as a
        // plain matmul followed by the scale `mulf`, so the matmul's output is
        // the UN-scaled code-dot. A checkpoint's codes are large precisely
        // because its per-channel scales are small (gemma-4-26b q_proj: codes
        // rms 84, scales ~2.6e-4) — a code-dot of a 2816-deep row against a
        // normed activation reaches ~1e5, which is INF in f16 (max 65504), and
        // MEASURED at the 26b's first prefill: t1053 (q_proj out) went inf at
        // seg1, NaN by the first rmsnorm after it, and the forward died in
        // `argmax`'s NaN compare. The interpreter's `matmul2d` keys the result
        // tile's dtype on the ACTIVATION's, so the pre-scale chain stays F32
        // only if the outs seed here is F32 too — `accumulate_outs` rounds the
        // sum to the seed's dtype, and an f16 seed would round the code-dot
        // right back to f16 (inf). The scale `mulf` then computes in f32 and
        // rounds the SCALED value once, which the region's f16 holds.
        let dims = vec![i64::from(m), i64::from(n)];
        let init = self.f32_splat(0.0, dims.clone());
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
        let ty = self.f32_ty(dims.clone());
        let op = self.typed(op, ty);
        self.push(op);

        // ⭐ DEQUANT IS A COLUMN-WISE MULTIPLY — arithmetic the contraction owes, not a tiling
        // decision, so it stays here. `wscale` is the checkpoint's `[1, n]` per-output-column row and
        // scales every column this contraction computed. The product computes in f32 and rounds
        // once to f16 — the SCALED value, which the region's f16 holds (the overflow was only ever
        // the pre-scale code-dot above).
        let s_val = {
            let acc = self.tile(scale_view, zero, zero, 1, n);
            self.load_tile(acc, 1, n)
        };
        let scaled = self.binop(OpKind::ArithMulf, part, s_val, dims);

        self.store_tile(out_view, a_row, zero, m, n, scaled);
    }

    fn rope(&mut self, tensors: RopeTensors, geom: RopeGeometry) {
        let RopeTensors {
            x_t,
            cos_t,
            sin_t,
            out_t,
        } = tensors;
        // The cos/sin tables are pre-tiled to the rope's full width by the host, so position `ri`'s
        // row starts at `ri · tbl_cols`; the first `half` of each row holds the rotary values, and
        // every head's slice is identical.
        let RopeGeometry {
            cols,
            hd,
            rows,
            tbl_cols,
        } = geom;
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
            mask: self
                .mask
                .map(|(t, _)| ktir_superdsc::ktir_node::BufferId::new(t)),
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
