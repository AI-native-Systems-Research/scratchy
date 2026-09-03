//! Bridge 1: `SubtileTape -> TileOp` (TileIR). One [`TileOp`] declaration per `Instr::Compute`,
//! derived MECHANICALLY from the SubtileIR node's `SubOp` kind + its output/input `Region` shapes —
//! no hand-picked residency closure, no per-call-site `Vec<ItDim>`, and (the point) NO branch on the
//! row count. `rows` is read off the node's own output region and fed to the iteration domain as an
//! extent; nothing here inspects `rows > 1` to choose a different declaration.
//!
//! PURE SYNTAX, by construction: every arm below reads only `n.op`'s own kind, `n.output.region`'s
//! shape, and `n.inputs.len()` — never a decision about batching, scheduling, or decomposition. That
//! belongs to the bridges downstream (`tile_op_tiled_op` for scheduling, `tiled_op_sdsc_op` for
//! composite decomposition). Coverage is EXPLICIT: a `SubOp` kind not yet mapped is a returned `Err`
//! naming the node, never a silently-wrong `TileOp`.

use crate::ir::island::tile_op::{TileOp, TileOpKind};
use scratchy_subtile::subtile_ir::{SubOp, SubtileIR, SubtileNode};
use scratchy_subtile::subtile_tape::{Instr, SubtileTape};
use scratchy_subtile::superdsc_opspec::{Df, ItDim};

/// One `Instr::Compute`, lowered: the tape instruction's node id (so a caller can still find which
/// SubtileIR node this came from) plus the `TileOp`(s) derived from it. MOST `SubOp` kinds decompose
/// into exactly one `TileOp` (`tile_ops.len() == 1`); a structurally-composite node (RoPE's rotate
/// matmul + 3 pointwise steps) decomposes into several DISTINCT `TileOp`s, one per constituent
/// op-shape — never one dishonestly-averaged declaration standing in for all of them.
pub struct TiledCompute {
    pub node: scratchy_subtile::subtile_ir::SubtileId,
    pub tile_ops: Vec<TileOp>,
}

/// Lower every `Instr::Compute` in `tape` to a `TiledCompute`, in tape order. `AllocSlot`/`FreeSlot`/
/// `OpenLoop`/`CloseLoop` carry no computation of their own (they are the tape's slot-lifecycle and
/// loop-nesting structure) and are skipped here — only `Compute` instructions produce `TileOp`s.
///
/// `Err` names the FIRST unhandled node (kind + id) rather than emitting a guessed `TileOp` — the
/// same "unrepresentable, not silently wrong" discipline the rest of the emitter uses.
pub fn lower_tape_to_tile_ir<F: scratchy_subtile::subtile_ir::RopeForm>(
    tape: &SubtileTape,
    graph: &SubtileIR<F>,
) -> Result<Vec<TiledCompute>, String> {
    let mut out = Vec::new();
    for instr in tape.instrs() {
        if let Instr::Compute { node, .. } = instr {
            let n = &graph.nodes[node.index()];
            let tile_ops = node_to_tile_ops(n)?;
            out.push(TiledCompute {
                node: *node,
                tile_ops,
            });
        }
    }
    Ok(out)
}

/// Derive the `TileOp`(s) for a `SubtileNode` — the mechanical `SubOp → (TileOpKind, iteration
/// domain)` mapping, one entry per DISTINCT op-shape the node's real lowering emits. `rows`/`cols`
/// come from `node.output.region` (the node's OWN shape, whatever it is — `rows==1` for a decode node
/// and `rows==31` for a prefill node hit the SAME arms below); nothing here reads `rows` to pick a
/// different arm — a node whose REAL lowering structurally differs by row count (RoPE's prefill
/// per-row loop) is instead modeled as a DIFFERENT SET of `TileOp`s for that shape, still derived
/// mechanically from `n.output`/`n.inputs`, never a hidden branch. The iteration domain mirrors the
/// existing builders' `[mb, out, y]` convention (`reduce_opspec_df`/`pointwise_broadcast_opspec_df` in
/// `lower_subtile_tape_to_superdsc.rs`) so the TileIR→SdscOp pass reuses their tensor-layout logic
/// unchanged. `pub(crate)`: called directly by the LIVE per-node lowering (`lower_elementwise_node`
/// et al. in `lower_subtile_tape_to_superdsc.rs`) — a node needs no tape/graph context beyond itself
/// to become `TileOp`(s), so the live path calls this without going through a whole `SubtileTape`.
pub(crate) fn node_to_tile_ops<F: scratchy_subtile::subtile_ir::RopeForm>(
    n: &SubtileNode<F>,
) -> Result<Vec<TileOp>, String> {
    let rows = n.output.region.rows.len;
    let cols = n.output.region.cols.len;
    let mb = ItDim {
        name: "mb",
        size: rows,
        is_reduction: false,
        is_stick: false,
        df: Df::Fp16,
    };
    let out_active = ItDim {
        name: "out",
        size: cols.max(1),
        is_reduction: false,
        is_stick: true,
        df: Df::Fp16,
    };
    let y = ItDim {
        name: "y",
        size: 1,
        is_reduction: false,
        is_stick: false,
        df: Df::Fp16,
    };

    match n.op {
        // The standard pointwise/reduce tile over `[mb, out_active, y]`:
        // the ops differ only in operand count, which
        // `crate::op_abi` declares. One arm, so adding such an op is a
        // table row and not an arm here.
        // Not members of the standard tile class: the shared front end expresses the
        // whole arch vocabulary now, so these reach every target and each one answers
        // for itself. Enumerated rather than `_` so a new SubOp is E0004 here.
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
        | SubOp::Mean => Err(format!("{:?} is not a spyre standard tile op", n.op)),
        crate::op_abi::spyre_standard_tile_pat!() => {
            let n_operands = match crate::op_abi::spyre_standard_operands(&n.op)
                .expect("spyre_standard_tile_pat member without an operand row")
            {
                crate::op_abi::Operands::FromInputs => n.inputs.len() as u32 + 1,
                crate::op_abi::Operands::Fixed(k) => k,
            };
            Ok(vec![TileOp {
                kind: TileOpKind::PointwiseOrReduce { n_operands },
                dims: vec![mb, out_active, y],
                df: Df::Fp16,
            }])
        }
        // ⛔ NOT A POINTWISE TILE. A reshape re-lays the buffer out (the device
        // layout is a function of the ROW COUNT), so it is a restickify, not an
        // elementwise walk over `[mb, out_active, y]`. Refused here rather than
        // silently admitted to the standard class, whose dims would describe
        // the wrong movement.
        SubOp::Reshape => Err(
            "SubOp::Reshape is a restickify, not a standard tile — the device layout is a \
             function of the ROW COUNT, so this is a re-laying copy the SuperDSC lowering must \
             emit, never a pointwise walk"
                .to_string(),
        ),
        // A matmul K-chunk's OUTPUT is the dense [mr, nr] partial; `out` = nr (the N/stick dim), `in`
        // (K) comes from the FIRST input's column extent (the A-slice's [mr, kr]), matching how
        // `matmul_lx_resident` reads `per_core_extent("in")`.
        SubOp::MatmulTile { .. } => {
            let k = n.inputs.first().map(|t| t.region.cols.len).unwrap_or(1);
            let dims = vec![
                mb,
                ItDim {
                    name: "in",
                    size: k.max(1),
                    is_reduction: true,
                    is_stick: false,
                    df: Df::Fp16,
                },
                out_active,
                y,
            ];
            Ok(vec![TileOp {
                kind: TileOpKind::Matmul,
                dims,
                df: Df::Fp16,
            }])
        }
        // SumReduce is a split-K COMBINE — elementwise over equal-shaped inputs, not a stick reduce.
        // n_operands = inputs + the output (mirrors `pointwise_lx_resident`'s convention).
        // Shape-preserving elementwise: unary (Silu) reads 1 input, binary (Mul/Add) reads 2 — both
        // plus the output give `n_operands`.
        // Fused gate/up SiluMul: 2 inputs (gate, up) + output.
        // ScalarMul: 1 tensor input (the scalar rides in the op, not as an operand) + output. DEVICE
        // width, not the logical `cols`: a ScalarMul on padded logits (`[.,49159]`) must address the
        // SAME device layout its producer matmul emitted (`DeviceWidth::for_output`, ≥2^20 macs bumps
        // to ≥8-core-splittable) — `for_pointwise` is that rule for a pointwise CONSUMER, a pure
        // function of `cols` alone (no `m`/rows term), matching `lower_scalarmul_node` exactly. A
        // no-op for 64-aligned tensors (residual/embedding), so decode/prefill/padded-logits all hit
        // this ONE formula.
        SubOp::ScalarMul { .. } => {
            let dev_cols =
                crate::lower_subtile_tape_to_superdsc::DeviceWidth::for_pointwise(cols).get();
            let out_dev = ItDim {
                name: "out",
                size: dev_cols.max(1),
                is_reduction: false,
                is_stick: true,
                df: Df::Fp16,
            };
            Ok(vec![TileOp {
                kind: TileOpKind::PointwiseOrReduce { n_operands: 2 },
                dims: vec![mb, out_dev, y],
                df: Df::Fp16,
            }])
        }
        // RmsNormReduce is the reduce phase (mean(x²) over the WHOLE row) — a genuine stick reduce,
        // `is_reduction: true` on `out`, output region is `[m,1]` (the per-row scalar), so `cols` here
        // is the whole-row extent read from the ONE input (x `[m, hidden]`), not the 1-wide output.
        SubOp::RmsNormReduce { .. } => {
            let hidden = n
                .inputs
                .first()
                .map(|t| t.region.cols.len)
                .unwrap_or(cols.max(1));
            let reduce_out = ItDim {
                name: "out",
                size: hidden.max(1),
                is_reduction: true,
                is_stick: true,
                df: Df::Fp16,
            };
            Ok(vec![TileOp {
                kind: TileOpKind::PointwiseOrReduce { n_operands: 2 }, // data + accum
                dims: vec![mb, reduce_out, y],
                df: Df::Fp16,
            }])
        } // RmsNormApply (x * inv_rms * gamma) and the whole-form RmsNorm are both shape-preserving
          // elementwise over the OUTPUT's [rows,cols] — 3 tensor inputs (x, inv_rms/mean+rinv, gamma) + output.
          // RoPE (RopeRotate/RopeAppend): PURE SYNTAX — the node's OWN declared shape and input count,
          // exactly like every other arm above (Elementwise/SiluMul/RmsNormApply). NOT decomposed into
          // its constituent primitive ops (rotate-matmul + xc + rs + add) here: that decomposition, and
          // any decision about how to batch/split rows or heads for the device, is a SEPARATE concern —
          // either a later IR-to-IR transformation on TileIR, or the TileIR→SdscOp step itself (which is
          // where "translate to a lower level of abstraction" legitimately happens). Lowering only
          // TRANSLATES; it does not SCHEDULE. `n_operands` = every declared input (RopeRotate: 3;
          // RopeAppend: 6, including the cache-write operands `lower_rope_node` currently drops from its
          // OWN rotation math) + the output — no cherry-picking which inputs "count".
          // AttnDecode: PURE SYNTAX, same convention as every arm above — the node's own [rows,cols]
          // output shape and its own declared input count (Q + alternating (K_seg,V_seg) pairs) + the
          // output. This does NOT characterize attention as "really" softmax(QKᵀ/√d)V internally, nor
          // decide per-head/GQA/KV-cache-placement structure — that decomposition (the eventual
          // TileIR→SdscOp step, or a later explicit transform) is exactly the concern the RoPE fix above
          // says does NOT belong in lowering. `PointwiseOrReduce{n_operands}` is a true, non-invented
          // statement here: the node reads `n_operands-1` tensors and writes 1, at this shape — nothing more.
    }
}

/// Convenience for a caller that KNOWS (by `SubOp` kind) its node decomposes into exactly one
/// `TileOp` — `Elementwise`/`SumReduce`/`SiluMul`/`ScalarMul`/`RmsNorm*`/`MatmulTile`, everything
/// except the composite RoPE mapping. `Err` if that expectation is violated (a future `node_to_tile_ops`
/// change making a kind multi-op would surface here as a build-time-adjacent error, not a silent
/// wrong pick of `tile_ops[0]`).
pub(crate) fn node_to_single_tile_op<F: scratchy_subtile::subtile_ir::RopeForm>(
    n: &SubtileNode<F>,
) -> Result<TileOp, String> {
    let mut ops = node_to_tile_ops(n)?;
    if ops.len() != 1 {
        return Err(format!(
            "node {}: expected exactly 1 TileOp, node_to_tile_ops produced {}",
            n.id.index(),
            ops.len()
        ));
    }
    Ok(ops.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    // `EwKind` is named by the assertion below. Without it this module has
    // NEVER compiled, so every test in it was inert — it asserted nothing while
    // looking like coverage.
    use scratchy_subtile::subtile_ir::{EwKind, ValidatedGraph, lower_region};
    use scratchy_subtile::subtile_tape::lower_dag_to_tape;
    use std::num::NonZeroU32;

    /// The actual end-to-end proof: build a REAL `SubtileIR` + `SubtileTape` from the `add_only`
    /// fixture (untiled — `nb` >= the 2048-col shape, so `lower_region` emits ONE `Elementwise(Add)`
    /// node, no `SumReduce` combine) and run this pass over it. A non-empty, error-free result means
    /// this is a working lowering, not just code that happens to compile.
    #[test]
    fn add_only_fixture_lowers_to_tile_ir() {
        let input = scratchy_subtile::fixtures::add_only_input();
        let nb = NonZeroU32::new(2048).unwrap();
        let rg = lower_region(&input, nb);
        let valid = ValidatedGraph::new(&rg).expect("add_only fixture validates");
        let tape = lower_dag_to_tape(&valid);
        assert!(
            !tape.instrs().is_empty(),
            "the tape has at least one Compute"
        );

        let tiled = lower_tape_to_tile_ir(&tape, &rg).expect("Add is a mapped SubOp kind");
        assert!(
            !tiled.is_empty(),
            "at least one Compute -> at least one TiledCompute"
        );
        for tc in &tiled {
            let node = &rg.nodes[tc.node.index()];
            assert!(matches!(node.op, SubOp::Elementwise(EwKind::Add)));
            // The TileOp's iteration domain carries the SAME extents as the node's own output region —
            // proving the lowering reads shape from the node, not a hardcoded constant.
            assert_eq!(
                tc.tile_ops.len(),
                1,
                "a plain Elementwise node is exactly one TileOp"
            );
            let tile_op = &tc.tile_ops[0];
            let mb_dim = tile_op.dims.iter().find(|d| d.name == "mb").unwrap();
            let out_dim = tile_op.dims.iter().find(|d| d.name == "out").unwrap();
            assert_eq!(mb_dim.size, node.output.region.rows.len);
            assert_eq!(out_dim.size, node.output.region.cols.len);
            assert!(matches!(
                tile_op.kind,
                TileOpKind::PointwiseOrReduce { n_operands: 3 }
            ));
        }
    }

    /// `rows==1` (decode-shaped: the fixture's `m=1`) hits the EXACT SAME `node_to_tile_ops` arm as any
    /// other row count would — there is no `rows > 1` branch in this file to diverge from it. Asserted
    /// directly: the `mb` dim size equals the fixture's `m=1`.
    #[test]
    fn m_equals_1_is_just_an_extent() {
        let input = scratchy_subtile::fixtures::add_only_input();
        let nb = NonZeroU32::new(2048).unwrap();
        let rg = lower_region(&input, nb);
        let valid = ValidatedGraph::new(&rg).expect("validates");
        let tape = lower_dag_to_tape(&valid);
        let tiled = lower_tape_to_tile_ir(&tape, &rg).expect("mapped");
        let mb_dim = tiled[0].tile_ops[0]
            .dims
            .iter()
            .find(|d| d.name == "mb")
            .unwrap();
        assert_eq!(mb_dim.size, 1, "add_only_input is m=1 by construction");
    }

    /// RoPE lowers as PURE SYNTAX: ONE `TileOp` reflecting the node's own `[rows,cols]` shape and its
    /// OWN input count — NOT decomposed into its constituent primitive ops (rotate-matmul + xc + rs +
    /// add) at this stage. That decomposition is a SEPARATE concern (an explicit later transformation,
    /// or the TileIR→SdscOp step), never smuggled into the SubtileTape→TileIR translation. Proves
    /// decode (rows=1, this fixture) needs NO special case: it hits the exact same arm any row count
    /// would, with the SAME `n_operands = 3 inputs + 1 output` any RopeRotate node has.
    #[test]
    fn rope_lowers_as_one_undecomposed_tile_op() {
        let input = scratchy_subtile::fixtures::rope_rotate_only_input();
        let nb = NonZeroU32::new(4096).unwrap(); // wide enough to keep this untiled
        let rg = lower_region(&input, nb);
        let valid = ValidatedGraph::new(&rg).expect("rope_rotate_only fixture validates");
        let tape = lower_dag_to_tape(&valid);
        let tiled = lower_tape_to_tile_ir(&tape, &rg).expect("RopeRotate is a mapped SubOp kind");
        assert_eq!(
            tiled.len(),
            1,
            "rope_rotate_only_input is a single RopeRotate node"
        );
        let ops = &tiled[0].tile_ops;
        assert_eq!(ops.len(), 1, "one node -> one TileOp, undecomposed");
        assert!(
            matches!(ops[0].kind, TileOpKind::PointwiseOrReduce { n_operands: 4 }),
            "3 declared inputs (x, cos, sin) + 1 output"
        );
        let mb_dim = ops[0].dims.iter().find(|d| d.name == "mb").unwrap();
        assert_eq!(
            mb_dim.size, 1,
            "rope_rotate_only_input is rows=1 by construction"
        );
    }
}
