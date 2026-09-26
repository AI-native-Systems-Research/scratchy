//! M2 front-end swap: the metal instruction stream built from the
//! SHARED subtile decode tape (`to_wavefront::LoweredDecode` — the same
//! artifact spyre lowers), NOT from instruction selection.
//!
//! The stream this produces is asserted EQUAL (instruction-for-
//! instruction, barrier-for-barrier, post-loop-compression) to the
//! instruction-selection output at every expansion until the flip;
//! after the flip, instruction selection stops running for metal and
//! this is the only producer.
//!
//! Structure: three local rewrites on the shared tape (rope fold,
//! add+rmsnorm fold, silu·mul-vs-fused-mlp — the per-preset shapes),
//! then a 1:1 mapping to `Instruction` values. Arena slots come from
//! the SAME `colored_slot_map` authority via `LoweredDecode::op_tiles`;
//! layers come from the weight bindings' unroll indices; quantization
//! comes from the tape's `GemmWeight`.

use scratchy_ir::Instruction;
use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::handoff::{SlotMap, WeightKind, WeightSlot};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweredOp};

/// The per-model facts the mapping needs beyond the tape itself.
pub struct StreamFacts<'a> {
    /// `WeightId` → dotted path, for accessor base names.
    /// Accessor path per `WeightId`, precomputed by the caller. The `WeightTable` it comes
    /// from is the shared compiler's, so the DATA crosses the boundary rather than the type.
    pub weight_paths: &'a [Vec<String>],
    /// Accessor bases whose DSL subtree declares its own `shared_expert`. Single-owner rule:
    /// when the DSL owns that subtree, a fused MoE op's internal shared tail is disabled.
    /// Precomputed by the caller for the same reason as `weight_paths` — the `WeightTable` it
    /// is queried from belongs to the shared compiler.
    pub dsl_shared_expert_bases: &'a [String],
    /// The program + model, for the shared per-instruction passes the
    /// instruction-selection route ran (mixed-bit MoE repack).
    /// Emission rank per FUF tile, flattened from the scheduler's wave
    /// order — the SAME order the instruction-selection walk emitted
    /// in. Ops sort by (rank, tape index); the ten uniform families'
    /// tape order already coincides, and parallel-branch arches
    /// (gemma4-moe) get the wave interleaving.
    pub wave_rank: &'a std::collections::HashMap<u32, usize>,
    /// The embedding table's accessor base name (the embed weight is
    /// not a tape source — the tape starts at the embedded hidden).
    pub embed_base: String,
    pub hidden_size: u32,
    /// `Some((group_size, bits))` when the embedding table is
    /// affine-quantized (AffineEmbed); `None` → dense Embed.
    pub embed_quant: Option<(u32, u32)>,
    /// Whether the fused-MLP instruction (FusedGateUpSiluMul) is in
    /// play for this preset (dense) or the split qmm+SiluMul shape
    /// (affine quant).
    pub fused_mlp: bool,
    /// `intermediate_size` — the SiluMul width arg.
    pub intermediate_size: u32,
}

fn slot_of(lowered: &LoweredDecode, slots: &SlotMap, op_idx: usize) -> Result<u32, String> {
    let (tile, out_slot) =
        lowered.op_tiles[op_idx].ok_or_else(|| format!("op {op_idx}: no tile provenance"))?;
    // A missing key is a REFUSAL, not a panic: under tape scheduling
    // the coloring comes from `tape_slot_map`, so an unregistered
    // tile means the walk gave this op no provenance while emission
    // still wants its slot. Naming the op AND its kind turns that
    // into a one-line diagnosis instead of a bare SlotMap panic.
    if !slots
        .entries()
        .any(|((t, sl), _)| *t == scratchy_subtile::handoff::TileId(tile) && *sl == out_slot)
    {
        return Err(format!(
            "op {op_idx} ({}) resolves tile {tile}#{out_slot}, which the coloring never              registered",
            scratchy_subtile::ops::lowered_op_name(&lowered.input.ops[op_idx].op),
        ));
    }
    Ok(slots.of(scratchy_subtile::handoff::TileId(tile), out_slot))
}

fn layer_of(lowered: &LoweredDecode, ext: usize) -> u32 {
    match &lowered.bindings[ext] {
        SourceBinding::Weight { index, .. } | SourceBinding::WeightScale { index, .. } => {
            index.map(|u| u.0 as u32).unwrap_or(0)
        }
        _ => 0,
    }
}

/// Layer arg honoring the trailing-numeral path rule: the unroll index
/// when present, else a trailing numeral path segment (merger_mlp.2).
fn layer_of_with_path(lowered: &LoweredDecode, facts: &StreamFacts<'_>, ext: usize) -> u32 {
    match &lowered.bindings[ext] {
        SourceBinding::Weight { id, index } | SourceBinding::WeightScale { id, index } => {
            if let Some(u) = index {
                return u.0 as u32;
            }
            let joined = facts.weight_paths[*id as usize].join("_");
            joined
                .rfind('_')
                .and_then(|pos| joined[pos + 1..].parse::<u32>().ok())
                .unwrap_or(0)
        }
        _ => 0,
    }
}

/// Which source an op's first weight input is, if any.
fn weight_ext(lowered: &LoweredDecode, op_idx: usize) -> Option<usize> {
    lowered.input.ops[op_idx]
        .inputs
        .iter()
        .find_map(|r| match r {
            InputRef::Ext(e)
                if matches!(
                    lowered.bindings[*e],
                    SourceBinding::Weight { .. } | SourceBinding::WeightScale { .. }
                ) =>
            {
                Some(*e)
            }
            _ => None,
        })
}

fn base_of(lowered: &LoweredDecode, facts: &StreamFacts<'_>, ext: usize) -> Result<String, String> {
    match &lowered.bindings[ext] {
        SourceBinding::Weight { id, .. } | SourceBinding::WeightScale { id, .. } => {
            let path = &facts.weight_paths[*id as usize];
            // A trailing numeral (segment OR `_<n>` suffix after the
            // join — interning may bake the underscore in) is a LAYER
            // index (nn.Sequential members like merger_mlp.2 — the
            // split_base_layer rule), not part of the accessor base.
            //
            // 🛑 KNOWN DEFECT, measured 2026-08-22: dropping the
            // numeral here is why Qwen2-VL's merger applies
            // `visual.merger.mlp.2` TWICE and never loads
            // `visual.merger.mlp.0` — `merger.mlp_0` and
            // `merger.mlp_2` both truncate to base `merger_mlp`, the
            // accessor group collapses to ONE Unindexed field, and the
            // emitted accessor ignores its layer argument. The fix is
            // NOT to keep the numeral here: the layer is appended
            // again downstream, which yields `merger_mlp_0_0` /
            // `merger_mlp_2_2` and still resolves no safetensors key.
            // It belongs at the accessor-NAME composition site, which
            // must carry (base, layer) as a pair instead of a string.
            let mut joined = path.join("_");
            if let Some(pos) = joined.rfind('_')
                && joined[pos + 1..].parse::<u32>().is_ok()
            {
                joined.truncate(pos);
            }
            Ok(joined)
        }
        other => Err(format!("source {ext}: not a weight ({other:?})")),
    }
}

fn ws(kind: WeightKind, base: String) -> Vec<WeightSlot> {
    vec![WeightSlot { kind, base }]
}

fn rope_rotary_ident(lowered: &LoweredDecode, rope_op: &scratchy_subtile::lower::OpDesc) -> String {
    let local = rope_op.inputs.iter().any(|r| match r {
        InputRef::Ext(e) => matches!(
            lowered.bindings[*e],
            SourceBinding::Cos { local: true } | SourceBinding::Sin { local: true }
        ),
        _ => false,
    });
    if local {
        "rotary_local".to_string()
    } else {
        "rotary".to_string()
    }
}

fn weight_ext_of(od: &scratchy_subtile::lower::OpDesc, lowered: &LoweredDecode) -> Option<usize> {
    od.inputs.iter().find_map(|r| match r {
        InputRef::Ext(e)
            if matches!(
                lowered.bindings[*e],
                SourceBinding::Weight { .. } | SourceBinding::WeightScale { .. }
            ) =>
        {
            Some(*e)
        }
        _ => None,
    })
}

fn gain_add_slot(lowered: &LoweredDecode, slots: &SlotMap, op_idx: usize) -> Option<u32> {
    lowered
        .norm_gain_add_tiles
        .get(&op_idx)
        .map(|t| slots.of(scratchy_subtile::handoff::TileId(*t), 0))
}

/// Slot of an op's SECOND output (multi-output producers: GateSplit's
/// gate, VisionRope's k) through the same coloring authority.
fn second_out_slot(lowered: &LoweredDecode, slots: &SlotMap, op_idx: usize) -> Result<u32, String> {
    lowered.op_tiles[op_idx]
        .map(|(t, _)| slots.of(scratchy_subtile::handoff::TileId(t), 1))
        .ok_or_else(|| format!("op {op_idx}: no provenance for second output"))
}

fn in_op(r: &InputRef) -> Option<usize> {
    match r {
        InputRef::Op(i) => Some(*i),
        InputRef::Ext(_) => None,
    }
}

/// The additive offset the KV operand produced by `op` carries into the
/// cache of `layer`, and — for a bias — the producing projection's accessor,
/// which the writer's weight site must hold (`op_abi::rope_append_bias_slots`).
fn kv_offset_of(
    lowered: &LoweredDecode,
    facts: &StreamFacts<'_>,
    op: usize,
    layer: u32,
) -> Result<(scratchy_ir::KvOffset, Option<WeightSlot>), String> {
    use scratchy_ir::{BiasStorage, KvOffset};
    let Some((gemm, weight)) = scratchy_subtile::lower::kv_operand_bias(&lowered.input.ops, op)?
    else {
        return Ok((KvOffset::Centered, None));
    };
    let e = weight_ext(lowered, gemm).ok_or("kv bias projection: no weight")?;
    if layer_of_with_path(lowered, facts, e) != layer {
        return Err(format!(
            "op {op}: a KV bias from another layer's projection"
        ));
    }
    let storage = match weight {
        GemmWeight::Affine { .. } => BiasStorage::Affine,
        GemmWeight::Dense | GemmWeight::Fp8Dynamic => BiasStorage::Dense,
    };
    let linear = WeightSlot {
        kind: WeightKind::Linear,
        base: base_of(lowered, facts, e)?,
    };
    Ok((KvOffset::LinearBias(storage), Some(linear)))
}

/// The bridge's output for one decode canonical: backbone + lm_head
/// streams (UNCOMPRESSED — the caller runs the same
/// `apply_loop_compression` the instruction-selection route runs) with
/// their per-instruction hazard signatures.
pub struct BridgedStream {
    pub backbone: Vec<Instruction>,
    pub lm_head: Vec<Instruction>,
    pub backbone_sigs: Vec<HazardSig>,
    pub lm_head_sigs: Vec<HazardSig>,
    /// Parallel to `backbone` / `lm_head`: the accessor weight slots
    /// (`WeightAccessors` kind + base name) each instruction resolves
    /// through — what `emit_weight_accessors_impl` consumes.
    pub backbone_weight_slots: Vec<Vec<WeightSlot>>,
    pub lm_head_weight_slots: Vec<Vec<WeightSlot>>,
    /// ⭐ WHICH BACKBONE INSTRUCTIONS EACH SOURCE OP EMITTED: `[start, end)`.
    ///
    /// The bridge walks ops in TAPE order but emits a variable number of instructions per op
    /// (fusions emit none, a folded norm+rope emits one for three ops, the head rows emit
    /// without an op at all). So a loop the shared re-roll found in TAPE STEPS has no meaning on
    /// this stream until the two are related — and this is that relation.
    ///
    /// Without it the only way to find the loop here is to SEARCH the instruction stream for a
    /// repeating run, which is a second re-roll implementation over a second representation.
    /// That is what this replaces.
    pub backbone_span_of_op: Vec<Option<(usize, usize)>>,
}

/// RopeAppendNormed fold info, keyed by the RopeAppend op index:
/// (raw q/k/v op indices, q/k gain weight sources).
pub struct NormedRope {
    q_raw: usize,
    k_raw: usize,
    v_raw: usize,
    q_gain: Option<usize>,
    k_gain: Option<usize>,
}

/// NormAddScalarMul fold info, keyed by the ScalarWeightMul index.
pub struct Nasm {
    delta: usize,
    residual: InputRef,
    norm_gain: usize,
    scalar_w: usize,
}

/// The fold pre-pass result: which tape ops fuse into which emitted
/// instruction, plus the per-fold detail the emission arms read.
/// Computed ONCE per stream; the tape-side slot colorer consumes the
/// same plan (a fused op's output never materializes — only fold
/// winners define slots).
pub struct FoldPlan {
    pub(crate) fused_into: Vec<Option<usize>>,
    pub(crate) consumers: Vec<usize>,
    pub(crate) rope_normed: std::collections::HashMap<usize, NormedRope>,
    pub(crate) msn_bias: std::collections::HashMap<usize, usize>,
    pub(crate) nasm: std::collections::HashMap<usize, Nasm>,
}

/// Run the fold pre-pass over the tape — every fusion decision the
/// emission loop honors, mirroring the instruction-selection
/// matchers exactly (see each block's comments).
pub fn fold_plan(lowered: &LoweredDecode, facts: &StreamFacts<'_>) -> FoldPlan {
    let ops = &lowered.input.ops;
    let n = ops.len();
    // Which ops are consumed by a fold (skipped in the main walk).
    let mut fused_into: Vec<Option<usize>> = vec![None; n];
    // Consumer counts — fusion legality is single-consumer at every
    // folded node (mirrors the instruction-selection matches()).
    let mut consumers: Vec<usize> = vec![0; n];
    for od in ops.iter() {
        // AttnDecode's k/v edges (inputs 3..) alias the rope triple —
        // ONE consumer in the FUF (the rope) shows as two on the tape.
        // Count only the FUF-real edges for fusion legality.
        let real = match od.op {
            LoweredOp::AttnDecode { .. } => &od.inputs[..1],
            _ => &od.inputs[..],
        };
        for r in real {
            if let InputRef::Op(a) = r {
                consumers[*a] += 1;
            }
        }
    }
    if lowered.input.result < n {
        consumers[lowered.input.result] += 1;
    }
    let mut rope_normed: std::collections::HashMap<usize, NormedRope> =
        std::collections::HashMap::new();
    // Norm-op → trailing BiasAdd op folded into MeanSubRmsNormBiasAdd.
    let mut msn_bias: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    // NormAddScalarMul fold info, keyed by the ScalarWeightMul index:
    // (norm op, add op, delta producer, residual producer, norm gain ext,
    // scalar weight ext).
    let mut nasm: std::collections::HashMap<usize, Nasm> = std::collections::HashMap::new();
    // rope fold: RopeRotate(q) + RopeAppend(k,v) → one RopeAppend.
    // add+rmsnorm fold: Add + RmsNorm → FusedAddRmsNorm (the RmsNorm
    // drives; its Add input is folded).
    // mlp shape: Silu + Mul (+ the two gemms when fused_mlp).
    for (i, od) in ops.iter().enumerate() {
        match od.op {
            LoweredOp::RopeAppend { .. } => {
                // Find the sibling RopeRotate feeding the SAME attention
                // (the q rotate) — by convention it's the closest
                // preceding RopeRotate not yet folded.
                let mut j = i;
                while j > 0 {
                    j -= 1;
                    if matches!(ops[j].op, LoweredOp::RopeRotate { .. }) && fused_into[j].is_none()
                    {
                        fused_into[j] = Some(i);
                        break;
                    }
                }
            }
            LoweredOp::RmsNorm { .. } => {
                // Cohere LayerNorm trio: rmsnorm(sub(x, mean(x)), w)
                // → MeanSubRmsNorm. The mean and sub fold into the norm.
                if let Some(InputRef::Op(sb)) = ops[i].inputs.first()
                    && matches!(ops[*sb].op, LoweredOp::Sub)
                    && fused_into[*sb].is_none()
                    && let Some(mu) = in_op(&ops[*sb].inputs[1])
                    && matches!(ops[mu].op, LoweredOp::Mean)
                    && fused_into[mu].is_none()
                {
                    fused_into[*sb] = Some(i);
                    fused_into[mu] = Some(i);
                    // Vision/encoder LayerNorm carries a bias: a
                    // single-consumer BiasAdd on the norm output folds
                    // in (MeanSubRmsNormBiasAdd). The BiasAdd op index
                    // becomes the fold DRIVER so the emission lands at
                    // its position with its output slot.
                    if consumers[i] == 1
                        && let Some((bi, _)) = ops.iter().enumerate().find(|(_, o)| {
                            matches!(o.op, LoweredOp::BiasAdd)
                                && matches!(o.inputs.first(), Some(InputRef::Op(x)) if *x == i)
                        })
                    {
                        msn_bias.insert(i, bi);
                    }
                }
                // Multi-consumer Adds allowed (the fused instruction
                // writes the residual back in place); the WINNING norm
                // is the one with the smallest FUF tile id — "the first
                // we find" in the canonical matcher's node order, which
                // is not always tape order (gemma4-moe's two residual
                // norms).
                if let Some(InputRef::Op(a)) = ops[i].inputs.first()
                    && matches!(ops[*a].op, LoweredOp::Add)
                {
                    let winner = ops
                        .iter()
                        .enumerate()
                        .filter(|(j, o)| {
                            matches!(o.op, LoweredOp::RmsNorm { .. })
                                && matches!(o.inputs.first(), Some(InputRef::Op(x)) if x == a)
                                && fused_into[*j].is_none()
                        })
                        .min_by_key(|(j, _)| {
                            lowered.op_tiles[*j].map(|(t, _)| t).unwrap_or(u32::MAX)
                        })
                        .map(|(j, _)| j);
                    if winner == Some(i) && fused_into[*a].is_none() {
                        fused_into[*a] = Some(i);
                    }
                }
            }
            LoweredOp::Mul => {
                // Silu+Mul → SiluMul; under the fused-MLP preset the
                // gate/up gemms fold in too (FusedGateUpSiluMul).
                if let Some(si) = in_op(&ops[i].inputs[0])
                    && matches!(ops[si].op, LoweredOp::Silu | LoweredOp::Gelu)
                    && fused_into[si].is_none()
                {
                    fused_into[si] = Some(i);
                    if facts.fused_mlp {
                        if let Some(g) = in_op(&ops[si].inputs[0]) {
                            fused_into[g] = Some(i);
                        }
                        if let Some(u) = in_op(&ops[i].inputs[1]) {
                            fused_into[u] = Some(i);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // ── Chain folds (Gemma4) ────────────────────────────────────
    // Peel a rope operand chain: reshape-backs and norms fold into the
    // rope; a pre-norm Reshape VIEW survives as the raw operand.
    // DRY-RUN peel: collect the chain without committing fold marks.
    // The RopeAppendNormed fusion fires ONLY on the Gemma4 signature —
    // a unit-norm on the V chain; per-head gain norms alone (qwen3)
    // stay as their own instructions.
    let peel = |start: usize,
                fused_into: &Vec<Option<usize>>|
     -> (usize, Option<usize>, bool, Vec<usize>) {
        let mut cur = start;
        let mut gain: Option<usize> = None;
        let mut normed = false;
        let mut folded: Vec<usize> = Vec::new();
        loop {
            match &ops[cur].op {
                LoweredOp::Reshape {
                    rows_mult: 1,
                    rows_div: 1,
                    ..
                } if fused_into[cur].is_none() && consumers[cur] == 1 => {
                    folded.push(cur);
                    match in_op(&ops[cur].inputs[0]) {
                        Some(p2) => cur = p2,
                        None => break,
                    }
                }
                LoweredOp::RmsNorm { .. } if fused_into[cur].is_none() && consumers[cur] == 1 => {
                    normed = true;
                    gain = weight_ext_of(&ops[cur], lowered);
                    folded.push(cur);
                    match in_op(&ops[cur].inputs[0]) {
                        Some(p2) => cur = p2,
                        None => break,
                    }
                }
                LoweredOp::RmsNormUnit { .. }
                    if fused_into[cur].is_none() && consumers[cur] == 1 =>
                {
                    normed = true;
                    folded.push(cur);
                    match in_op(&ops[cur].inputs[0]) {
                        Some(p2) => cur = p2,
                        None => break,
                    }
                }
                _ => break,
            }
        }
        (cur, gain, normed, folded)
    };
    for (i, od) in ops.iter().enumerate() {
        if let LoweredOp::RopeAppend { .. } = od.op {
            let rot = fused_into.iter().position(|f| *f == Some(i));
            let Some(rot) = rot else { continue };
            let q0 = in_op(&ops[rot].inputs[0]);
            let k0 = in_op(&od.inputs[0]);
            let v0 = in_op(&od.inputs[3]);
            let (Some(q0), Some(k0), Some(v0)) = (q0, k0, v0) else {
                continue;
            };
            let (q_raw, q_gain, _qn, q_folded) = peel(q0, &fused_into);
            let (k_raw, k_gain, _kn, k_folded) = peel(k0, &fused_into);
            // The canonical matcher takes V WITHOUT reshape peeling:
            // fusion fires only when the rope's V edge IS the unit
            // norm directly (a per-head V view blocks it — the 26b
            // global layer stays unfused, exactly as instruction
            // selection leaves it).
            let (v_raw, vn, v_folded) = if matches!(ops[v0].op, LoweredOp::RmsNormUnit { .. })
                && fused_into[v0].is_none()
                && consumers[v0] == 1
            {
                let raw = in_op(&ops[v0].inputs[0]).unwrap_or(v0);
                (raw, true, vec![v0])
            } else {
                (v0, false, Vec::new())
            };
            if vn {
                for f in q_folded.iter().chain(&k_folded).chain(&v_folded) {
                    fused_into[*f] = Some(i);
                }
                rope_normed.insert(
                    i,
                    NormedRope {
                        q_raw,
                        k_raw,
                        v_raw,
                        q_gain,
                        k_gain,
                    },
                );
            }
        }
        if let LoweredOp::ScalarWeightMul = od.op {
            // scalar_weight_mul(add(rmsnorm(delta, gain), residual), w)
            let Some(a) = in_op(&od.inputs[0]) else {
                continue;
            };
            if !matches!(ops[a].op, LoweredOp::Add) || fused_into[a].is_some() {
                continue;
            }
            let Some(nrm) = in_op(&ops[a].inputs[0]) else {
                continue;
            };
            if !matches!(ops[nrm].op, LoweredOp::RmsNorm { .. }) || fused_into[nrm].is_some() {
                continue;
            }
            let Some(delta) = in_op(&ops[nrm].inputs[0]) else {
                continue;
            };
            let (Some(norm_gain), Some(scalar_w)) =
                (weight_ext_of(&ops[nrm], lowered), weight_ext(lowered, i))
            else {
                continue;
            };
            fused_into[a] = Some(i);
            fused_into[nrm] = Some(i);
            nasm.insert(
                i,
                Nasm {
                    delta,
                    residual: ops[a].inputs[1],
                    norm_gain,
                    scalar_w,
                },
            );
        }
    }
    FoldPlan {
        fused_into,
        consumers,
        rope_normed,
        msn_bias,
        nasm,
    }
}

/// Output width per op, propagated through the tape (norms and
/// elementwise ops preserve their input width; the registry's
/// `lowered_op_out_cols` supplies the rest). Shared by the colorer's
/// pool keys and the emission arms that need an operand's TRUE width
/// (a per-head view's head_dim, not hidden_size).
pub fn op_widths(lowered: &LoweredDecode) -> Vec<u32> {
    let ops = &lowered.input.ops;
    let mut width: Vec<u32> = Vec::with_capacity(ops.len());
    for od in ops.iter() {
        let operand_cols = |k: usize| match od.inputs.get(k) {
            Some(InputRef::Op(j)) => width[*j],
            Some(InputRef::Ext(e)) => lowered.input.sources[*e].cols,
            None => 0,
        };
        width.push(scratchy_subtile::ops::lowered_op_out_cols(
            &od.op,
            operand_cols,
        ));
    }
    width
}

/// Build the decode instruction stream from the shared tape.
pub fn decode_instruction_stream(
    lowered: &LoweredDecode,
    slots: &SlotMap,
    facts: &StreamFacts<'_>,
    // ⭐ NOT AN `Option`. The walk IS the shared tape's — emission order, barrier groups and the
    // layer loop all come from it, and `slots` is therefore the TAPE colouring (solver-coloured
    // lifetimes are unsound under tape order). There used to be a `None` arm that ordered by
    // wave rank instead: the instruction-selection route, a second emission order over the same
    // ops. Its last caller is gone, so the alternative is gone with it and the type says so.
    tape_items: &[scratchy_target_metal::from_tape::TapeItem],
) -> Result<BridgedStream, String> {
    use Instruction as I;
    let ops = &lowered.input.ops;
    let FoldPlan {
        mut fused_into,
        consumers,
        rope_normed,
        msn_bias,
        nasm,
    } = fold_plan(lowered, facts);
    let _ = &consumers;
    let op_width = op_widths(lowered);

    // THE declared-alias invariant, checked once against the real
    // coloring. `metal_alias_operand` says which operand a kernel
    // writes over; the colorer is supposed to have given that operand
    // and this op ONE slot. If they ever disagree the kernel writes
    // into a buffer some other op still owns, and the symptom is
    // garbage tokens far from the cause — the failure mode that cost
    // a VOCAB-SIZED slot on gemma-class logits.
    //
    // A refusal, not a panic: this runs at macro expansion, and the
    // arch that trips it should name itself in the build log rather
    // than abort the whole compile.
    for (i, od) in ops.iter().enumerate() {
        let Some(k) = scratchy_target_metal::op_abi::metal_alias_operand(&od.op) else {
            continue;
        };
        let (Some(InputRef::Op(src)), Ok(mine)) = (
            od.inputs.get(k as usize).copied(),
            slot_of(lowered, slots, i),
        ) else {
            continue;
        };
        if let Ok(theirs) = slot_of(lowered, slots, src)
            && mine != theirs
        {
            return Err(format!(
                "op {i} ({:?}) declares it writes over operand {k}, but the \
                 coloring gave it slot {mine} and that operand slot {theirs} \
                 — the kernel would clobber a buffer another op owns",
                od.op
            ));
        }
    }
    let mut backbone: Vec<Instruction> = Vec::new();
    let mut bb_sigs: Vec<HazardSig> = Vec::new();
    let mut bb_ws: Vec<Vec<WeightSlot>> = Vec::new();
    // Emission GROUP = the op's SUBGRAPH (wave rank): instructions from
    // one fan_out share a group and serialize after their head — the
    // real rule, replacing the synthetic per-instruction counter (whose
    // manual regrouping missed multi-claim impls like gemma3's paired
    // qk-norms).
    let group_of = |i: usize| -> usize {
        lowered.op_tiles[i]
            .and_then(|(t, _)| facts.wave_rank.get(&t).copied())
            .unwrap_or(usize::MAX - i)
    };
    let _ = &group_of;
    let mut group = 0usize;
    // Head: the embedded-hidden source becomes the embed instructions.
    let embed_slot = {
        // The first op reading the EmbeddedHidden source tells us its
        // colored slot — by construction the embed writes the slot the
        // first RmsNorm reads.
        let first_norm = ops
            .iter()
            .position(|od| matches!(od.op, LoweredOp::RmsNorm { .. }))
            .ok_or("no first RmsNorm")?;
        let _ = first_norm;
        0u32
    };
    let has_embedded_hidden = lowered
        .bindings
        .iter()
        .any(|b| matches!(b, SourceBinding::EmbeddedHidden));
    let embed_ident = facts.embed_base.to_string();
    let mut head_scalar_mul: Option<usize> = None;
    if has_embedded_hidden {
        match facts.embed_quant {
            Some((gs, bits)) => {
                backbone.push(I::AffineEmbed(embed_slot, gs, bits));
                bb_ws.push(ws(WeightKind::AffineQuantEmbedding, embed_ident));
            }
            None => {
                backbone.push(I::Embed(embed_slot));
                bb_ws.push(ws(WeightKind::Embedding, embed_ident));
            }
        }
        bb_sigs.push(HazardSig {
            group,
            reads: vec![],
            writes: vec![embed_slot],
            kv_w: None,
            kv_r: None,
            metadata: false,
        });
        group += 1;
        // The embed-multiplier ScalarMul (granite) sits BETWEEN the embed
        // and the splice on the instruction-selection stream.
        if let Some((i0, od0)) = ops
            .iter()
            .enumerate()
            .find(|(_, od)| !matches!(od.op, LoweredOp::RmsNorm { .. }))
            && let LoweredOp::ScalarMul { scale } = od0.op
            && matches!(od0.inputs.first(), Some(InputRef::Ext(_)))
        {
            if scale == 1.0 {
                // Identity multiply — instruction selection elides it.
                head_scalar_mul = Some(i0);
            } else {
                let out = slot_of(lowered, slots, i0)?;
                backbone.push(I::ScalarMul(embed_slot, out, scale));
                bb_ws.push(Vec::new());
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![embed_slot],
                    writes: vec![out],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                });
                group += 1;
                head_scalar_mul = Some(i0);
            }
        }
        backbone.push(I::SpliceMmEmbeds(embed_slot));
        bb_ws.push(Vec::new());
        bb_sigs.push(HazardSig {
            group,
            reads: vec![embed_slot],
            writes: vec![embed_slot],
            kv_w: None,
            kv_r: None,
            metadata: false,
        });
    }

    let mut lm_head: Vec<Instruction> = Vec::new();
    let mut lm_sigs: Vec<HazardSig> = Vec::new();
    let mut lm_ws: Vec<Vec<WeightSlot>> = Vec::new();

    // Emission order = the scheduler's wave order (rank per tile),
    // tape index as the within-subgraph tiebreak.
    // ⭐ THE WALK IS THE TAPE'S. Not a re-derived order over the op list — the emission
    // sequence, and the LAYER LOOP inside it, are read straight off the shared `SubtileTape`
    // items. Feed it the ROLLED tape and the emitted stream is rolled by construction: there is
    // no repeating run left for anything downstream to search for, which is what removes the
    // second re-roll rather than unifying it.
    //
    // Tape groups sit ABOVE the head rows' groups so the embed write
    // fences against the first level's readers.
    let head_base = group + 1;
    // Per-op instruction span, recorded as the walk emits. See
    // `BridgedStream::backbone_span_of_op`.
    let mut span_of_op: Vec<Option<(usize, usize)>> = vec![None; ops.len()];
    // Flatten the walk into (source op, group) plus the loop boundaries.
    use scratchy_target_metal::from_tape::TapeItem;
    enum Emit {
        Op { i: usize, group: usize },
        OpenLoop { iters: u32, stride: u32 },
        CloseLoop,
    }
    let plan: Vec<Emit> = {
        let items = tape_items;
        let mut out = Vec::with_capacity(items.len());
        let mut rank_group: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut next_group = 0usize;
        for it in items.iter() {
            match it {
                TapeItem::Step(st) => {
                    // ⛔ ORDER AND GROUPING ARE DIFFERENT FACTS. The tape gives the ORDER;
                    // the wave rank gives which ops may dispatch TOGETHER. Using the step
                    // position as the group gave every op its own group, so a barrier
                    // fenced between every pair — correct output, 12% slower decode.
                    // Groups are renumbered by first appearance so they stay monotonic
                    // along the tape walk while ops of one wave still share theirs.
                    let rank = lowered.op_tiles[st.source_op.0]
                        .and_then(|(t, _)| facts.wave_rank.get(&t).copied());
                    let g = match rank {
                        Some(r) => *rank_group.entry(r).or_insert_with(|| {
                            let g = next_group;
                            next_group += 1;
                            g
                        }),
                        None => {
                            let g = next_group;
                            next_group += 1;
                            g
                        }
                    };
                    out.push(Emit::Op {
                        i: st.source_op.0,
                        group: head_base + g,
                    });
                }
                TapeItem::OpenLoop { iters, stride, .. } => out.push(Emit::OpenLoop {
                    iters: *iters,
                    stride: *stride,
                }),
                TapeItem::CloseLoop { .. } => out.push(Emit::CloseLoop),
            }
        }
        out
    };
    // Backpatch position of the `Loop` marker: its body length is only known at CloseLoop.
    // ⛔ A STACK, NOT AN `Option`. gemma-4 nests: an outer loop over the six-layer `SSSSSG` cell
    // and an inner one over the five sliding layers inside it. One slot here would silently drop
    // the outer loop's backpatch when the inner one closed.
    let mut open_loop: Vec<(usize, u32, u32)> = Vec::new();
    for step in &plan {
        let (i, this_group) = match step {
            Emit::OpenLoop { iters, stride } => {
                open_loop.push((backbone.len(), *iters, *stride));
                continue;
            }
            Emit::CloseLoop => {
                let (at, iters, stride) = open_loop.pop().ok_or("CloseLoop without OpenLoop")?;
                let period = backbone.len() - at;
                if period > 0 {
                    backbone.insert(
                        at,
                        scratchy_ir::Instruction::Loop(iters, period as u32, stride),
                    );
                    bb_ws.insert(at, Vec::new());
                    // The marker dispatches nothing, so it reads and writes nothing. Its
                    // group is the body's first row's, so it fences where the body does.
                    let g = bb_sigs.get(at).map(|s: &HazardSig| s.group).unwrap_or(0);
                    bb_sigs.insert(
                        at,
                        HazardSig {
                            group: g,
                            reads: vec![],
                            writes: vec![],
                            kv_w: None,
                            kv_r: None,
                            metadata: true,
                        },
                    );
                }
                continue;
            }
            Emit::Op { i, group } => (*i, *group),
        };
        let od = &ops[i];
        if fused_into[i].is_some() || head_scalar_mul == Some(i) {
            continue;
        }
        let span_start = backbone.len();
        group = this_group;
        let out = slot_of(lowered, slots, i);
        // THE unary-elementwise shape, written once: read operand 0's
        // slot (an `Ext` operand is the embed slot), write the op's
        // own, bind no weights, and declare the plain RAW/WAW
        // signature. Three ops spelled these sixteen lines out
        // identically; a fourth would have spelled them a fourth
        // time. A macro rather than a closure because the bodies push
        // to three of the loop's own buffers.
        // Resolve operand `k`'s slot (an `Ext` operand is the embed
        // slot) — the same two lines eleven arms opened with.
        macro_rules! in_slot {
            ($k:expr) => {
                match &od.inputs[$k] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                }
            };
        }
        // THE plain-op shape: bind no weights, read the listed slots,
        // write the op's own. Written ONCE. Every arm that spelled the
        // `HazardSig` out by hand was restating the same fact about
        // metal's hazard model, and a wrong copy is invisible until a
        // race shows up as garbage tokens.
        macro_rules! push_plain {
            ($instr:expr, reads: [$($r:expr),* $(,)?], writes: [$($w:expr),* $(,)?]) => {{
                backbone.push($instr);
                bb_ws.push(Vec::new());
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![$($r),*],
                    writes: vec![$($w),*],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                });
            }};
        }
        match &od.op {
            // ONE arm for the whole elementwise-unary class; the
            // members and their constructors are declared together in
            // `metal_op_abi`. Adding an activation touches that file
            // only.
            // Zero-input source ops, declared as a class beside the
            // unary one.
            // One-in/one-out ops that read their OWN fields: the row
            // builds the instruction, the bridge supplies the slots.
            scratchy_target_metal::metal_in_out_pat!() => {
                let out = out?;
                let (n_in, metadata) = scratchy_target_metal::op_abi::metal_in_out_shape(&od.op)
                    .expect("metal_in_out_pat member without a shape row");
                let mut ins: Vec<u32> = Vec::with_capacity(n_in as usize);
                for k in 0..n_in as usize {
                    ins.push(in_slot!(k));
                }
                let instr = scratchy_target_metal::op_abi::metal_in_out_build(&od.op, &ins, out)
                    .expect("metal_in_out_pat member without a build row");
                backbone.push(instr);
                bb_ws.push(Vec::new());
                bb_sigs.push(HazardSig {
                    group,
                    reads: ins,
                    writes: vec![out],
                    kv_w: None,
                    kv_r: None,
                    metadata,
                });
            }
            scratchy_target_metal::metal_source_pat!() => {
                let ctor = scratchy_target_metal::op_abi::metal_source_ctor(&od.op)
                    .expect("metal_source_pat member without a ctor row");
                let out = out?;
                push_plain!(ctor(out), reads: [], writes: [out]);
            }
            scratchy_target_metal::metal_unary_pat!() => {
                let ctor = scratchy_target_metal::op_abi::metal_unary_ctor(&od.op)
                    .expect("metal_unary_pat member without a ctor row");
                let out = out?;
                let in_slot = in_slot!(0);
                push_plain!(ctor(in_slot, out), reads: [in_slot], writes: [out]);
            }
            LoweredOp::RmsNorm { .. } => {
                let out = out?;
                let a = &od.inputs[0];
                match a {
                    InputRef::Ext(_) => {
                        // First norm: reads the embedded hidden.
                        let layer = weight_ext(lowered, i)
                            .map(|e| layer_of(lowered, e))
                            .unwrap_or(0);
                        if let LoweredOp::RmsNorm { gain_offset, .. } = &od.op
                            && *gain_offset != 0.0
                        {
                            backbone.push(I::ScalarOffsetRmsNorm(
                                embed_slot,
                                out,
                                layer,
                                *gain_offset,
                                facts.hidden_size,
                                1,
                            ));
                        } else {
                            backbone.push(I::RmsNorm(embed_slot, out, layer, facts.hidden_size, 1));
                        }
                        bb_ws.push(ws(
                            WeightKind::RmsNorm,
                            base_of(
                                lowered,
                                facts,
                                weight_ext(lowered, i).ok_or("norm: no weight")?,
                            )?,
                        ));
                        bb_sigs.push(HazardSig {
                            group,
                            reads: vec![embed_slot],
                            writes: vec![out],
                            kv_w: None,
                            kv_r: None,
                            metadata: false,
                        });
                    }
                    InputRef::Op(a) => {
                        if matches!(ops[*a].op, LoweredOp::Sub)
                            && fused_into[*a] == Some(i)
                            && let Some(bi) = msn_bias.get(&i).copied()
                        {
                            let in_slot = match &ops[*a].inputs[0] {
                                InputRef::Op(p2) => slot_of(lowered, slots, *p2)?,
                                InputRef::Ext(_) => embed_slot,
                            };
                            let out_b = slot_of(lowered, slots, bi)?;
                            let layer = weight_ext(lowered, i)
                                .map(|e| layer_of(lowered, e))
                                .unwrap_or(0);
                            fused_into[bi] = Some(i);
                            backbone.push(I::MeanSubRmsNormBiasAdd(in_slot, out_b, layer));
                            // Bias-carrying LayerNorm accessor (vision);
                            // the bias rides the LayerNorm bundle.
                            bb_ws.push(ws(
                                WeightKind::LayerNorm,
                                base_of(
                                    lowered,
                                    facts,
                                    weight_ext(lowered, i).ok_or("msnb: no weight")?,
                                )?,
                            ));
                            bb_sigs.push(HazardSig {
                                group,
                                reads: vec![in_slot],
                                writes: vec![out_b],
                                kv_w: None,
                                kv_r: None,
                                metadata: false,
                            });
                        } else if matches!(ops[*a].op, LoweredOp::Sub) && fused_into[*a] == Some(i)
                        {
                            // MeanSubRmsNorm(in, out, layer): in = the
                            // Sub's x operand (the pre-centering value;
                            // the embedded hidden at the first layer).
                            let in_slot = match &ops[*a].inputs[0] {
                                InputRef::Op(p2) => slot_of(lowered, slots, *p2)?,
                                InputRef::Ext(_) => embed_slot,
                            };
                            let layer = weight_ext(lowered, i)
                                .map(|e| layer_of(lowered, e))
                                .unwrap_or(0);
                            backbone.push(I::MeanSubRmsNorm(in_slot, out, layer));
                            // The accessor kind is RmsNorm — the gain
                            // resolves through rms_norm_at like every
                            // other norm (the LayerNorm KIND is the
                            // bias-carrying cuda variant).
                            bb_ws.push(ws(
                                WeightKind::RmsNorm,
                                base_of(
                                    lowered,
                                    facts,
                                    weight_ext(lowered, i).ok_or("meansub: no weight")?,
                                )?,
                            ));
                            bb_sigs.push(HazardSig {
                                group,
                                reads: vec![in_slot],
                                writes: vec![out],
                                kv_w: None,
                                kv_r: None,
                                metadata: false,
                            });
                        } else if let LoweredOp::Reshape {
                            rows_mult,
                            rows_div: 1,
                            cols,
                        } = ops[*a].op
                        {
                            // Per-head norm (qwen3 qk-norm): rows are
                            // heads x tokens; gain is [cols]. Gemma3's
                            // per-head gains carry the (1+w) offset.
                            let in_slot = slot_of(lowered, slots, *a)?;
                            let layer = weight_ext(lowered, i)
                                .map(|e| layer_of(lowered, e))
                                .unwrap_or(0);
                            if let LoweredOp::RmsNorm { gain_offset, .. } = &od.op
                                && *gain_offset != 0.0
                            {
                                backbone.push(I::ScalarOffsetRmsNorm(
                                    in_slot,
                                    out,
                                    layer,
                                    *gain_offset,
                                    cols,
                                    rows_mult,
                                ));
                            } else {
                                backbone.push(I::RmsNorm(in_slot, out, layer, cols, rows_mult));
                            }
                            bb_ws.push(ws(
                                WeightKind::RmsNorm,
                                base_of(
                                    lowered,
                                    facts,
                                    weight_ext(lowered, i).ok_or("qk-norm: no weight")?,
                                )?,
                            ));
                            // The instruction-selection impls CLAIM the
                            // folded (w + scalar) Add tile — its colored
                            // slot rides their hazard writes (and is
                            // REUSED across the q/k gain-adds, carrying
                            // the WAW that serializes the second norm).
                            let mut writes = vec![out];
                            if let Some(gslot) = gain_add_slot(lowered, slots, i) {
                                writes.insert(0, gslot);
                            }
                            bb_sigs.push(HazardSig {
                                group,
                                reads: vec![in_slot],
                                writes,
                                kv_w: None,
                                kv_r: None,
                                metadata: false,
                            });
                        } else if matches!(ops[*a].op, LoweredOp::Add) && fused_into[*a] == Some(i)
                        {
                            // FusedAddRmsNorm(x_slot, residual_slot, layer,
                            // hidden, mult): the residual accumulates in
                            // place at arg1; the normed value lands over
                            // arg0. arg1 = the Add's OTHER input's slot.
                            let add = &ops[*a];
                            let x = in_op(&add.inputs[0]).ok_or("add in0 not op")?;
                            let x_slot = slot_of(lowered, slots, x)?;
                            let res_slot = match &add.inputs[1] {
                                InputRef::Op(r) => slot_of(lowered, slots, *r)?,
                                InputRef::Ext(_) => embed_slot,
                            };
                            let layer = weight_ext(lowered, i)
                                .map(|e| layer_of(lowered, e))
                                .unwrap_or(0);
                            let _ = out;
                            if let LoweredOp::RmsNorm { gain_offset, .. } = &od.op
                                && *gain_offset != 0.0
                            {
                                backbone.push(I::FusedAddRmsNormWithOffset(
                                    x_slot,
                                    res_slot,
                                    layer,
                                    *gain_offset,
                                ));
                            } else {
                                backbone.push(I::FusedAddRmsNorm(
                                    x_slot,
                                    res_slot,
                                    layer,
                                    facts.hidden_size,
                                    1,
                                ));
                            }
                            bb_ws.push(ws(
                                WeightKind::RmsNorm,
                                base_of(
                                    lowered,
                                    facts,
                                    weight_ext(lowered, i).ok_or("farn: no weight")?,
                                )?,
                            ));
                            let mut writes = vec![res_slot, out];
                            if let Some(gslot) = gain_add_slot(lowered, slots, i) {
                                writes.push(gslot);
                            }
                            bb_sigs.push(HazardSig {
                                group,
                                reads: vec![x_slot, res_slot],
                                writes,
                                kv_w: None,
                                kv_r: None,
                                metadata: false,
                            });
                        } else {
                            let in_slot = slot_of(lowered, slots, *a)?;
                            let layer = weight_ext(lowered, i)
                                .map(|e| layer_of(lowered, e))
                                .unwrap_or(0);
                            // The operand's TRUE width — a per-head
                            // qk-norm reads a [m*heads, head_dim] view
                            // (gemma3-1b: 256, not hidden 1152).
                            let w = op_width[i];
                            if let LoweredOp::RmsNorm { gain_offset, .. } = &od.op
                                && *gain_offset != 0.0
                            {
                                backbone.push(I::ScalarOffsetRmsNorm(
                                    in_slot,
                                    out,
                                    layer,
                                    *gain_offset,
                                    w,
                                    1,
                                ));
                            } else {
                                backbone.push(I::RmsNorm(in_slot, out, layer, w, 1));
                            }
                            bb_ws.push(ws(
                                WeightKind::RmsNorm,
                                base_of(
                                    lowered,
                                    facts,
                                    weight_ext(lowered, i).ok_or("norm: no weight")?,
                                )?,
                            ));
                            let mut writes = vec![out];
                            if let Some(gslot) = gain_add_slot(lowered, slots, i) {
                                writes.insert(0, gslot);
                            }
                            bb_sigs.push(HazardSig {
                                group,
                                reads: vec![in_slot],
                                writes,
                                kv_w: None,
                                kv_r: None,
                                metadata: false,
                            });
                        }
                    }
                }
            }
            LoweredOp::Gemm { n: gn, weight } => {
                let out = out?;
                let in_slot = match &od.inputs[0] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let e = weight_ext(lowered, i).ok_or_else(|| format!("gemm {i}: no weight"))?;
                let layer = layer_of_with_path(lowered, facts, e);
                let k = lowered.input.sources[e].rows;
                let is_result = i == lowered.input.result;
                let inst = match weight {
                    GemmWeight::Dense => I::Gemm(in_slot, out, layer, *gn, k),
                    GemmWeight::Fp8Dynamic => {
                        return Err(format!("gemm {i}: fp8 not yet bridged"));
                    }
                    GemmWeight::Affine { affine } => I::AffineQmm(
                        in_slot,
                        out,
                        layer,
                        *gn,
                        k,
                        affine.group().get(),
                        affine.bits().get(),
                        affine_qmm_vector_limit(k, *gn),
                    ),
                };
                let sig = HazardSig {
                    group,
                    reads: vec![in_slot],
                    writes: vec![out],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                };
                let slot = ws(WeightKind::Linear, base_of(lowered, facts, e)?);
                if is_result {
                    lm_head.push(inst);
                    lm_sigs.push(sig);
                    lm_ws.push(slot);
                } else {
                    backbone.push(inst);
                    bb_sigs.push(sig);
                    bb_ws.push(slot);
                }
            }
            LoweredOp::RopeAppend {
                layer,
                is_global,
                interleaved,
                ..
            } if rope_normed.contains_key(&i) => {
                let info = &rope_normed[&i];
                let q_slot = slot_of(lowered, slots, info.q_raw)?;
                let k_slot = slot_of(lowered, slots, info.k_raw)?;
                let v_slot = slot_of(lowered, slots, info.v_raw)?;
                backbone.push(I::RopeAppendNormed(
                    q_slot,
                    k_slot,
                    v_slot,
                    q_slot,
                    k_slot,
                    v_slot,
                    *layer,
                    *interleaved,
                    *is_global,
                ));
                let mut wsv = Vec::new();
                if let Some(g) = info.q_gain {
                    wsv.push(WeightSlot {
                        kind: WeightKind::RmsNorm,
                        base: base_of(lowered, facts, g)?,
                    });
                }
                if let Some(g) = info.k_gain {
                    wsv.push(WeightSlot {
                        kind: WeightKind::RmsNorm,
                        base: base_of(lowered, facts, g)?,
                    });
                }
                wsv.push(WeightSlot {
                    kind: WeightKind::CosSin,
                    base: rope_rotary_ident(lowered, od),
                });
                bb_ws.push(wsv);
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![q_slot, k_slot, v_slot],
                    writes: vec![q_slot, k_slot, v_slot],
                    kv_w: Some(*layer),
                    kv_r: None,
                    metadata: false,
                });
            }
            LoweredOp::RopeAppend {
                layer,
                is_global,
                interleaved,
                ..
            } => {
                // The folded rotate gives the q slot; this op's k/v.
                let rot = fused_into
                    .iter()
                    .position(|f| *f == Some(i))
                    .ok_or_else(|| format!("rope {i}: no folded rotate"))?;
                let q = in_op(&ops[rot].inputs[0]).ok_or("rotate in0 not op")?;
                let q_slot = slot_of(lowered, slots, q)?;
                let k = in_op(&od.inputs[0]).ok_or("rope in0 (k) not op")?;
                let v = in_op(&od.inputs[3]).ok_or("rope in3 (v) not op")?;
                let k_slot = slot_of(lowered, slots, k)?;
                let v_slot = slot_of(lowered, slots, v)?;
                let (k_off, k_linear) = kv_offset_of(lowered, facts, k, *layer)?;
                let (v_off, v_linear) = kv_offset_of(lowered, facts, v, *layer)?;
                let kv_offsets = scratchy_ir::KvOffsets { k: k_off, v: v_off };
                backbone.push(I::RopeAppend(
                    q_slot,
                    k_slot,
                    v_slot,
                    q_slot,
                    k_slot,
                    v_slot,
                    *layer,
                    *interleaved,
                    *is_global,
                    kv_offsets,
                ));
                bb_ws.push(scratchy_target_metal::op_abi::rope_append_weight_site(
                    WeightSlot {
                        kind: WeightKind::CosSin,
                        base: rope_rotary_ident(lowered, od),
                    },
                    k_linear,
                    v_linear,
                ));
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![q_slot, k_slot, v_slot],
                    writes: vec![q_slot, k_slot, v_slot],
                    kv_w: Some(*layer),
                    kv_r: None,
                    metadata: false,
                });
            }
            LoweredOp::AttnDecode { .. } => {
                let out = out?;
                // in = the roped q slot (via the folded rotate op).
                let q = in_op(&od.inputs[0]).ok_or("attn in0 not op")?;
                let q_slot = slot_of(lowered, slots, q)?;
                // layer from the rope append it consumes.
                let rope = in_op(&od.inputs[3]).ok_or("attn in3 not rope op")?;
                let layer = match &ops[rope].op {
                    LoweredOp::RopeAppend { layer, .. } => *layer,
                    _ => 0,
                };
                let sliding = matches!(od.op, LoweredOp::AttnDecode { sliding: true, .. });
                let rope_interleaved = in_op(&od.inputs[3])
                    .map(|r| {
                        matches!(
                            ops[r].op,
                            LoweredOp::RopeAppend {
                                interleaved: true,
                                ..
                            }
                        )
                    })
                    .unwrap_or(false);
                if od.m > 1 {
                    // Prefill shape: the paged prefill kernel, no
                    // rotary consumption (K arrives pre-roped).
                    if sliding {
                        backbone.push(I::SlidingAttentionPrefillPaged(q_slot, out, layer, false));
                    } else {
                        backbone.push(I::AttentionPrefillPaged(q_slot, out, layer, false));
                    }
                    bb_ws.push(Vec::new());
                } else {
                    // The attention impl's CosSin accessor is the GLOBAL
                    // rotary regardless of class (only the ROPE row uses
                    // the local one) — matches instruction selection.
                    if sliding {
                        backbone.push(I::SlidingAttentionViaCache(
                            q_slot,
                            out,
                            layer,
                            rope_interleaved,
                        ));
                    } else {
                        backbone.push(I::AttentionViaCache(q_slot, out, layer, rope_interleaved));
                    }
                    bb_ws.push(ws(WeightKind::CosSin, "rotary".to_string()));
                }
                let k_read = in_op(&od.inputs[3]).and_then(|r| slot_of(lowered, slots, r).ok());
                let v_read = in_op(&od.inputs[4]).and_then(|r| slot_of(lowered, slots, r).ok());
                let mut reads = vec![q_slot];
                reads.extend(k_read);
                reads.extend(v_read);
                bb_sigs.push(HazardSig {
                    group,
                    reads,
                    writes: vec![out],
                    kv_w: None,
                    kv_r: Some(layer),
                    metadata: false,
                });
            }
            LoweredOp::Mul => {
                let si = in_op(&od.inputs[0]).ok_or("mul in0 not op")?;
                let gate = in_op(&ops[si].inputs[0]).ok_or("silu in not op")?;
                let up = in_op(&od.inputs[1]).ok_or("mul in1 not op")?;
                let out = out?;
                if facts.fused_mlp {
                    // FusedGateUpSiluMul(in_slot, out_slot, layer) — the
                    // gate gemm's input and weight layer carry it.
                    let gin = match &ops[gate].inputs[0] {
                        InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                        InputRef::Ext(_) => embed_slot,
                    };
                    let layer = weight_ext(lowered, gate)
                        .map(|e| layer_of(lowered, e))
                        .unwrap_or(0);
                    if matches!(ops[si].op, LoweredOp::Gelu) {
                        backbone.push(I::FusedGateUpGeluMul(gin, out, layer));
                    } else {
                        backbone.push(I::FusedGateUpSiluMul(gin, out, layer));
                    }
                    let g_base = base_of(
                        lowered,
                        facts,
                        weight_ext(lowered, gate).ok_or("gate: no weight")?,
                    )?;
                    let u_base = base_of(
                        lowered,
                        facts,
                        weight_ext(lowered, up).ok_or("up: no weight")?,
                    )?;
                    bb_ws.push(ws(WeightKind::Linear, format!("{g_base}__fused__{u_base}")));
                    let gate_slot = slot_of(lowered, slots, gate)?;
                    let up_slot = slot_of(lowered, slots, up)?;
                    bb_sigs.push(HazardSig {
                        group,
                        reads: vec![gin],
                        writes: vec![gate_slot, up_slot, out],
                        kv_w: None,
                        kv_r: None,
                        metadata: false,
                    });
                } else {
                    // Split quant shape: gate qmm, up qmm, silumul are
                    // ONE emission group (one fan_out on the
                    // instruction-selection side) — but the gemms were
                    // already emitted as separate groups by the Gemm
                    // arm. Regroup: the two preceding backbone entries
                    // (gate, up) join this group.
                    let gate_slot = slot_of(lowered, slots, gate)?;
                    let up_slot = slot_of(lowered, slots, up)?;
                    if matches!(ops[si].op, LoweredOp::Gelu) {
                        backbone.push(I::GeluMul(gate_slot, up_slot, out));
                    } else {
                        backbone.push(I::SiluMul(gate_slot, up_slot, out, facts.intermediate_size));
                    }
                    bb_ws.push(Vec::new());
                    bb_sigs.push(HazardSig {
                        group,
                        reads: vec![gate_slot, up_slot],
                        writes: vec![out],
                        kv_w: None,
                        kv_r: None,
                        metadata: false,
                    });
                }
            }
            LoweredOp::SiluMul => {
                let gate = in_op(&od.inputs[0]).ok_or("silumul in0")?;
                let up = in_op(&od.inputs[1]).ok_or("silumul in1")?;
                let out = out?;
                let gs_ = slot_of(lowered, slots, gate)?;
                let us_ = slot_of(lowered, slots, up)?;
                push_plain!(
                    I::SiluMul(gs_, us_, out, facts.intermediate_size),
                    reads: [gs_, us_],
                    writes: [out]);
            }
            LoweredOp::ScalarMul { scale } if *scale == 1.0 => {
                // Identity — instruction selection elides the row; the
                // op's slot aliases its input (in-place by contract).
                continue;
            }
            LoweredOp::ScalarMul { scale } => {
                let out = out?;
                let in_slot = match &od.inputs[0] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                };
                push_plain!(I::ScalarMul(in_slot, out, *scale), reads: [in_slot], writes: [out]);
            }
            LoweredOp::BiasAdd => {
                // MetalBiasAdd(in_slot, out_slot, layer, n, is_affine).
                // The accessor shares the UPSTREAM gemm's LinearLayer
                // (the bias rides on it), so base name, layer, and the
                // affine flag all come from that gemm.
                let out = out?;
                let up = in_op(&od.inputs[0]).ok_or("bias in0 not op")?;
                let in_slot = slot_of(lowered, slots, up)?;
                let (layer, base, is_affine, n) = match &ops[up].op {
                    LoweredOp::Gemm { n, weight } => {
                        let e = weight_ext(lowered, up).ok_or("bias upstream: no weight")?;
                        (
                            layer_of_with_path(lowered, facts, e),
                            base_of(lowered, facts, e)?,
                            matches!(weight, GemmWeight::Affine { .. }),
                            *n,
                        )
                    }
                    other => return Err(format!("bias upstream is {other:?}, not a gemm")),
                };
                backbone.push(I::MetalBiasAdd(in_slot, out, layer, n, is_affine));
                bb_ws.push(ws(WeightKind::Linear, base));
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![in_slot],
                    writes: vec![out],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                });
            }
            LoweredOp::Add => {
                // Unfused residual add (gemma4-moe layer tail). The
                // instruction is in-place: `Add(a, out)` accumulates a
                // into the out slot (which aliases the other operand).
                let out = out?;
                let a = match &od.inputs[0] {
                    InputRef::Op(x) => slot_of(lowered, slots, *x)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let b = match &od.inputs[1] {
                    InputRef::Op(x) => slot_of(lowered, slots, *x)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let other = if out == b { a } else { b };
                push_plain!(I::Add(other, out), reads: [a, b], writes: [out]);
            }
            LoweredOp::ScalarWeightMul if nasm.contains_key(&i) => {
                let info = &nasm[&i];
                let out = out?;
                let delta_slot = slot_of(lowered, slots, info.delta)?;
                let res_slot = match info.residual {
                    InputRef::Op(r) => slot_of(lowered, slots, r)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let layer = layer_of(lowered, info.scalar_w);
                backbone.push(I::NormAddScalarMul(
                    delta_slot,
                    res_slot,
                    out,
                    layer,
                    facts.hidden_size,
                ));
                bb_ws.push(vec![
                    WeightSlot {
                        kind: WeightKind::RmsNorm,
                        base: base_of(lowered, facts, info.norm_gain)?,
                    },
                    WeightSlot {
                        kind: WeightKind::RmsNorm,
                        base: base_of(lowered, facts, info.scalar_w)?,
                    },
                ]);
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![delta_slot, res_slot],
                    writes: vec![res_slot, out],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                });
            }
            LoweredOp::ScalarWeightMul => {
                // Standalone (unfused) — the loaded-[1]-weight multiply.
                let out = out?;
                let in_slot = match &od.inputs[0] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let e = weight_ext(lowered, i).ok_or("scalar_weight_mul: no weight")?;
                backbone.push(I::ScalarWeightMul(in_slot, out, layer_of(lowered, e)));
                bb_ws.push(ws(WeightKind::RmsNorm, base_of(lowered, facts, e)?));
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![in_slot],
                    writes: vec![out],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                });
            }
            LoweredOp::GateSplit { .. } => {
                // GateSplit(qg, q_out, gate_out): q = this op's colored
                // slot 0; gate = its slot-1 provenance entry, resolved
                // through the SAME coloring authority.
                let out = out?;
                let in_slot = match &od.inputs[0] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let gate_slot = lowered.op_tiles[i]
                    .map(|(t, _)| slots.of(scratchy_subtile::handoff::TileId(t), 1))
                    .ok_or("gate_split: no provenance")?;
                push_plain!(I::GateSplit(in_slot, out, gate_slot), reads: [in_slot], writes: [out, gate_slot]);
            }
            LoweredOp::GateApply => {
                let out = out?;
                let attn = match &od.inputs[0] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let gate = match &od.inputs[1] {
                    InputRef::Op(a) => {
                        // The gate came from a GateSplit's SECOND output.
                        lowered.op_tiles[*a]
                            .map(|(t, _)| slots.of(scratchy_subtile::handoff::TileId(t), 1))
                            .ok_or("gate_apply: gate provenance")?
                    }
                    InputRef::Ext(_) => embed_slot,
                };
                push_plain!(I::GateApply(attn, gate, out), reads: [attn, gate], writes: [out]);
            }
            LoweredOp::VisionRope => {
                let sl = |r: &InputRef| -> Result<u32, String> {
                    match r {
                        InputRef::Op(a) => slot_of(lowered, slots, *a),
                        InputRef::Ext(_) => Ok(embed_slot),
                    }
                };
                let q = sl(&od.inputs[0])?;
                let k = sl(&od.inputs[1])?;
                let q_out = out?;
                let k_out = second_out_slot(lowered, slots, i)?;
                push_plain!(I::VisionRope(q, k, q_out, k_out), reads: [q, k], writes: [q_out, k_out]);
            }
            LoweredOp::VarlenAttention { cu_kind } => {
                let out = out?;
                let sl = |r: &InputRef| -> Result<u32, String> {
                    match r {
                        InputRef::Op(a) => slot_of(lowered, slots, *a),
                        InputRef::Ext(_) => Ok(embed_slot),
                    }
                };
                let q = sl(&od.inputs[0])?;
                let k = match &od.inputs[1] {
                    InputRef::Op(a) if matches!(ops[*a].op, LoweredOp::VisionRope) => {
                        second_out_slot(lowered, slots, *a)?
                    }
                    r => sl(r)?,
                };
                let v = sl(&od.inputs[2])?;
                push_plain!(I::VarlenAttention(q, k, v, out, *cu_kind), reads: [q, k, v], writes: [out]);
            }
            LoweredOp::EncoderAttn { .. } => {
                let out = out?;
                let sl = |r: &InputRef| -> Result<u32, String> {
                    match r {
                        InputRef::Op(a) => slot_of(lowered, slots, *a),
                        InputRef::Ext(_) => Ok(embed_slot),
                    }
                };
                let q = sl(&od.inputs[0])?;
                let k = sl(&od.inputs[1])?;
                let v = sl(&od.inputs[2])?;
                push_plain!(I::EncoderAttention(q, k, v, out), reads: [q, k, v], writes: [out]);
            }
            LoweredOp::GatedDeltaNet => {
                let out = out?;
                let slot_of_in = |r: &InputRef| -> Result<u32, String> {
                    match r {
                        InputRef::Op(a) => slot_of(lowered, slots, *a),
                        InputRef::Ext(_) => Ok(embed_slot),
                    }
                };
                let qkv = slot_of_in(&od.inputs[0])?;
                let z = slot_of_in(&od.inputs[1])?;
                let a = slot_of_in(&od.inputs[2])?;
                let b2 = slot_of_in(&od.inputs[3])?;
                let e = weight_ext(lowered, i).ok_or("gdn: no weight")?;
                let layer = layer_of(lowered, e);
                backbone.push(I::GatedDeltaNet(qkv, z, a, b2, out, layer));
                bb_ws.push(ws(WeightKind::GatedDeltaNet, base_of(lowered, facts, e)?));
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![qkv, z, a, b2],
                    writes: vec![out],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                });
            }
            LoweredOp::GemmaMoe {
                num_experts,
                top_k,
                moe_inter,
                group_size,
                bits,
            } => {
                let out = out?;
                let router_in = match &od.inputs[0] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let expert_in = match &od.inputs[1] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let (rw, ew) = match (&od.inputs[2], &od.inputs[3]) {
                    (InputRef::Ext(r), InputRef::Ext(e)) => (*r, *e),
                    _ => return Err(format!("op {i}: gemma_moe bundle inputs not sources")),
                };
                let layer = layer_of(lowered, rw);
                backbone.push(I::GemmaMoe(
                    router_in,
                    expert_in,
                    out,
                    layer,
                    *num_experts,
                    *top_k,
                    *moe_inter,
                    facts.hidden_size,
                    *group_size,
                    *bits,
                ));
                bb_ws.push(vec![
                    WeightSlot {
                        kind: WeightKind::GemmaRouter,
                        base: base_of(lowered, facts, rw)?,
                    },
                    WeightSlot {
                        kind: WeightKind::GemmaSwitchGlu,
                        base: base_of(lowered, facts, ew)?,
                    },
                ]);
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![router_in, expert_in],
                    writes: vec![out],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                });
            }
            LoweredOp::Moe {
                qwen_shared,
                num_experts,
                top_k,
                moe_inter,
                shared_inter,
                norm_topk,
                group_size,
                bits,
            } => {
                let out = out?;
                let in_slot = match &od.inputs[0] {
                    InputRef::Op(a) => slot_of(lowered, slots, *a)?,
                    InputRef::Ext(_) => embed_slot,
                };
                let e = weight_ext(lowered, i).ok_or("moe: no weight")?;
                let layer = layer_of(lowered, e);
                let base = base_of(lowered, facts, e)?;
                // Single-owner rule (mirrors the instruction-selection
                // impl): a DSL-declared `<base>.shared_expert` subtree
                // owns those weights — the fused op's internal shared
                // tail is disabled.
                let dsl_owns_shared = facts.dsl_shared_expert_bases.iter().any(|b| b == &base);
                let eff_shared = if dsl_owns_shared { 0 } else { *shared_inter };
                if *qwen_shared {
                    backbone.push(I::MetalSharedFusedMoe(
                        in_slot,
                        out,
                        layer,
                        *num_experts,
                        *top_k,
                        *moe_inter,
                        facts.hidden_size,
                        eff_shared,
                        *group_size,
                        *bits,
                        *norm_topk,
                    ));
                    bb_ws.push(ws(WeightKind::SharedFusedMoe, base));
                } else {
                    backbone.push(I::MetalFusedMoe(
                        in_slot,
                        out,
                        layer,
                        *num_experts,
                        *top_k,
                        *moe_inter,
                        facts.hidden_size,
                        *group_size,
                        *bits,
                    ));
                    bb_ws.push(ws(WeightKind::FusedMoe, base));
                }
                bb_sigs.push(HazardSig {
                    group,
                    reads: vec![in_slot],
                    writes: vec![out],
                    kv_w: None,
                    kv_r: None,
                    metadata: false,
                });
            }
            LoweredOp::RmsNormUnit { .. } => {
                // Standalone unit norm (unfused rope path, e.g. the
                // k_eq_v global layer where multi-consumer chains
                // block the fusion). Width/mult from the input view.
                let out = out?;
                let a = in_op(&od.inputs[0]).ok_or("rmsnorm_unit in not op")?;
                let (hidden, mult) = match ops[a].op {
                    LoweredOp::Reshape {
                        rows_mult,
                        rows_div: 1,
                        cols,
                    } => (cols, rows_mult),
                    _ => (facts.hidden_size, 1),
                };
                let in_slot = slot_of(lowered, slots, a)?;
                push_plain!(I::RmsNormUnit(in_slot, out, hidden, mult), reads: [in_slot], writes: [out]);
            }
            // Unfused GELU (tanh approximation) — the vision MLPs
            // apply it standalone, unlike the text FFNs whose
            // gate/up/act collapse into FusedGateUpGeluMul.
            LoweredOp::Mean | LoweredOp::Sub => {
                return Err(format!(
                    "op {i}: {:?} escaped the MeanSubRmsNorm fold",
                    od.op
                ));
            }
            LoweredOp::Silu | LoweredOp::RopeRotate { .. } => {
                return Err(format!("op {i}: {:?} escaped its fold", od.op));
            }
        }
        // Close this op's instruction span (empty span if it emitted nothing).
        if backbone.len() > span_start {
            span_of_op[i] = Some((span_start, backbone.len()));
        }
    }
    debug_assert_eq!(backbone.len(), bb_ws.len());
    // ⛔ THE MoE BIT REPACK IS THE CALLER'S. It needs the weight `Program` and `ModelParams`,
    // which are the shared compiler's types — pulling them across this boundary would drag the
    // whole classified-weights layer into the target. It is a pure post-pass over the finished
    // stream, so the caller applies it to `backbone`/`lm_head` on the way out.
    Ok(BridgedStream {
        backbone,
        lm_head,
        backbone_sigs: bb_sigs,
        lm_head_sigs: lm_sigs,
        backbone_weight_slots: bb_ws,
        lm_head_weight_slots: lm_ws,
        backbone_span_of_op: span_of_op,
    })
}

/// Per-emitted-instruction hazard signature: which arena slots the
/// instruction (and every shared op folded into it) reads/writes, its
/// KV-layer IO, and its emission GROUP (instructions emitted by one
/// logical unit — e.g. the split quant-MLP triple — serialize after
/// their group head exactly as the instruction-selection fan_out did).
#[derive(Clone)]
pub struct HazardSig {
    pub group: usize,
    pub reads: Vec<u32>,
    pub writes: Vec<u32>,
    pub kv_w: Option<u32>,
    pub kv_r: Option<u32>,
    /// Metadata rows (views) dispatch nothing: flag stays false and
    /// they are excluded from hazard bookkeeping — same rule as the
    /// instruction-selection walk (a metadata row must never flush
    /// pending sets, see the Gemma4 lost-barrier note there).
    pub metadata: bool,
}

/// The MTL4 barrier flags for a bridge stream — same rules as the
/// instruction-selection hazard walk: RAW/WAW/WAR over arena slots +
/// KV-layer conflicts; the first dispatch never fences; non-head
/// members of a group always fence.
pub fn barrier_flags(sigs: &[HazardSig]) -> Vec<bool> {
    barrier_flags_with(sigs, true)
}

/// `serialize_group_tails` selects the GROUP semantic: instruction
/// selection's groups are fan_outs (multiple rows of one subgraph
/// SERIALIZE after their head — tails always fence); the tape's
/// groups are WAVES (same level = independent by construction — the
/// colorer's level-granular lifetimes guarantee no same-level slot
/// conflicts — so tails do NOT fence and only cross-group hazards
/// do). Passing `true` reproduces the instruction-selection walk.
pub fn barrier_flags_with(sigs: &[HazardSig], serialize_group_tails: bool) -> Vec<bool> {
    use std::collections::HashSet;
    let mut flags = Vec::with_capacity(sigs.len());
    let mut pw: HashSet<u32> = HashSet::new();
    let mut pr: HashSet<u32> = HashSet::new();
    let mut pkw: HashSet<u32> = HashSet::new();
    let mut pkr: HashSet<u32> = HashSet::new();
    let mut first = true;
    let mut prev_group: Option<usize> = None;
    for sig in sigs {
        if sig.metadata {
            flags.push(false);
            continue;
        }
        let group_tail = serialize_group_tails && prev_group == Some(sig.group);
        let arena_conflict = sig.reads.iter().any(|s| pw.contains(s))
            || sig.writes.iter().any(|s| pw.contains(s))
            || sig.writes.iter().any(|s| pr.contains(s));
        let kv_conflict = sig.kv_r.map(|l| pkw.contains(&l)).unwrap_or(false)
            || sig
                .kv_w
                .map(|l| pkw.contains(&l) || pkr.contains(&l))
                .unwrap_or(false);
        let need = if group_tail {
            true
        } else {
            !first && (arena_conflict || kv_conflict)
        };
        if need && !group_tail {
            pw.clear();
            pr.clear();
            pkw.clear();
            pkr.clear();
        }
        if need && group_tail {
            // Group tails serialize but the instruction-selection walk
            // computed ONE hazard flush per fan_out — the pending sets
            // were cleared (or not) at the group head only.
        }
        for &s in &sig.writes {
            pw.insert(s);
        }
        for &s in &sig.reads {
            pr.insert(s);
        }
        if let Some(l) = sig.kv_w {
            pkw.insert(l);
        }
        if let Some(l) = sig.kv_r {
            pkr.insert(l);
        }
        flags.push(need);
        first = false;
        prev_group = Some(sig.group);
    }
    flags
}

// ── Tape-owned slot coloring (M2b step 2) ───────────────────────────
//
// A linear-scan colorer over the TAPE's emission order — the
// replacement for `colored_slot_map`'s solver-order coloring on the
// swapped path. Same invariants, tape vocabulary:
//   - emission position = (wave level, tape index);
//   - alias collapse: an output that shares storage with an upstream
//     op's output pins to that owner's color (chains resolve);
//   - shape-partitioned pools keyed by (rows, cols) — a freed color
//     returns to its own pool only;
//   - inputs of an op that reads ACROSS its input (Gemm, attention
//     classes) stay live through the op's def;
//   - the embedded-hidden source is COLOR 0 (the bridge's head emits
//     the embed at slot 0);
//   - the result op's color is protected (never reused).
//
// Ops outside the piloted vocabulary refuse loudly — the per-arch
// flip only claims arches whose tapes color end to end.
pub fn tape_slot_map(
    lowered: &LoweredDecode,
    plan: &FoldPlan,
    levels: &[usize],
) -> Result<(SlotMap, u32, u32), String> {
    use scratchy_subtile::lower::LoweredOp as L;
    let ops = &lowered.input.ops;
    let n = ops.len();
    assert_eq!(levels.len(), n, "levels parallel to ops");

    // Output widths, propagated through the tape (pool partitioning).
    let width = op_widths(lowered);

    // Storage-alias per op: `Some(j)` = op i's output IS op j's
    // buffer (in-place mutation or view). Mirrors the alias_rules
    // vocabulary evaluated over tape op kinds + the fold plan.
    let in_op_at = |i: usize, k: usize| -> Option<usize> {
        match ops[i].inputs.get(k) {
            Some(InputRef::Op(j)) => Some(*j),
            _ => None,
        }
    };
    // Rope ops that belong to a FUSED-NORM fold (the gemma-4 class):
    // their alias peels back to the raw projection, while an unfused
    // rope aliases its direct input. The fold plan is the authority —
    // the same plan the emission arms read.
    let normed_rope_ops: std::collections::BTreeSet<usize> = plan
        .rope_normed
        .keys()
        .copied()
        .chain(
            plan.fused_into
                .iter()
                .enumerate()
                .filter(|(_, f)| f.is_some_and(|w| plan.rope_normed.contains_key(&w)))
                .map(|(j, _)| j),
        )
        .collect();
    let mut alias_of: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        // The unconditional in-place facts (Reshape views operand 0;
        // `add_inplace` mutates the residual at operand 1; the
        // elementwise activations rewrite their input and rebind it
        // as the output) are DECLARED in `metal_op_abi`, because the
        // emitter arms need the same fact and a disagreement between
        // the two is invisible until the colorer hands out a second
        // color — which is how gemma-class logits once minted a
        // VOCAB-SIZED slot, and gemma2's Gelu an INTERMEDIATE-sized
        // one. Contextual aliasing (fold-dependent, value-dependent)
        // stays below as code.
        if let Some(k) = scratchy_target_metal::op_abi::metal_alias_operand(&ops[i].op) {
            alias_of[i] = in_op_at(i, k as usize);
        }
        match &ops[i].op {
            // The fused-norm winner's norm output mutates the delta
            // buffer in place (NormDeltaResidual).
            L::RmsNorm { .. } => {
                let folded_add = plan
                    .fused_into
                    .iter()
                    .position(|f| *f == Some(i))
                    .filter(|a| matches!(ops[*a].op, L::Add));
                if let Some(a) = folded_add {
                    alias_of[i] = in_op_at(a, 0);
                }
            }
            // Identity multiply is elided — a pure alias.
            L::ScalarMul { scale } if *scale == 1.0 => alias_of[i] = in_op_at(i, 0),
            // Rope is in place on the raw projections
            // (RopeTripleAlias): the rotate half mutates raw q, the
            // append half mutates raw k (v goes to the cache; its
            // tape output is dead — attention reads the cache).
            // Rope is in place on the RAW PROJECTION, which for the
            // fused-norm class (gemma-4: per-head qk-norm on a
            // reshaped view) sits behind a Reshape → norm → Reshape
            // chain. Instruction selection peels exactly this chain
            // (alias_rules::RopeTripleAliasPeeled); aliasing the
            // DIRECT input instead put the rope's output on the
            // norm's color and produced garble on device.
            L::RopeRotate { .. } | L::RopeAppend { .. } if normed_rope_ops.contains(&i) => {
                let mut cur = in_op_at(i, 0);
                let mut hops = 0;
                while let Some(j) = cur {
                    if !matches!(
                        ops[j].op,
                        L::Reshape { .. } | L::RmsNorm { .. } | L::RmsNormUnit { .. }
                    ) {
                        break;
                    }
                    hops += 1;
                    assert!(hops < 8, "rope alias peel runaway at op {i}");
                    match in_op_at(j, 0) {
                        Some(k) => cur = Some(k),
                        None => break,
                    }
                }
                alias_of[i] = cur;
            }
            // The UNFUSED rope (qwen3's per-head qk-norm class) aliases
            // its DIRECT input — instruction selection's
            // RopeTripleAlias. Peeling here emitted nothing on device.
            L::RopeRotate { .. } | L::RopeAppend { .. } => alias_of[i] = in_op_at(i, 0),
            _ => {}
        }
    }
    // The FIRST residual add's residual operand is the embedded
    // hidden (an Ext source, not an op): the kernel mutates the
    // embed buffer in place, so that add — and via the alias chain
    // every later residual add — IS color 0. Without the pin the
    // colorer mints a fresh color the kernel never writes.
    let mut pinned_zero: Vec<bool> = vec![false; n];
    for i in 0..n {
        // The embed normalizer (gemma's sqrt(d) ScalarMul) reads the
        // embedded hidden and `scale_inplace` MUTATES that buffer —
        // instruction selection emits `ScalarMul(0, 0, ..)`, writing
        // the embed slot. Pin it to colour 0 like the residual adds,
        // or the tape writes a fresh slot and every later read of the
        // scaled hidden is off-buffer.
        if matches!(ops[i].op, L::ScalarMul { .. })
            && let Some(InputRef::Ext(e)) = ops[i].inputs.first()
            && matches!(
                lowered.bindings.get(*e),
                Some(scratchy_subtile::handoff::SourceBinding::EmbeddedHidden)
            )
        {
            pinned_zero[i] = true;
        }
        if matches!(ops[i].op, L::Add)
            && let Some(InputRef::Ext(e)) = ops[i].inputs.get(1)
            && matches!(
                lowered.bindings.get(*e),
                Some(scratchy_subtile::handoff::SourceBinding::EmbeddedHidden)
            )
        {
            pinned_zero[i] = true;
        }
    }
    let resolve = |mut i: usize| -> usize {
        let mut hops = 0;
        while let Some(j) = alias_of[i] {
            i = j;
            hops += 1;
            assert!(hops <= n, "alias cycle at op {i}");
        }
        i
    };

    // Emission order + positions.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (levels[i], i));
    let mut pos: Vec<usize> = vec![0; n];
    for (p, &i) in order.iter().enumerate() {
        pos[i] = p;
    }

    // Last use per OWNER op (alias chains resolved), at WAVE-LEVEL
    // granularity: ops of one level dispatch concurrently between
    // barriers, so a buffer whose last reader sits at level L is
    // reusable only at level > L — freeing at position granularity
    // hands the slot to a same-level op and races the read on
    // device (fluent-garble signature, reproduced on the llama
    // pilot before this rule).
    let mut last_use_lvl: Vec<usize> = (0..n).map(|i| levels[i]).collect();
    for i in 0..n {
        for r in &ops[i].inputs {
            if let InputRef::Op(j) = r {
                let o = resolve(*j);
                last_use_lvl[o] = last_use_lvl[o].max(levels[i]);
            }
        }
    }

    // Linear scan.
    let mut active: Vec<(usize, u32, usize)> = Vec::new(); // (last_use, color, owner op)
    let mut free_by_shape: std::collections::HashMap<(u32, u32), std::collections::BTreeSet<u32>> =
        std::collections::HashMap::new();
    let mut color_shape: std::collections::HashMap<u32, (u32, u32)> =
        std::collections::HashMap::new();
    let mut color_of: Vec<Option<u32>> = vec![None; n];
    // Color 0 = the embedded hidden (the bridge head's embed slot).
    let mut next_color: u32 = 1;
    color_shape.insert(0, (0, 0)); // never pooled: shape key no op matches
    let result_op = resolve(lowered.input.result);

    for &i in &order {
        let cur_lvl = levels[i];
        let reads_across = matches!(
            ops[i].op,
            L::Gemm { .. } | L::VisionRope | L::VarlenAttention { .. } | L::EncoderAttn { .. }
        );
        let cur_inputs: std::collections::HashSet<usize> = if reads_across {
            ops[i]
                .inputs
                .iter()
                .filter_map(|r| match r {
                    InputRef::Op(j) => Some(resolve(*j)),
                    InputRef::Ext(_) => None,
                })
                .collect()
        } else {
            Default::default()
        };
        active.retain(|&(lu, color, owner)| {
            if lu < cur_lvl && !cur_inputs.contains(&owner) && owner != result_op {
                let s = color_shape[&color];
                free_by_shape.entry(s).or_default().insert(color);
                false
            } else {
                true
            }
        });

        if pinned_zero[resolve(i)] || pinned_zero[i] {
            color_of[i] = Some(0);
            continue;
        }
        if let Some(_j) = alias_of[i] {
            let owner = resolve(i);
            let c = color_of[owner]
                .ok_or_else(|| format!("tape colorer: alias target of op {i} uncolored"))?;
            color_of[i] = Some(c);
            continue;
        }
        let shape = (ops[i].m, width[i]);
        let pool = free_by_shape.entry(shape).or_default();
        let c = if let Some(&c) = pool.iter().next() {
            pool.remove(&c);
            c
        } else {
            let c = next_color;
            next_color += 1;
            color_shape.insert(c, shape);
            c
        };
        color_of[i] = Some(c);
        active.push((last_use_lvl[i], c, i));
    }

    // ⭐ TWINS APPLIED AS A POST-PASS, so they cannot depend on scan ORDER. Doing it inside the
    // scan only worked when the twin's target had already been coloured; a target later in the
    // emission order left the twin silently unapplied, which is how a class copy kept its own
    // slot and the rolled body stopped matching.
    //
    // Same exclusions as before: an op pinned to colour 0, or one that aliases another, follows
    // that constraint instead — those chains thread ACROSS layers and are already uniform.
    // ⛔ THE INVARIANT THE ARENA ACTUALLY RESTS ON: two ops may share a colour only if their
    // live ranges are disjoint. The scan frees a colour when `last_use < cur_level`, so this is
    // the same test applied to the FINISHED colouring.
    //
    // ⭐ AND IT IS THE ONLY CHECK THAT CAN SEE A BAD COLOURING. The roll proof expands the rolled
    // emission and compares it against the un-rolled one — but BOTH are emitted through this same
    // map, so a colouring that overlaps two live values is equally wrong on each side and the
    // proof passes. That is exactly how a class-split colourer once shipped green and produced
    // `lelelelelele` on the card. This fails at macro expansion instead.
    //
    // Colour 0 and aliased ops are exempt BY CONSTRUCTION: the residual chain is pinned to 0 and
    // threads through every layer in place, and an alias shares its owner's buffer deliberately.
    {
        let mut by_color: std::collections::HashMap<u32, Vec<(usize, usize, usize)>> =
            Default::default();
        for i in 0..n {
            if alias_of[i].is_some() || pinned_zero[i] || pinned_zero[resolve(i)] {
                continue;
            }
            if let Some(c) = color_of[i]
                && c != 0
            {
                by_color
                    .entry(c)
                    .or_default()
                    .push((levels[i], last_use_lvl[i], i));
            }
        }
        for (c, mut v) in by_color {
            v.sort_unstable();
            for w in v.windows(2) {
                let (a_def, a_last, a_op) = w[0];
                let (b_def, b_last, b_op) = w[1];
                if a_last >= b_def {
                    return Err(format!(
                        "tape colorer: colour {c} is held by op {a_op} (live {a_def}..={a_last}) \
                         and op {b_op} (live {b_def}..={b_last}) AT THE SAME TIME — one kernel \
                         would write over a buffer the other still owns"
                    ));
                }
            }
        }
    }

    // Project colors onto the (tile, slot) keys the bridge queries.
    let mut sm = SlotMap::new();
    for (i, t) in lowered.op_tiles.iter().enumerate() {
        if let (Some((tile, slot)), Some(c)) = (t, color_of[i]) {
            sm.insert_at(scratchy_subtile::handoff::TileId(*tile), *slot, c);
            // Two-output ops (GateSplit's gate half, VisionRope's k')
            // bind a SECOND buffer the emission reads via
            // `second_out_slot`. It is a distinct allocation, so it
            // gets its own color; no lifetime is tracked for it, which
            // is conservative (never reused) and correct.
            if matches!(ops[i].op, L::GateSplit { .. } | L::VisionRope) {
                let c2 = next_color;
                next_color += 1;
                color_shape.insert(c2, (ops[i].m, width[i]));
                sm.insert_at(scratchy_subtile::handoff::TileId(*tile), slot + 1, c2);
            }
        }
    }
    let final_slot =
        color_of[resolve(lowered.input.result)].ok_or("tape colorer: result op uncolored")?;
    // The folded (w + offset) gain-Add tiles: instruction selection
    // CLAIMS them, so their colored slot rides the norm's write and
    // the emission resolves them via `gain_add_slot`. They are not
    // tape ops — nothing in the scan colors them — so register each
    // against its norm's color, which is the buffer they share.
    for (op_idx, tile) in &lowered.norm_gain_add_tiles {
        // FILL ONLY: when a tape op already carries this tile's
        // provenance the projection above assigned its real color;
        // overwriting it changed gemma-3's emitted stream (caught by
        // its bit-exactness gate). This entry exists solely for the
        // tiles no op claims.
        let already = sm
            .entries()
            .any(|((t, sl), _)| *t == scratchy_subtile::handoff::TileId(*tile) && *sl == 0);
        if already {
            continue;
        }
        if let Some(c) = color_of.get(*op_idx).copied().flatten() {
            sm.insert_at(scratchy_subtile::handoff::TileId(*tile), 0, c);
        }
    }
    Ok((sm, next_color, final_slot))
}

/// The accessor FIELD NAMES a bridged stream implies — one per
/// distinct weight-slot base across every emitted instruction.
///
/// This is the tape-side replacement for `collect_accessors`'s walk
/// over solved subgraphs' `Implementation::required_weights` — the
/// LAST instruction-selection dependency in the per-arch `Weights`
/// struct, and therefore the last thing keeping the solve alive for
/// a tape-scheduled arch.
pub fn tape_accessor_names(slots: &[Vec<WeightSlot>]) -> std::collections::BTreeSet<String> {
    slots
        .iter()
        .flat_map(|v| v.iter().map(|s| s.base.to_string()))
        .collect()
}

/// The accessor GROUP SIGNATURE a bridged stream implies:
/// `base -> (accessor type, sorted layer indices)`.
///
/// The layer index of each weight slot comes from the instruction's
/// own iteration-index field — the one the opcode-shape table names
/// `layer` and loop compression already keys on — so this reads the
/// SAME fact the emitted stream carries, not a re-derivation.
/// `None` in the layer set means the slot's instruction has no layer
/// field (an unindexed accessor: embed, final norm, lm_head).
pub fn tape_accessor_groups(
    instances: &[Instruction],
    slots: &[Vec<WeightSlot>],
    layer_field_of: &dyn Fn(&Instruction) -> Option<u64>,
) -> std::collections::BTreeMap<String, (String, std::collections::BTreeSet<Option<u64>>)> {
    let mut out: std::collections::BTreeMap<
        String,
        (String, std::collections::BTreeSet<Option<u64>>),
    > = Default::default();
    for (k, per_op) in slots.iter().enumerate() {
        let layer = instances.get(k).and_then(layer_field_of);
        for s in per_op {
            // The table holds the ACCESSOR return type (`&T` — a
            // borrow of the stored field); the struct FIELD type is
            // the owned `T`. Strip the borrow: one table, both
            // positions, with the relationship stated rather than a
            // second table.
            let ty = s
                .kind
                .rust_type_name()
                .replace(' ', "")
                .trim_start_matches('&')
                .to_string();
            let e = out
                .entry(s.base.to_string())
                .or_insert((ty, Default::default()));
            e.1.insert(layer);
        }
    }
    out
}

// ── Emission helpers that outlived instruction selection ──────────

use proc_macro2::TokenStream;
use quote::quote;

pub fn emit_bucket_barriers_static(static_ident: &syn::Ident, barriers: &[bool]) -> TokenStream {
    let elements = barriers.iter().map(|b| {
        if *b {
            quote! { true }
        } else {
            quote! { false }
        }
    });
    quote! {
        #[cfg(feature = "metal")]
        static #static_ident: &[bool] = &[ #(#elements),* ];
    }
}

/// The MLX qmv batch limit for an affine-quantized matmul — the one
/// helper that outlived `macros/src/metal/`: it reads the target
/// crate's table, so it is emission data, not instruction selection.
pub fn affine_qmm_vector_limit(k: u32, n: u32) -> u32 {
    scratchy_target_metal::quantized::get_qmv_batch_limit(
        k,
        n,
        scratchy_target_metal::targets::AppleSiliconGen::M4,
    )
}
