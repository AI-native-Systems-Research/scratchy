// SPDX-License-Identifier: Apache-2.0
//! THE model's weight binding, derived ONCE from the shared tape.
//!
//! Which tensors a model needs, what kind each is, and which layer it belongs
//! to are facts about the MODEL, not about a target. They are already carried,
//! target-agnostically, by the artifact both backends lower:
//! `LoweredDecode.bindings` (`SourceBinding::Weight { id, index }`) plus the
//! op that consumes each binding.
//!
//! This walk reads exactly that and nothing else — no instruction stream, no
//! kernel ABI — so every target gets the SAME answer and a new
//! target inherits it by construction rather than by re-deriving it.
//!
//! ⛔ WHY THIS EXISTS AT ALL. Without it the accessor set had one producer and
//! it was target-locked: one target's instruction-stream bridge. Under any other
//! target it is empty, so the emitted `Weights` struct carried NO weight fields — verified
//! in generated output:
//!
//! ```ignore
//! pub struct Weights { pub rotary: RotaryCache }   // that is all
//! ```
//!
//! and that target's worker consequently re-found all 362 granite tensors at
//! RUNTIME by string-formatting `"{disk}.weight"` off a JSON manifest. That
//! string rule cannot express a bare `nn.Parameter` (gemma-4's `layer_scalar`,
//! which has no `.weight` suffix) or a non-default decoder root
//! (`model.language_model`), which is why gemma-4 could not run there. The
//! fix is not a better string rule; it is emitting the fields at all.

use crate::classified::{Program, UnrollIndex, WeightId};
use crate::to_wavefront::{LoweredDecode, SourceBinding};
use crate::weight_vocab::{WeightAccessor, WeightKind};
use scratchy_subtile::lower::{GemmWeight, InputRef};
use scratchy_subtile::subtile_ir::{EwKind, SubOp};

/// The weight-binding external of op `op_idx`, if it binds one.
///
/// Mirrors the bridge's `weight_ext`: an op binds at most one weight-carrying
/// external, identified by the BINDING (not by position), so an op whose
/// operand order changes still resolves the same tensor.
/// How the target's kernels want an MLP's gate and up projections stored.
///
/// ⛔ NOT A `bool`. It decides the NAME of an emitted field, and the emitted load body recovers
/// that field's sources by splitting the name on `__fused__` — so the wrong answer is not a
/// wrong flag, it is a field that resolves to no safetensors key.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MlpPacking {
    /// ONE buffer holding both, named `{gate}__fused__{up}` — a fused-MLP kernel reads a
    /// single packed weight.
    Packed,
    /// Two weights: an MLP of two matmuls plus a `SiluMul`, with no packed-buffer kernel.
    Split,
}

/// How the embedding table is stored.
///
/// ⛔ THIS IS NOT VISIBLE FROM THE TAPE. The gather `embed_tokens[id]` is a host lookup before
/// the first op, so it binds to no tape op — its storage lives in the FUF node. Getting it
/// wrong does not mis-name anything; it emits a `load` body that calls `load_affine_dequant` on
/// a dense `Embedding`, which does not compile. That is the good case.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmbedStorage {
    Dense,
    AffineQuant,
}

/// THE TARGET'S WEIGHT ABI — every fact the shared accessor walk needs and cannot derive.
///
/// ⭐⭐ ONE ARGUMENT, NAMED FIELDS, NO BOOLEANS. This was two adjacent `bool` parameters, which
/// is the same hazard in miniature that this whole file exists to remove: `from_tape(l, program,
/// false, false)` says nothing about which `false` is which, and swapping them COMPILES. A
/// struct literal has to name what it is setting, and the two fields are not even the same type,
/// so the transposed call is a compile error rather than a model that loads the wrong weights.
///
/// It is data, per CLAUDE.md: target facts are const tables the shared passes take as INPUT,
/// never logic and never a second walk.
#[derive(Clone, Copy, Debug)]
pub struct WeightAbi {
    pub mlp: MlpPacking,
    pub embed: EmbedStorage,
}

/// The accessor BASE of a weight: its path joined. Its layer is its unroll index's alone, appended
/// downstream; a path ending in a numeral (Qwen2-VL's `merger.mlp.0`, `merger.mlp.2`) is its own
/// base. This is a target bridge's `base` exactly — two spellings of one base is what this file
/// exists to prevent.
fn accessor_base(program: &Program, id: WeightId) -> String {
    program.weights.path(id).join("_")
}

/// The field name of a gate/up pair PACKED INTO ONE weight: `{gate}__fused__{up}` plus the
/// layer, which is the spelling the load body inverts by splitting on `__fused__`.
fn fused_field_name(
    program: &Program,
    gate: WeightId,
    up: WeightId,
    index: Option<UnrollIndex>,
) -> syn::Ident {
    let base = format!(
        "{}__fused__{}",
        accessor_base(program, gate),
        accessor_base(program, up)
    );
    match index {
        Some(i) => quote::format_ident!("{}_{}", base, i.0),
        None => quote::format_ident!("{}", base),
    }
}

/// The gate/up pair a `Mul` closes, when the op is the tail of an MLP
/// `silu(gate(x)) * up(x)`.
///
/// ⭐ ONE DETECTION, TWO CONSUMERS. A target's bridge finds this same triple to emit
/// `FusedGateUpSiluMul`; the accessor set finds it to name ONE packed weight field instead of
/// two. They were separate three-line walks over the same shape, which is two chances to
/// disagree about what "fused" means — and the accessor name is derived from the pair, so a
/// disagreement is a field the load body cannot resolve.
///
/// Returns `(silu_or_gelu, gate, up)` as op indices.
pub(crate) fn gate_up_pair(
    lowered: &LoweredDecode,
    mul_op: usize,
) -> Option<(usize, usize, usize)> {
    let in_op = |r: &InputRef| match r {
        InputRef::Op(i) => Some(*i),
        InputRef::Ext(_) => None,
    };
    let ops = &lowered.input.ops;
    if !matches!(ops[mul_op].op, SubOp::Elementwise(EwKind::Mul)) {
        return None;
    }
    let si = in_op(&ops[mul_op].inputs[0])?;
    let gate = in_op(ops[si].inputs.first()?)?;
    let up = in_op(ops[mul_op].inputs.get(1)?)?;
    Some((si, gate, up))
}

fn weight_ext(lowered: &LoweredDecode, op_idx: usize) -> Option<usize> {
    weight_exts(lowered, op_idx).first().copied()
}

/// Does this op read a mean-centered activation — i.e. is its operand 0 the `Sub` half of a
/// LayerNorm's `Mean -> Sub` centering?
fn reads_centered(lowered: &LoweredDecode, op_idx: usize) -> bool {
    let centered = matches!(
        lowered.input.ops[op_idx].inputs.first(),
        Some(InputRef::Op(src))
            if matches!(lowered.input.ops[*src].op, SubOp::Elementwise(EwKind::Sub))
    );
    // ⛔ CENTERING ALONE IS NOT ENOUGH. modernbert has a norm that reads a centered activation
    // and is nonetheless declared `RmsNorm` — judging on the `Sub` alone flipped it the other
    // way (`expected &RmsNorm, found &LayerNorm`). A LayerNorm is centre + scale + BIAS; an
    // RmsNorm is scale only. So the bias consumer is the half that actually separates them.
    centered
        && lowered.input.ops.iter().any(|d| {
            matches!(d.op, SubOp::Elementwise(EwKind::BiasAdd))
                && d.inputs
                    .iter()
                    .any(|r| matches!(r, InputRef::Op(j) if *j == op_idx))
        })
}

/// EVERY weight-carrying source an op binds, in operand order.
///
/// ⛔ AN OP CAN BIND MORE THAN ONE (an fp8 matmul binds its weight and its scale), and taking
/// only the first is a silent wrong answer. Ops that bind one weight are the common case, not the
/// only one.
fn weight_exts(lowered: &LoweredDecode, op_idx: usize) -> Vec<usize> {
    lowered.input.ops[op_idx]
        .inputs
        .iter()
        .filter_map(|r| match r {
            InputRef::Ext(e)
                if matches!(
                    lowered.bindings()[*e],
                    SourceBinding::Weight { .. }
                        | SourceBinding::WeightScale { .. }
                        | SourceBinding::BundleTensor { .. }
                ) =>
            {
                Some(*e)
            }
            _ => None,
        })
        .collect()
}

/// `(WeightId, layer)` of a weight-carrying binding.
fn weight_of(lowered: &LoweredDecode, ext: usize) -> Option<(WeightId, Option<UnrollIndex>)> {
    match &lowered.bindings()[ext] {
        SourceBinding::Weight { id, index }
        | SourceBinding::WeightScale { id, index }
        | SourceBinding::BundleTensor { id, index, .. } => Some((WeightId(*id), *index)),
        _ => None,
    }
}

/// The kind of weight an op binds — a function of the OP and its own quant
/// form, both of which live on the shared tape.
///
/// ⛔ RETURNS `Err` RATHER THAN A DEFAULT for an op this walk has not been
/// taught. A silent "assume Linear" would emit a `Weights` field of the wrong
/// type and fail far away at load, so a new weight-binding opcode is a build
/// error here — the same discipline a target's colour rule's exhaustive match
/// uses.
fn kind_of(lowered: &LoweredDecode, op_idx: usize) -> Result<WeightKind, String> {
    use SubOp as L;
    let op = &lowered.input.ops[op_idx].op;
    Ok(match op {
        // ⭐ A LAYERNORM IS A SHAPE, NOT AN OPCODE. The DSL declares a `LayerNorm` field, but by
        // the time it reaches the tape it is a `Mean -> Sub -> RmsNorm -> BiasAdd` chain, and the
        // gain hangs off the RmsNorm. Judging that RmsNorm on its own calls it a `RmsNorm` and
        // emits an accessor of the wrong type — `expected &LayerNorm, found &RmsNorm` on
        // qwen2-vl and locateanything. The centering `Sub` on its input is what distinguishes
        // the two, so the kind is read from the op's NEIGHBOURHOOD, not the op alone.
        L::RmsNorm { .. } if reads_centered(lowered, op_idx) => WeightKind::LayerNorm,
        L::MatmulTile { weight, .. } => match weight {
            GemmWeight::Dense => WeightKind::Linear,
            GemmWeight::Fp8Dynamic => WeightKind::Fp8,
            // The quantized-linear kinds are named by the PRESET, which the
            // tape carries on the gemm itself.
            GemmWeight::Affine { .. } => WeightKind::Linear,
        },
        L::RmsNorm { .. } | L::RmsNormUnit { .. } => WeightKind::RmsNorm,
        L::EmbeddingGather { .. } => WeightKind::Embedding,
        L::GatedDeltaNet | L::Mamba2 { .. } => WeightKind::GatedDeltaNet,
        // A MoE block's steps bind their router's or their experts' bundle — the bundle table.
        L::RouterNorm { router, .. }
        | L::RouterLogits { router, .. }
        | L::RouteExpertScale { router }
        | L::RouteSigmoidBias { router }
        | L::RouteBias { router } => router.weight_kind(),
        L::ExpertMatmul { bundle, .. } => bundle.weight_kind(),
        L::RopeRotate { .. } | L::RopeAppend { .. } => WeightKind::CosSin,
        // gpt-oss attention sinks: the trailing `[num_heads]` weight
        // input on the decode op (a plain AttnDecode binds no weight,
        // so this arm only fires on the 6-input sinks form).
        L::AttnDecode { .. } => WeightKind::AttnSinks,
        // A standalone (unfused) per-layer scalar parameter — granite's
        // `layer_scalar` class — reads through the norm accessor. Unlike
        // `BiasAdd` (skipped before this is ever called; see `from_tape`'s
        // main loop), a `ScalarWeightMul` genuinely owns its weight: it does
        // not ride on an upstream gemm's accessor.
        L::ScalarWeightMul => WeightKind::RmsNorm,
        // A DFlash2 drafter's: its attention's q/k norms and its commit's k norm are per-head
        // gains; its convolutions' base kernels and its selector's codebooks bare tensors.
        L::DraftAttention { .. } | L::DraftContextKv { .. } => WeightKind::RmsNorm,
        L::DraftConv { .. } | L::DraftSelect => WeightKind::Tensor,
        other => {
            return Err(format!(
                "weight_bindings: op {other:?} binds a weight but this walk has no kind for it — \
                 add its arm rather than defaulting, or the emitted Weights field gets the wrong type"
            ));
        }
    })
}

/// The struct FIELD type for a weight kind.
///
/// ⛔ `weight_kind_rust_type` yields the ACCESSOR type `&T` — a borrow of the
/// stored field, which is what an accessor RETURNS. The field itself stores the
/// owned `T`, so the borrow is stripped. Emitting the borrow verbatim is a
/// `missing lifetime specifier` on every field.
fn field_type(kind: &WeightKind) -> Result<proc_macro2::TokenStream, String> {
    crate::codegen::weight_kind_rust_type(kind)
        .to_string()
        .replace(' ', "")
        .trim_start_matches('&')
        .parse()
        .map_err(|e| format!("weight_bindings: accessor type does not re-parse: {e}"))
}

/// The model's weight binding: the accessor set the `Weights` struct is built
/// from, plus the KIND of each field.
///
/// The kind is carried alongside rather than recovered from `rust_type` later,
/// because the binding emission needs it to know how to reach the tensor INSIDE
/// a field (`.weight` on a norm, a variant match on a quantized linear) and
/// re-deriving it by string-matching the emitted type would be a second answer
/// to a question already settled here.
pub struct TapeWeights {
    pub accessors: Vec<WeightAccessor>,
    /// Emitted field name → kind. Keyed by the accessor's own name, so it works
    /// for every group shape (`base`, `base[L]`, `base_L`).
    pub kind_of_field: std::collections::BTreeMap<String, WeightKind>,
}

/// Every weight the model binds, as accessors + kinds — THE input to the
/// `Weights` struct emission, for every target.
///
/// One entry per distinct `(base, layer)`: a weight bound by several ops (a
/// norm read twice, a tied head) is ONE field, deduped here rather than by the
/// consumer.
pub fn from_tape(
    lowered: &LoweredDecode,
    program: &Program,
    // ⭐ THE TARGET FACTS, TAKEN AS INPUT. Packing was the ONLY thing the two accessor sets
    // ever disagreed about (measured on llama-3.2-1b: 131 vs 147, differing in exactly the 16
    // fused entries against 32 separate ones, the other 115 identical). They arrive as data
    // rather than being re-derived, so this stays one walk every target gets one answer from.
    abi: WeightAbi,
) -> Result<TapeWeights, String> {
    // Keyed by the emitted field name so a repeat binding cannot mint a second
    // field for one tensor.
    let mut by_field: std::collections::BTreeMap<String, WeightAccessor> = Default::default();
    let mut kind_of_field: std::collections::BTreeMap<String, WeightKind> = Default::default();

    // Which gemms are the halves of a packed gate/up, and which of the pair each one is. The
    // GATE carries the packed accessor (its base leads the name); the UP is then skipped, or it
    // would mint a second field for a buffer that no longer exists on its own.
    let mut packed_partner: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut packed_skip: std::collections::BTreeSet<usize> = Default::default();
    if abi.mlp == MlpPacking::Packed {
        for op_idx in 0..lowered.input.ops.len() {
            if let Some((_, gate, up)) = gate_up_pair(lowered, op_idx) {
                packed_partner.insert(gate, up);
                packed_skip.insert(up);
            }
        }
    }

    for op_idx in 0..lowered.input.ops.len() {
        if packed_skip.contains(&op_idx) {
            continue;
        }
        // `BiasAdd`'s own weight operand mints no field of its own — the bias
        // rides on its upstream gemm's accessor (`AffineQuantLinear`/`Linear`
        // both carry an optional bias loaded off the SAME prefix), matching
        // a target's own `BiasAdd` step lowering, which derives
        // base/kind from the upstream gemm and never reads this op's weight
        // input. Minting a second, independent accessor here for the same
        // on-disk tensor produced a spurious `<gemm>.bias`-named field that
        // tried to load it a second time under a `RmsNorm` kind — panicking at
        // load with `weight not found: ...bias.weight` on any arch whose
        // gemm+bias pair reaches this walk (e.g. qwen2's q/k/v_proj bias).
        if matches!(
            lowered.input.ops[op_idx].op,
            SubOp::Elementwise(EwKind::BiasAdd)
        ) {
            continue;
        }
        // ⭐ ONE ACCESSOR PER BOUND WEIGHT, NOT PER OP.
        for ext in weight_exts(lowered, op_idx) {
            let Some((id, index)) = weight_of(lowered, ext) else {
                continue;
            };
            let kind = kind_of(lowered, op_idx)?;
            // A packed pair is ONE field named for both halves, whose sources are both weights in
            // the order the name spells them — which is how the load body recovers them.
            if let Some(&up_op) = packed_partner.get(&op_idx) {
                let Some((up_id, up_index)) =
                    weight_ext(lowered, up_op).and_then(|e| weight_of(lowered, e))
                else {
                    return Err(format!(
                        "op {op_idx}: gate/up pair's up-projection at op {up_op} binds no weight"
                    ));
                };
                let name = fused_field_name(program, id, up_id, index);
                let owned = field_type(&kind)?;
                kind_of_field.insert(name.to_string(), kind.clone());
                by_field
                    .entry(name.to_string())
                    .or_insert_with(|| WeightAccessor {
                        name,
                        rust_type: owned,
                        kind,
                        source_weights: vec![(id, index), (up_id, up_index)],
                    });
                continue;
            }
            let name = crate::emit::weight_field_name(program, id, index);
            let owned = field_type(&kind)?;
            kind_of_field.insert(name.to_string(), kind.clone());
            by_field
                .entry(name.to_string())
                .or_insert_with(|| WeightAccessor {
                    name,
                    rust_type: owned,
                    kind,
                    source_weights: vec![(id, index)],
                });
        }
    }

    // ── The embedding table ──────────────────────────────────────────
    //
    // ⛔ NOT A TAPE OP, SO THE WALK ABOVE CANNOT SEE IT. The tape begins at
    // the ALREADY-EMBEDDED hidden row: the gather `embed_tokens[input_id]` is
    // a cheap host lookup the runtime performs before the first op, so the
    // embedding binds to no tape op and appears in no `SourceBinding`.
    // It is nonetheless a weight the model must load — and the one a tied
    // `lm_head` redirects to — so it is resolved the same way the bridge
    // resolves its own `embed_base`: by name against the program's weight
    // table. Omitting it emits a `Weights` whose `load` body references an
    // `embed_tokens` field that was never declared.
    let embed =
        (program.weights.ids()).find(|id| program.weights.path(*id).join("_").contains("embed"));
    if let Some(id) = embed {
        let kind = match abi.embed {
            EmbedStorage::AffineQuant => WeightKind::AffineQuantEmbedding,
            EmbedStorage::Dense => WeightKind::Embedding,
        };
        let name = crate::emit::weight_field_name(program, id, None);
        let owned = field_type(&kind)?;
        kind_of_field.insert(name.to_string(), kind.clone());
        by_field
            .entry(name.to_string())
            .or_insert_with(|| WeightAccessor {
                name,
                rust_type: owned,
                kind,
                source_weights: vec![(id, None)],
            });
    }

    Ok(TapeWeights {
        accessors: by_field.into_values().collect(),
        kind_of_field,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classified::Prelude;
    use scratchy_subtile::lower::{LoweringInput, OpDesc};
    use scratchy_subtile::subtile_ir::SourceShape;

    /// A `Program` whose weight table interns `paths`, in order, so
    /// `WeightId(i)` names `paths[i]`.
    fn program_with(paths: &[&[&str]]) -> Program {
        let mut weights = crate::classified::WeightTable::default();
        for p in paths {
            let _ = weights.intern_str(p.iter().map(|s| (*s).to_string()).collect());
        }
        Program {
            statements: Vec::new(),
            locals: Default::default(),
            weights,
            reshape_targets: Default::default(),
            dim_casts: Default::default(),
            prelude: Prelude::Decoder,
            decoder_safetensors_prefix: None,
            weight_leaf_renames: Vec::new(),
        }
    }

    fn gemm(w_ext: usize) -> OpDesc {
        OpDesc {
            op: SubOp::MatmulTile {
                n: 8,
                weight: GemmWeight::Dense,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(w_ext)],
        }
    }

    fn norm(w_ext: usize) -> OpDesc {
        OpDesc {
            op: SubOp::RmsNorm {
                eps: 1e-5,
                gain: scratchy_subtile::subtile_ir::GainConvention::Scale,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(w_ext)],
        }
    }

    /// `silu(x)` over op `src` — the middle of a gate/up MLP tail.
    fn silu(src: usize) -> OpDesc {
        OpDesc {
            op: SubOp::Elementwise(EwKind::Silu),
            m: 1,
            inputs: vec![InputRef::Op(src)],
        }
    }

    /// `a * b` over two ops — the `Mul` that closes the tail.
    fn mul(a: usize, b: usize) -> OpDesc {
        OpDesc {
            op: SubOp::Elementwise(EwKind::Mul),
            m: 1,
            inputs: vec![InputRef::Op(a), InputRef::Op(b)],
        }
    }

    /// The MLP tail `silu(gate(x)) * up(x)`: ops 0=gate, 1=silu, 2=up, 3=mul.
    fn gate_up_tape() -> LoweredDecode {
        decode(
            vec![
                SourceBinding::EmbeddedHidden,
                SourceBinding::Weight { id: 0, index: None },
                SourceBinding::Weight { id: 1, index: None },
            ],
            vec![gemm(1), silu(0), gemm(2), mul(1, 2)],
        )
    }

    /// ⭐⭐ THE ONE THING THE TWO PRODUCERS EVER DISAGREED ABOUT, AS A TEST.
    ///
    /// A packed-MLP kernel reads gate and up from ONE buffer; a split-MLP target binds two
    /// weights. That is a target-ABI fact, so it is an ARGUMENT — and this pins both answers from
    /// one walk, which is the whole reason a target's inline producer could be deleted.
    ///
    /// The packed name is load-bearing, not cosmetic: the emitted load body recovers the
    /// sources by splitting it on `__fused__`, so a different spelling is a field that resolves
    /// to no safetensors key.
    #[test]
    fn packing_gate_up_is_one_field_named_for_both_and_unpacking_is_two() {
        let program = program_with(&[&["mlp", "gate_proj"], &["mlp", "up_proj"]]);

        let packed = from_tape(
            &gate_up_tape(),
            &program,
            WeightAbi {
                mlp: MlpPacking::Packed,
                embed: EmbedStorage::Dense,
            },
        )
        .expect("derives");
        let names: Vec<String> = packed
            .accessors
            .iter()
            .map(|a| a.name.to_string())
            .collect();
        assert!(
            names.contains(&"mlp_gate_proj__fused__mlp_up_proj".to_string()),
            "packed pair must be ONE field named for both halves, got {names:?}"
        );
        assert!(
            !names.iter().any(|n| n == "mlp_up_proj"),
            "the up half must NOT also get its own field — the buffer it named is gone: {names:?}"
        );
        let fused = packed
            .accessors
            .iter()
            .find(|a| a.name.to_string().contains("__fused__"))
            .expect("fused accessor");
        assert_eq!(
            fused.source_weights,
            vec![(WeightId(0), None), (WeightId(1), None)],
            "sources must be gate THEN up — the order the name spells, which is how the load \
             body inverts it"
        );

        // ⛔ NON-VACUITY: without the fact, the SAME tape yields two separate fields. If this
        // half ever matched the packed one, the argument would be doing nothing.
        let split = from_tape(
            &gate_up_tape(),
            &program,
            WeightAbi {
                mlp: MlpPacking::Split,
                embed: EmbedStorage::Dense,
            },
        )
        .expect("derives");
        let split_names: Vec<String> = split.accessors.iter().map(|a| a.name.to_string()).collect();
        assert!(
            split_names.contains(&"mlp_gate_proj".to_string())
                && split_names.contains(&"mlp_up_proj".to_string()),
            "unpacked must be two fields, got {split_names:?}"
        );
        assert!(
            !split_names.iter().any(|n| n.contains("__fused__")),
            "unpacked must name no packed buffer: {split_names:?}"
        );
    }

    fn decode(bindings: Vec<SourceBinding>, ops: Vec<OpDesc>) -> LoweredDecode {
        let n_ops = ops.len();
        LoweredDecode::new(
            LoweringInput {
                exported: None,
                sources: vec![SourceShape { rows: 1, cols: 8 }; bindings.len()],
                ops,
                result: n_ops.saturating_sub(1),
            },
            bindings,
            vec![None; n_ops],
            Default::default(),
            vec![None; n_ops],
            Vec::new(),
        )
        .expect("a binding fixture names at most one embed row")
    }

    /// ⭐ THE PROPERTY THE WHOLE GENERATED LOADER RESTS ON: one field per
    /// DISTINCT `(weight, layer)`, and a weight bound by several ops is ONE
    /// field. Before this walk existed the accessor set was empty off one target
    /// and the worker re-found every tensor by string, so "did we derive a
    /// field for each bound weight" is the invariant worth pinning.
    #[test]
    fn one_field_per_distinct_weight_and_layer() {
        // q_proj at layers 0 and 1 (two fields), and ONE norm bound by two
        // separate ops (one field, not two).
        let program = program_with(&[&["self_attn", "q_proj"], &["input_layernorm"]]);
        let tw = from_tape(
            &decode(
                vec![
                    SourceBinding::EmbeddedHidden,
                    SourceBinding::Weight {
                        id: 0,
                        index: Some(UnrollIndex(0)),
                    },
                    SourceBinding::Weight {
                        id: 0,
                        index: Some(UnrollIndex(1)),
                    },
                    SourceBinding::Weight {
                        id: 1,
                        index: Some(UnrollIndex(0)),
                    },
                ],
                vec![gemm(1), gemm(2), norm(3), norm(3)],
            ),
            &program,
            // these fixtures describe a split MLP and a dense embedding — the shape they
            // asserted before these facts were arguments
            WeightAbi {
                mlp: MlpPacking::Split,
                embed: EmbedStorage::Dense,
            },
        )
        .expect("derives");

        let mut names: Vec<String> = tw.accessors.iter().map(|a| a.name.to_string()).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "input_layernorm_0".to_string(),
                "self_attn_q_proj_0".to_string(),
                "self_attn_q_proj_1".to_string(),
            ],
            "one field per (weight, layer); the twice-bound norm must NOT mint two"
        );
    }

    /// The kind comes from the OP, not from the weight's name — a gemm's
    /// operand is a Linear even when it is called `input_layernorm`.
    #[test]
    fn kind_comes_from_the_consuming_op() {
        let program = program_with(&[&["w"]]);
        let tw = from_tape(
            &decode(
                vec![
                    SourceBinding::EmbeddedHidden,
                    SourceBinding::Weight { id: 0, index: None },
                ],
                vec![norm(1)],
            ),
            &program,
            // these fixtures describe a split MLP and a dense embedding — the shape they
            // asserted before these facts were arguments
            WeightAbi {
                mlp: MlpPacking::Split,
                embed: EmbedStorage::Dense,
            },
        )
        .expect("derives");
        assert_eq!(
            ty_of(&tw, "w").as_deref(),
            Some("crate::__gpu::layers::RmsNorm")
        );

        let tw = from_tape(
            &decode(
                vec![
                    SourceBinding::EmbeddedHidden,
                    SourceBinding::Weight { id: 0, index: None },
                ],
                vec![gemm(1)],
            ),
            &program,
            // these fixtures describe a split MLP and a dense embedding — the shape they
            // asserted before these facts were arguments
            WeightAbi {
                mlp: MlpPacking::Split,
                embed: EmbedStorage::Dense,
            },
        )
        .expect("derives");
        assert_eq!(
            ty_of(&tw, "w").as_deref(),
            Some("crate::__gpu::layers::LinearLayer")
        );
    }

    /// The emitted Rust type of accessor `name`, spaces stripped.
    ///
    /// Note the leading `&` is NOT here: `weight_kind_rust_type` yields the ACCESSOR type `&T`,
    /// and `field_type` strips the borrow to get the type the struct FIELD holds.
    ///
    /// Read from the accessor: `rust_type` is a 1:1 encoding of the field's `WeightKind`.
    fn ty_of(tw: &TapeWeights, name: &str) -> Option<String> {
        tw.accessors
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.rust_type.to_string().replace(' ', ""))
    }

    /// ⛔ AN UNTAUGHT WEIGHT-BINDING OP MUST REFUSE, NOT DEFAULT. A silent
    /// "assume Linear" emits a `Weights` field of the wrong TYPE and fails far
    /// away at load, which is exactly the class of failure this walk replaced.
    #[test]
    fn unknown_weight_binding_op_refuses() {
        let program = program_with(&[&["w"]]);
        let err = from_tape(
            &decode(
                vec![
                    SourceBinding::EmbeddedHidden,
                    SourceBinding::Weight { id: 0, index: None },
                ],
                vec![OpDesc {
                    // `Mul` binds no weight in any real tape; pointing one at a
                    // weight external is the stand-in for a NEW opcode nobody
                    // has taught this walk yet.
                    op: SubOp::Elementwise(EwKind::Mul),
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                }],
            ),
            &program,
            // these fixtures describe a split MLP and a dense embedding — the shape they
            // asserted before these facts were arguments
            WeightAbi {
                mlp: MlpPacking::Split,
                embed: EmbedStorage::Dense,
            },
        );
        let err = match err {
            Err(e) => e,
            Ok(_) => panic!("must refuse an op it has no kind for, but it derived one"),
        };
        assert!(
            err.contains("no kind for it"),
            "the refusal must name the cause, got: {err}"
        );
    }
}
