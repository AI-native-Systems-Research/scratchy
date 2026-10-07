// SPDX-License-Identifier: Apache-2.0
//! ONE CANONICAL, LOWERED — the entry the `#[forward]` macro calls for each decode canonical.
//!
//! Metal's orchestration of the shared passes: the tape and its re-roll ([`tape_program`]), the
//! fold pass under `op_abi::METAL_FUSIONS`, the colourer under `op_abi::METAL_COLOUR_FACTS`. Then
//! metal's step records ([`steps_from_tape`]), laid out along the ROLLED tape and proven against
//! the unrolled layout; where the roll does not prove, the cut is moved until it does. Out come
//! the step tape the bake lowers — each row's weights named by the model's [`SourceManifest`],
//! which the macro emits the model's resolver from — and the arena the bucket statics are sized
//! from.
//!
//! [`tape_program`]: crate::tape_program
//! [`steps_from_tape`]: crate::steps_from_tape

use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};
use scratchy_subtile::handoff::{LoweredDecode, SlotMap, WeightKind, WeightSlot};
use scratchy_subtile::kv_codec::{KvCodecError, expand_kv_codec};
use scratchy_subtile::sample_rows::{SampleRowsError, expand_sample_rows};
use scratchy_subtile::tape_colouring::{Colour, ColourCount, ColourError, colour_tape};
use scratchy_subtile::tape_folding::{FoldError, ModelFoldFacts, fold_tape};
use scratchy_subtile::wave_schedule::wave_order;
use scratchy_target_metal::from_tape::{TapeItem, roll_at};
use scratchy_target_metal::op_abi::{
    METAL_COLOUR_FACTS, METAL_FUSIONS, METAL_KV_CODEC, METAL_SAMPLE_ROWS, METAL_WAVE_ORDER_ROWS,
    metal_colour_rule,
};
use scratchy_target_metal::tape::ids::SourceIx;
use scratchy_target_metal::tape::model_consts::MetalModelConsts;
use scratchy_target_metal::tape::step::{
    AffineBits, AffineGroupSize, HiddenSize, IntermediateSize, LayerId, MetalStepTape,
    RotaryTables, RowSource,
};

use crate::steps_from_tape::{
    Assembled, MoeBits, OpColours, Recording, StepRefusal, UnrolledSteps, assemble, proves,
};

/// A model's SOURCE MANIFEST: every weight family its tapes bind — a `(kind, accessor)` bundle on
/// the model's `Weights`, every layer — in first-seen order. A family's position is the
/// [`SourceIx`] its rows' `Binding::Source`s carry; [`Self::resolver`] is the model's resolver.
#[derive(Default)]
pub struct SourceManifest {
    families: Vec<WeightSlot>,
}

/// A weight kind metal binds no bundle of (the cuda-only quantized linears and MoEs).
#[derive(Debug)]
pub struct UnboundKind {
    pub source: String,
    pub kind: WeightKind,
}

impl SourceManifest {
    /// `family`'s index, interning it on first sight.
    pub fn intern(&mut self, family: &WeightSlot) -> SourceIx {
        let at = self.families.iter().position(|f| f == family);
        let at = at.unwrap_or_else(|| {
            self.families.push(family.clone());
            self.families.len() - 1
        });
        SourceIx(at as u32)
    }

    fn row(&mut self, site: &[WeightSlot]) -> Vec<RowSource> {
        let source = |s: &WeightSlot| RowSource {
            kind: s.kind.clone(),
            ix: self.intern(s),
        };
        site.iter().map(source).collect()
    }

    /// The rotary table each attention class re-ropes cached K with: the global `rotary`, and
    /// for sliding layers `rotary_local` when the model declares one.
    pub fn rotary_tables(&mut self, local: bool) -> RotaryTables {
        let mut table = |base: &str| {
            let (kind, base) = (WeightKind::CosSin, base.to_string());
            self.intern(&WeightSlot { kind, base })
        };
        let global = table("rotary");
        let sliding = if local { table("rotary_local") } else { global };
        RotaryTables { global, sliding }
    }

    /// The model's `ModelSources` impl: one arm per family, `ix => bundle at layer`.
    pub fn resolver(&self) -> Result<TokenStream, UnboundKind> {
        let r = quote!(::scratchy_target_metal::tape::lowered::SourceRef);
        let ids = quote!(::scratchy_target_metal::tape::ids);
        let arms = (0u32..).zip(&self.families).map(|(ix, f)| {
            let (ix, base) = (Literal::u32_unsuffixed(ix), format_ident!("{}", f.base));
            use WeightKind as K;
            let bundle = match f.kind {
                K::CosSin => return Ok(quote!(#ix => Some(#r::CosSin(self.#base.cos_sin_cache)),)),
                K::Embedding => quote!(Embedding),
                K::AffineQuantEmbedding => quote!(AffineQuantEmbedding),
                K::RmsNorm => quote!(RmsNorm),
                K::LayerNorm => quote!(LayerNorm),
                K::Linear => quote!(Linear),
                K::FusedMoe => quote!(FusedMoe),
                K::SharedFusedMoe => quote!(SharedFusedMoe),
                K::GemmaRouter => quote!(GemmaRouter),
                K::GemmaSwitchGlu => quote!(GemmaSwitchGlu),
                K::GatedDeltaNet => quote!(GatedDeltaNet),
                K::Marlin
                | K::Bnb4
                | K::Fp8
                | K::DeepSeekMoe
                | K::DeepSeekMoeFp8
                | K::DeepSeekMoeGgml => {
                    let (source, kind) = (f.base.clone(), f.kind.clone());
                    return Err(UnboundKind { source, kind });
                }
            };
            Ok(quote!(#ix => Some(#r::#bundle(self.#base(layer.get()))),))
        });
        let arms = arms.collect::<Result<Vec<_>, _>>()?;
        let names = self.families.iter().map(|f| f.base.as_str());
        let layered = self.families.iter().any(|f| f.kind != WeightKind::CosSin);
        let layer = if layered {
            quote!(layer)
        } else {
            quote!(_layer)
        };
        // A model every canonical of which metal refused (dense MoE) binds no source at all.
        let (ix, body) = if arms.is_empty() {
            (quote!(_ix), quote!(None))
        } else {
            let body = quote! {
                match ix.get() {
                    #(#arms)*
                    _ => None,
                }
            };
            (quote!(ix), body)
        };
        Ok(quote! {
            impl ::scratchy_target_metal::tape::lowered::ModelSources for Weights {
                const SOURCES: &'static [&'static str] = &[#(#names),*];
                fn source(&self, #ix: #ids::SourceIx, #layer: #ids::LayerId) -> Option<#r<'_>> {
                    #body
                }
            }
        })
    }
}

/// How a preset binds a gated MLP's gate and up projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpForm {
    /// ONE packed `__fused__` buffer, read by the fused gate/up kernel (dense).
    Packed,
    /// Two projections, then the activation-multiply (affine quant).
    Split,
}

/// The per-model facts the step records need beyond the tape — DATA, because the weight table,
/// program and model params they come from are the macro crate's types.
pub struct MetalStepFacts<'a> {
    /// `WeightId` → accessor path.
    pub weight_paths: &'a [Vec<String>],
    /// Accessor bases whose DSL subtree declares its own `shared_expert`: that subtree owns the
    /// shared expert, so a fused MoE's internal shared tail is disabled.
    pub dsl_shared_expert_bases: &'a [String],
    /// The embedding table's accessor base (the embed weight is not a tape source).
    pub embed_base: &'a str,
    /// The embedding's MLX-affine storage; `None` = dense.
    pub embed_quant: Option<(AffineGroupSize, AffineBits)>,
    pub mlp: MlpForm,
    /// A MoE layer's routed-expert projection widths, gate/up/down (`None` = the op's own).
    /// `None` altogether for an unquantized model: its MoE bits are never repacked.
    pub moe_expert_bits: Option<&'a dyn Fn(LayerId) -> [Option<AffineBits>; 3]>,
}

/// Which canonical this is, for the log and the refusals.
pub struct CanonicalAt<'a> {
    pub stem: &'a str,
    pub m: u64,
}

/// Where the backbone's layer loop was cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RollOutcome {
    /// Rolled, with `peel` leading iterations in the prologue.
    Rolled { peel: u32 },
    /// Every layer rolled by its class ([`scratchy_target_metal::from_tape::roll_layer_classes`]).
    LayerClasses,
    /// No cut proved: the unrolled layout.
    Unrolled,
}

/// One decode canonical, lowered.
pub struct MetalCanonical {
    pub steps: MetalStepTape,
    pub colours: ColourCount,
    /// The colour holding the forward's result.
    pub result: Colour,
    /// `(tile, output)` → colour, what the arena statics are sized from.
    pub arena: SlotMap,
    pub roll: RollOutcome,
}

/// Why a canonical could not be lowered. The macro panics with it: a declared pilot has no
/// fallback.
#[derive(Debug)]
pub enum CanonicalRefusal {
    KvCodec(KvCodecError),
    SampleRows(SampleRowsError),
    Fold(FoldError),
    Colour(ColourError),
    Steps(StepRefusal),
}

impl std::fmt::Display for CanonicalRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KvCodec(e) => write!(f, "the shared KV codec pass refused: {e}"),
            Self::SampleRows(e) => write!(f, "the shared sample-rows pass refused: {e}"),
            Self::Fold(e) => write!(f, "the shared fold pass refused: {e}"),
            Self::Colour(e) => write!(f, "tape colorer refused a DECLARED pilot: {e}"),
            Self::Steps(e) => write!(f, "the step records refused a DECLARED pilot: {e}"),
        }
    }
}

/// The barriers a decode step of `m` sequences runs on the unrolled tape `t` (flags `flags`): the
/// fenced rows whose runtime gate lets them run on it.
fn decode_barriers(t: &Assembled, flags: &[bool], m: u64) -> usize {
    use scratchy_target_metal::interpreter::metal::worker::{StepFacts, gate_matches};
    use scratchy_target_metal::tape::step::StepRow;
    let rows = m as u32;
    let decode = StepFacts {
        num_tokens: rows,
        num_seqs: rows,
        has_spec_tokens: false,
        unrotated_blocks: false,
    };
    let runs = |r: &StepRow| matches!(r, StepRow::Step(_, g) if gate_matches(*g, decode));
    (t.rows.iter().zip(flags))
        .filter(|&(r, &f)| f && runs(r))
        .count()
}

/// Lower one decode canonical: step tape (its weights interned into the model's `sources`),
/// colours and arena.
pub fn lower_canonical(
    l: &LoweredDecode,
    facts: &MetalStepFacts<'_>,
    consts: &MetalModelConsts,
    at: CanonicalAt<'_>,
    sources: &mut SourceManifest,
) -> Result<MetalCanonical, CanonicalRefusal> {
    use CanonicalRefusal::Steps;
    let CanonicalAt { stem, m } = at;
    // Metal's TurboQuant: the codec steps its declared facts insert.
    // Only on a model built with it (`MetalModelConsts::kv_codec`): a dense model runs none.
    let coded;
    let l = if consts.kv_codec.is_turboquant() {
        coded = expand_kv_codec(l, &METAL_KV_CODEC).map_err(CanonicalRefusal::KvCodec)?;
        &coded
    } else {
        l
    };
    // Metal's lm_head slice: the sampled rows its declared facts insert.
    let l = &expand_sample_rows(l, &METAL_SAMPLE_ROWS).map_err(CanonicalRefusal::SampleRows)?;
    // Whether the gate/up projections fold is this model's fact: a dense preset has the fused
    // projection kernel, and an affine one the fused one-row matvec, for its one-row bucket.
    let fold_projections = facts.mlp == MlpForm::Packed || m == 1;
    // The one-row bucket's affine matvecs normalize their input and add into the residual.
    let model = ModelFoldFacts {
        fold_projections,
        matvec_ends: m == 1,
        row_programs: m == 1,
    };
    // Metal's barriers drain everything in flight: independent branches run between the same ones.
    // A fused command reads what its fold absorbed and writes its epilogues, so the folds the tape
    // order makes keep every absorbed step ahead of the step it folds into, and every epilogue
    // with it: they fold the same again.
    let waved;
    let l = if m <= METAL_WAVE_ORDER_ROWS {
        let tp = crate::tape_program::tape_program(l, stem, m);
        let folds = fold_tape(&tp.graph, &tp.tape, l, &METAL_FUSIONS, model)
            .map_err(CanonicalRefusal::Fold)?;
        // What runs no command of its own: a step a fold computes inside another, and the expert
        // sort of a bake that gathers (its readers bind what it views).
        let mut free = vec![false; l.input.ops.len()];
        for &(a, _) in folds.absorbed_ops() {
            free[a] = true;
        }
        for (j, od) in l.input.ops.iter().enumerate() {
            if let scratchy_subtile::subtile_ir::SubOp::ExpertSort { k, .. } = od.op {
                free[j] |= od.m.saturating_mul(k.get())
                    < scratchy_target_metal::op_abi::METAL_SORTED_PAIRS;
            }
        }
        let (absorbed, epilogues) = (folds.absorbed_ops(), folds.epilogue_ops());
        waved = wave_order(l, metal_colour_rule, absorbed, epilogues, &free);
        &waved
    } else {
        l
    };
    let tp = crate::tape_program::tape_program(l, stem, m);
    let folds =
        fold_tape(&tp.graph, &tp.tape, l, &METAL_FUSIONS, model).map_err(CanonicalRefusal::Fold)?;
    let colouring = colour_tape(
        &tp.graph,
        &tp.tape,
        &l.bindings,
        &folds.fold_facts(),
        &METAL_COLOUR_FACTS,
    )
    .map_err(CanonicalRefusal::Colour)?;
    let steps = UnrolledSteps::read(l, &tp.unrolled).map_err(Steps)?;
    let colours = OpColours::project(l, &steps, &colouring).map_err(Steps)?;
    let records = Recording {
        l,
        graph: &tp.graph,
        steps: &steps,
        colours: &colours,
        folds: &folds,
        facts,
        hidden: HiddenSize(consts.hidden_size as u32),
        intermediate: IntermediateSize(consts.intermediate_size as u32),
    }
    .records()
    .map_err(Steps)?;
    let rolled = assemble(&tp.rolled, &records, MoeBits::Repacked).map_err(Steps)?;
    let unrolled = assemble(&tp.unrolled, &records, MoeBits::Repacked).map_err(Steps)?;
    // ⛔ THE LM_HEAD'S FLAGS ARE THE ROLLED WALK'S, whichever backbone layout is kept.
    let (rolled_flags, lm_head_barriers) = rolled.flags();
    let (unrolled_flags, _) = unrolled.flags();
    let barriers = decode_barriers(&unrolled, &unrolled_flags, m);
    let proof = |b: &Assembled, flags: &[bool]| proves(b, flags, &unrolled, &unrolled_flags);

    // ⭐ THE ROLL IS KEPT ONLY IF IT IS PROVABLY THE SAME PROGRAM, AND THE CUT IS MOVED UNTIL IT
    // IS. A body drawn from layer 0 lowers layer 0's unfused norm for every layer (metal folds a
    // layer's residual add into the NEXT layer's norm), so leading iterations are peeled into the
    // prologue. ⛔ NOT AN ARCH LIST: the proof decides, per model, per bucket — and every failing
    // cut is reported.
    let mut why: Vec<String> = Vec::new();
    let candidate = |items: &[TapeItem]| {
        let b = assemble(items, &records, MoeBits::Raw).map_err(|e| e.to_string())?;
        let (flags, _) = b.flags();
        proof(&b, &flags).map(|()| (b, flags))
    };
    let mut cut = match proof(&rolled, &rolled_flags) {
        Ok(()) => Some((
            RollOutcome::Rolled { peel: 0 },
            rolled.rows,
            rolled_flags,
            rolled.sites,
        )),
        Err(e) => {
            why.push(format!("peel=0: {e}"));
            None
        }
    };
    // ⭐ EVERY LAYER BY ITS CLASS — kept when it proves and is shorter than the shared tape's own
    // roll. Consecutive layers of one class are the same program, so they loop: gemma-3's
    // six-layer `SSSSSG` cell becomes one sliding body plus one global, and a mixed-precision
    // model — whose IDENTICAL cells are too few for the whole-cell cut to find — still rolls every
    // run of layers that match.
    if let Some(items) = &tp.layer_rolled {
        match candidate(items) {
            Ok((b, flags)) if cut.as_ref().is_none_or(|c| b.rows.len() < c.1.len()) => {
                cut = Some((RollOutcome::LayerClasses, b.rows, flags, b.sites));
            }
            Ok(_) => {}
            Err(e) => why.push(format!("layer-class: {e}")),
        }
    }
    // Whole cells, with leading iterations peeled — what gemma-4-31b needs: its arena is not
    // periodic at one layer, so the per-layer cut above does not prove there.
    if cut.is_none() {
        for peel in 1..4u32 {
            let Some(items) = roll_at(&tp.unrolled, &tp.rolled, peel) else {
                continue;
            };
            match candidate(&items) {
                Ok((b, flags)) => {
                    cut = Some((RollOutcome::Rolled { peel }, b.rows, flags, b.sites));
                    break;
                }
                Err(e) => why.push(format!("peel={peel}: {e}")),
            }
        }
    }
    let from = unrolled.rows.len();
    let (roll, backbone, backbone_barriers, backbone_sites) = cut.unwrap_or((
        RollOutcome::Unrolled,
        unrolled.rows,
        unrolled_flags,
        unrolled.sites,
    ));
    match roll {
        RollOutcome::Unrolled => eprintln!(
            "[m2-roll] {stem} m={m}: NO cut rolls — emitting the un-rolled tape. {}",
            why.join(" | ")
        ),
        roll => eprintln!(
            "[m2-roll] {stem} m={m}: rolled {roll:?} ({} rows from {from})",
            backbone.len()
        ),
    }
    let (colour_count, result) = (colours.count(), colours.result());
    eprintln!(
        "[m2-flip] {stem} m={m}: TAPE-SCHEDULED stream ACTIVE ({} instr, slots={} final={}, \
         {} barriers a decode step)",
        backbone.len(),
        colour_count.get(),
        result.index(),
        barriers,
    );
    Ok(MetalCanonical {
        steps: MetalStepTape {
            backbone,
            backbone_barriers,
            backbone_sources: backbone_sites.iter().map(|s| sources.row(s)).collect(),
            lm_head: rolled.lm_rows,
            lm_head_barriers,
            lm_head_sources: rolled.lm_sites.iter().map(|s| sources.row(s)).collect(),
        },
        colours: colour_count,
        result,
        arena: colours.arena().clone(),
        roll,
    })
}
