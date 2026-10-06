// SPDX-License-Identifier: Apache-2.0
//! METAL'S STEP RECORDS, BUILT FROM THE SHARED TAPE.
//!
//! One record per SOURCE OP, read off the UNROLLED tape: a step's operands are the tape's
//! dataflow (the step that wrote the slot, or a graph source), its role is the shared fold pass's
//! (`tape_folding` under `op_abi::METAL_FUSIONS`), and its buffers are the shared colourer's
//! (`tape_colouring` under `op_abi::METAL_COLOUR_FACTS`) projected onto the canonical tiles
//! ([`OpColours`]). The record's command, weight site and hazard signature are metal's opcode
//! lowering: which kernel a step or fold becomes, which slots it binds, which weights it reads.
//!
//! [`assemble`] then lays the records out along a walk of the tape's items — rolled, unrolled or
//! peeled — with the layer loops the shared re-roll found, and [`hazard_flags`] fences the result.
//! [`proves`] is the check that a rolled layout is the unrolled program.

use std::collections::{BTreeMap, HashMap, HashSet};

use scratchy_subtile::handoff::{
    ExpansionId, LoweredDecode, SlotMap, SourceBinding, TileId, WeightKind, WeightSlot,
};
use scratchy_subtile::kv_codec::CodecGuard;
use scratchy_subtile::lower::{GemmWeight, InputRef};
use scratchy_subtile::subtile_ir::{
    AttnMask, EwKind, ExpertBundle, ExpertProj, GainConvention, RopeFormTag, RowScale, SubOp,
    SubtileIR, SubtileId,
};
use scratchy_subtile::subtile_tape::SlotId;
use scratchy_subtile::tape_colouring::{Colour, ColourCount, Residency, TapeColouring};
use scratchy_subtile::tape_folding::{
    FusedShape, Fusion, ROW_PROGRAM_STEPS, StepOperand, StepRole, TapeFolds,
};
use scratchy_target_metal::from_tape::{StepInput, TapeItem};
use scratchy_target_metal::op_abi::{
    METAL_ELIDABLE, METAL_GUARD_GATES, MetalFusion, MoeWrite, metal_colour_rule, moe_write,
    rope_append_weight_site,
};
use scratchy_target_metal::tape::lowered::RuntimeGate;
use scratchy_target_metal::tape::step::{
    self as st, AffineBits as Bits, AffineGroupSize as Gs, ArenaSlotIdx as Slot, HiddenSize as W,
    IntermediateSize as Inter, LayerId, MetalStep, MoeRegion, MoeRows, MoeStep,
    RowsPerToken as Rows, StepRow,
};

use crate::canonical::MetalStepFacts;

/// Why a record could not be built, naming the source op when one is at fault.
#[derive(Debug)]
pub struct StepRefusal {
    op: Option<(usize, &'static str)>,
    why: Refused,
}

/// What about an op (or the canonical) metal's step records cannot express.
#[derive(Debug)]
pub enum Refused {
    NoFirstNorm,
    NoTapeStep,
    NoColour,
    NoTile,
    Uncoloured {
        tile: u32,
        slot: u8,
    },
    Clobbers {
        operand: u8,
        mine: Slot,
        theirs: Slot,
    },
    MissingOperand(u8),
    NotAStep(u8),
    NotASource(u8),
    /// An activation operand bound to a source other than the embedded hidden.
    ForeignActivation(u8),
    PerLayerOperand(u8),
    UnwrittenSlot(u8),
    NoWeight,
    NotAWeight(usize),
    Fp8Gemm,
    KvBias(String),
    KvBiasFromLayer {
        projection: LayerId,
        rope: LayerId,
    },
    BiasUpstream(&'static str),
    /// A fold member the folds kept (a Mean, Sub, Silu, query rotation or KV encode on its own).
    Escaped,
    /// A Mul whose activation no gated fold absorbed.
    NoGatedFold,
    /// Spyre's pre-fused SiluMul; metal folds Silu into Mul itself.
    SplitSiluMul,
    /// A tile-level decomposition (split-K combine, two-phase norm); metal lowers whole ops.
    TileLevel,
    NoFoldedRotate,
    FusionShape,
    OutOfDomain {
        field: &'static str,
        value: u8,
    },
    GainAddSharers {
        tile: u32,
    },
    LoopUnbalanced,
    /// A MoE block missing an op its geometry is read from.
    IncompleteMoeBlock,
    /// An off-arena operand whose producer writes no scratch region.
    NoRegion,
    /// A MoE softmax over a region that holds no scores.
    SoftmaxOver(MoeRegion),
    /// A fused MoE bundle's shared expert, which no DSL subtree declares and metal has no tail for.
    UndeclaredSharedExpert {
        base: String,
    },
    /// Every step of the construct may drop its commands or not run, so its fence can land on
    /// none: no step of it runs always, and no two run under complementary guards.
    GroupNeverRuns(ExpansionId),
    /// A KV codec step whose construct holds no KV writer to read its operands off.
    CodecUnanchored,
    /// A sample-rows step whose construct holds no matmul.
    NoSampledMatmul,
    /// A sampled matmul metal lowers no slice of (only an MLX-affine one).
    SampledNotAffine,
}

impl std::fmt::Display for StepRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.op {
            Some((i, name)) => write!(f, "op {i} ({name}): {:?}", self.why),
            None => write!(f, "{:?}", self.why),
        }
    }
}

fn canonical_refusal(why: Refused) -> StepRefusal {
    StepRefusal { op: None, why }
}

/// The UNROLLED tape by source op: the slot each op's step writes, the node it performs, and its
/// operands resolved to the op that wrote each slot or the source it reads. Every source op has
/// its step.
pub struct UnrolledSteps {
    slot: Vec<SlotId>,
    node: Vec<SubtileId>,
    args: Vec<Vec<InputRef>>,
    op_of: HashMap<SlotId, usize>,
}

impl UnrolledSteps {
    pub fn read(l: &LoweredDecode, unrolled: &[TapeItem]) -> Result<Self, StepRefusal> {
        let n = l.input.ops.len();
        let no = |j: usize, why| StepRefusal {
            op: Some((j, l.input.ops[j].op.name())),
            why,
        };
        let mut at: Vec<Option<(SlotId, SubtileId)>> = vec![None; n];
        let (mut args, mut op_of) = (vec![Vec::new(); n], HashMap::new());
        for item in unrolled {
            let TapeItem::Step(s) = item else { continue };
            let j = s.source_op.0;
            let no = |why| no(j, why);
            let mut a = Vec::with_capacity(s.inputs.len());
            for (k, input) in s.inputs.iter().enumerate() {
                a.push(match input {
                    StepInput::Slot(w) => InputRef::Op(
                        *op_of
                            .get(w)
                            .ok_or_else(|| no(Refused::UnwrittenSlot(k as u8)))?,
                    ),
                    StepInput::Source(t) => InputRef::Ext(t.index()),
                    StepInput::PerLayerSource { .. } => {
                        return Err(no(Refused::PerLayerOperand(k as u8)));
                    }
                });
            }
            op_of.insert(s.writes, j);
            at[j] = Some((s.writes, s.node));
            args[j] = a;
        }
        let at = at
            .into_iter()
            .enumerate()
            .map(|(j, a)| a.ok_or_else(|| no(j, Refused::NoTapeStep)));
        let (slot, node) = at.collect::<Result<Vec<_>, _>>()?.into_iter().unzip();
        Ok(Self {
            slot,
            node,
            args,
            op_of,
        })
    }
}

/// THE COLOUR AUTHORITY: the shared colouring projected onto the canonical `(tile, output)` keys.
///
/// ⛔ RESOLVE EVERY COLOUR THROUGH THIS, NOT BY `SlotId`. Two ops realising one tile share its
/// entry, and the later op's colour stands (the arena sizing reads the same map); the folded
/// `(w + offset)` gain adds are tiles no step writes and take their norm's colour, fill only.
pub struct OpColours {
    arena: SlotMap,
    by_tile: BTreeMap<(u32, u8), u32>,
    count: ColourCount,
    result: Colour,
}

impl OpColours {
    pub fn project(
        l: &LoweredDecode,
        steps: &UnrolledSteps,
        colouring: &TapeColouring,
    ) -> Result<Self, StepRefusal> {
        let no = |i: usize, why| StepRefusal {
            op: Some((i, l.input.ops[i].op.name())),
            why,
        };
        let colour = |i: usize| {
            let c = colouring.colour_of(steps.slot[i]);
            c.map(Colour::index).ok_or_else(|| no(i, Refused::NoColour))
        };
        let mut arena = SlotMap::new();
        for (i, t) in l.op_tiles.iter().enumerate() {
            let Some((tile, out)) = *t else { continue };
            arena.insert_at(TileId(tile), out, colour(i)?);
            if let Some(c2) = colouring.second_colour_of(steps.slot[i]) {
                arena.insert_at(TileId(tile), out + 1, c2.index());
            }
        }
        // ⛔ ASCENDING NORM ORDER, AND SHARERS MUST AGREE. The fill used to walk a `HashMap`, so
        // two norms sharing a gain tile with different colours would have emitted a different
        // stream per process; one order, and a refusal where order would matter.
        let mut gains: Vec<(usize, u32)> = l
            .norm_gain_add_tiles
            .iter()
            .map(|(o, t)| (*o, *t))
            .collect();
        gains.sort_unstable();
        let mut filled: BTreeMap<u32, u32> = BTreeMap::new();
        for (op, tile) in gains {
            let carried = arena
                .entries()
                .any(|((t, s), _)| *t == TileId(tile) && *s == 0);
            if carried && !filled.contains_key(&tile) {
                continue;
            }
            let c = colour(op)?;
            match filled.insert(tile, c) {
                Some(prev) if prev != c => return Err(no(op, Refused::GainAddSharers { tile })),
                _ => arena.insert_at(TileId(tile), 0, c),
            }
        }
        let by_tile = arena.entries().map(|((t, s), c)| ((t.0, *s), *c)).collect();
        Ok(Self {
            arena,
            by_tile,
            count: colouring.count(),
            result: colouring.result(),
        })
    }

    fn at(&self, tile: u32, slot: u8) -> Result<Slot, Refused> {
        let c = self.by_tile.get(&(tile, slot));
        c.map(|c| Slot(*c))
            .ok_or(Refused::Uncoloured { tile, slot })
    }

    /// The colour of op `i`'s output.
    fn of(&self, l: &LoweredDecode, i: usize) -> Result<Slot, Refused> {
        let (tile, out) = l.op_tiles[i].ok_or(Refused::NoTile)?;
        self.at(tile, out)
    }

    /// The colour of a two-output op's second buffer (GateSplit's gate, VisionRope's k).
    fn second(&self, l: &LoweredDecode, i: usize) -> Result<Slot, Refused> {
        let (tile, _) = l.op_tiles[i].ok_or(Refused::NoTile)?;
        self.at(tile, 1)
    }

    pub fn arena(&self) -> &SlotMap {
        &self.arena
    }

    pub fn count(&self) -> ColourCount {
        self.count
    }

    pub fn result(&self) -> Colour {
        self.result
    }
}

/// A KV codec step's writer: its layer, pairing, class, offsets and weight site.
type CodecWriter = (
    LayerId,
    RopeFormTag,
    AttnMask,
    st::KvOffsets,
    Vec<WeightSlot>,
);

/// Which arena buffers and KV layers a row touches — what the barrier walk fences on.
#[derive(Clone, Debug, Default)]
pub struct HazardSig {
    reads: Vec<Slot>,
    writes: Vec<Slot>,
    kv_w: Option<LayerId>,
    kv_r: Option<LayerId>,
    /// The op scratch (`moe_scratch`) — ONE location, shared by every MoE step and GDN.
    op_scratch: Access,
    /// The KV codec's staging buffer — ONE location, written by every stage, read by a coded
    /// attention.
    codec_staging: Access,
    /// The construct the row was expanded from: its rows fence as one.
    group: Option<ExpansionId>,
    /// A view or a loop marker: dispatches nothing, fences nothing, and is not bookkept.
    metadata: bool,
}

/// How a row touches a location, ordered by how much it conflicts with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum Access {
    #[default]
    Untouched,
    Read,
    /// Written (and read).
    Write,
}

/// One source op's record.
#[derive(Clone)]
struct Emission {
    step: MetalStep,
    /// The runtime gate every command of the row runs under.
    gate: Option<RuntimeGate>,
    /// A MoE's step with its routed-expert widths NOT repacked (see [`MoeBits`]).
    raw: Option<MetalStep>,
    sig: HazardSig,
    sites: Vec<WeightSlot>,
    lm_head: bool,
}

/// An expert projection's record parts: its step, its step at its op's uniform width when the
/// quantization declares the layer's widths (see [`MoeBits`]), the arena slots its rows read, and
/// its weight.
struct ExpertRecord {
    step: st::ExpertMatmul,
    raw: Option<st::ExpertMatmul>,
    reads: Vec<Slot>,
    weight: (WeightKind, usize),
}

/// A decode attention running its KV writer's record (`MetalFusion::RopedAttention`): both steps,
/// touching what either touches, the writer's weight sources first.
fn roped_attention(rope: Emission, attention: Emission) -> Emission {
    let (r, a) = (rope.sig, attention.sig);
    let sig = HazardSig {
        reads: [r.reads, a.reads].concat(),
        writes: [r.writes, a.writes].concat(),
        kv_w: r.kv_w.or(a.kv_w),
        kv_r: a.kv_r.or(r.kv_r),
        op_scratch: r.op_scratch.max(a.op_scratch),
        codec_staging: r.codec_staging.max(a.codec_staging),
        group: a.group,
        metadata: false,
    };
    let step = MetalStep::RopedAttention(Box::new(st::RopedAttention {
        writer: rope.step,
        attention: attention.step,
        writer_sources: rope.sites.len(),
    }));
    Emission {
        step,
        gate: attention.gate,
        raw: None,
        sig,
        sites: [rope.sites, attention.sites].concat(),
        lm_head: false,
    }
}

fn em(step: MetalStep, reads: &[Slot], writes: &[Slot], sites: Vec<WeightSlot>) -> Emission {
    let sig = HazardSig {
        reads: reads.to_vec(),
        writes: writes.to_vec(),
        ..HazardSig::default()
    };
    Emission {
        step,
        gate: None,
        raw: None,
        sig,
        sites,
        lm_head: false,
    }
}

/// Every source op's record, plus the head rows the embedded hidden needs.
pub struct StepRecords {
    head: Vec<Emission>,
    per_op: Vec<Option<Emission>>,
}

/// Everything one canonical's records are built from.
pub struct Recording<'a> {
    pub l: &'a LoweredDecode,
    pub graph: &'a SubtileIR,
    pub steps: &'a UnrolledSteps,
    pub colours: &'a OpColours,
    pub folds: &'a TapeFolds<MetalFusion>,
    pub facts: &'a MetalStepFacts<'a>,
    pub hidden: W,
    pub intermediate: Inter,
}

impl Recording<'_> {
    pub fn records(&self) -> Result<StepRecords, StepRefusal> {
        self.alias_invariant()?;
        // A construct fences as one, on its first row: some row of it must always run — one never
        // elided nor guarded, or one of two under complementary guards.
        let mut groups: BTreeMap<ExpansionId, Vec<Option<CodecGuard>>> = BTreeMap::new();
        for (i, e) in self.l.op_expansion.iter().enumerate() {
            if let Some(e) = e {
                let guards = groups.entry(e.id).or_default();
                if !METAL_ELIDABLE.contains(&self.op(i).kind()) {
                    guards.push(e.guard);
                }
            }
        }
        let runs = |g: &[Option<CodecGuard>]| {
            let covered = |n: CodecGuard| g.contains(&Some(n));
            g.iter()
                .any(|x| x.is_none_or(|x| x.negation().is_some_and(covered)))
        };
        if let Some((g, _)) = groups.iter().find(|(_, guards)| !runs(guards)) {
            return Err(canonical_refusal(Refused::GroupNeverRuns(*g)));
        }
        let (head, skip) = self.head()?;
        let per_op = (0..self.l.input.ops.len())
            .map(|i| self.record(i, skip))
            .collect::<Result<_, _>>()?;
        Ok(StepRecords { head, per_op })
    }

    /// The op source op `i`'s step performs: its tape node's.
    fn op(&self, i: usize) -> &SubOp {
        &self.graph.nodes[self.steps.node[i].index()].op
    }

    /// The width of the rows source op `i` writes: its output tensor's columns.
    fn width(&self, i: usize) -> st::ActivationWidth {
        st::ActivationWidth::of_cols(self.graph.shape(self.graph.op_output[i]).cols)
    }

    fn no(&self, i: usize, why: Refused) -> StepRefusal {
        StepRefusal {
            op: Some((i, self.op(i).name())),
            why,
        }
    }

    fn arg(&self, i: usize, k: u8) -> Result<InputRef, StepRefusal> {
        let a = self.steps.args[i].get(k as usize).copied();
        a.ok_or_else(|| self.no(i, Refused::MissingOperand(k)))
    }

    /// Operand `k` of `i`, which must be a step's output: the producing source op.
    fn step_arg(&self, i: usize, k: u8) -> Result<usize, StepRefusal> {
        match self.arg(i, k)? {
            InputRef::Op(a) => Ok(a),
            InputRef::Ext(_) => Err(self.no(i, Refused::NotAStep(k))),
        }
    }

    /// Operand `k` of `i`, which must be a graph source.
    fn source_arg(&self, i: usize, k: u8) -> Result<usize, StepRefusal> {
        match self.arg(i, k)? {
            InputRef::Ext(e) => Ok(e),
            InputRef::Op(_) => Err(self.no(i, Refused::NotASource(k))),
        }
    }

    fn colour(&self, i: usize) -> Result<Slot, StepRefusal> {
        self.colours.of(self.l, i).map_err(|w| self.no(i, w))
    }

    fn second(&self, i: usize) -> Result<Slot, StepRefusal> {
        self.colours.second(self.l, i).map_err(|w| self.no(i, w))
    }

    /// Where `i`'s gain `(w + offset)` add lives, when its gain routes through one.
    fn gain_add(&self, i: usize) -> Result<Option<Slot>, StepRefusal> {
        let tile = self.l.norm_gain_add_tiles.get(&i);
        tile.map(|t| self.colours.at(*t, 0).map_err(|w| self.no(i, w)))
            .transpose()
    }

    /// The buffer activation operand `k` of `i` reads: its producer's colour, or colour 0 for
    /// the embedded hidden (N12: no other source is an activation).
    fn read(&self, i: usize, k: u8) -> Result<Slot, StepRefusal> {
        match self.arg(i, k)? {
            InputRef::Op(a) => self.colour(a),
            InputRef::Ext(e) => self.embedded(i, k, e),
        }
    }

    fn embedded(&self, i: usize, k: u8, e: usize) -> Result<Slot, StepRefusal> {
        match self.l.bindings[e] {
            SourceBinding::EmbeddedHidden => Ok(Slot(0)),
            _ => Err(self.no(i, Refused::ForeignActivation(k))),
        }
    }

    /// The operand a fold names, read as an activation.
    fn read_operand(&self, i: usize, o: StepOperand) -> Result<Slot, StepRefusal> {
        self.read(self.op_at(i, o.step)?, o.operand.0)
    }

    /// The source a fold's weight operand names.
    fn source_operand(&self, i: usize, o: StepOperand) -> Result<usize, StepRefusal> {
        self.source_arg(self.op_at(i, o.step)?, o.operand.0)
    }

    /// The source op whose step writes `slot`.
    fn op_at(&self, i: usize, slot: SlotId) -> Result<usize, StepRefusal> {
        let j = self.steps.op_of.get(&slot).copied();
        j.ok_or_else(|| self.no(i, Refused::NoTapeStep))
    }

    /// `i`'s first operand bound to a weight.
    fn weight(&self, i: usize) -> Option<usize> {
        self.steps.args[i].iter().find_map(|r| match r {
            InputRef::Ext(e)
                if matches!(
                    self.l.bindings[*e],
                    SourceBinding::Weight { .. } | SourceBinding::WeightScale { .. }
                ) =>
            {
                Some(*e)
            }
            _ => None,
        })
    }

    fn weight_of(&self, i: usize) -> Result<usize, StepRefusal> {
        self.weight(i).ok_or_else(|| self.no(i, Refused::NoWeight))
    }

    /// A weight's layer: its unroll index, else 0.
    fn layer(&self, e: usize) -> LayerId {
        match &self.l.bindings[e] {
            SourceBinding::Weight { index, .. } | SourceBinding::WeightScale { index, .. } => {
                LayerId(index.map_or(0, |u| u.0 as u32))
            }
            _ => LayerId(0),
        }
    }

    /// The layer of `i`'s weight, or 0 when it has none.
    fn weight_layer(&self, i: usize) -> LayerId {
        self.weight(i).map_or(LayerId(0), |e| self.layer(e))
    }

    /// A weight's layer honouring the trailing-numeral path rule: the unroll index when present,
    /// else a trailing numeral path segment (`merger_mlp_2` → 2).
    fn layer_with_path(&self, e: usize) -> LayerId {
        match &self.l.bindings[e] {
            SourceBinding::Weight { id, index } | SourceBinding::WeightScale { id, index } => {
                if let Some(u) = index {
                    return LayerId(u.0 as u32);
                }
                let joined = self.facts.weight_paths[*id as usize].join("_");
                let tail = joined.rfind('_').map(|p| &joined[p + 1..]);
                LayerId(tail.and_then(|t| t.parse().ok()).unwrap_or(0))
            }
            _ => LayerId(0),
        }
    }

    /// A weight's accessor base: its path joined with `_`, less a trailing `_<digits>`.
    ///
    /// 🛑 KNOWN DEFECT, measured 2026-08-22: dropping the numeral is why Qwen2-VL's merger applies
    /// `visual.merger.mlp.2` TWICE and never loads `visual.merger.mlp.0` — `merger.mlp_0` and
    /// `merger.mlp_2` both truncate to base `merger_mlp`, the accessor group collapses to ONE
    /// Unindexed field, and the emitted accessor ignores its layer argument. The fix is NOT to keep
    /// the numeral here: the layer is appended again downstream, which yields `merger_mlp_0_0` /
    /// `merger_mlp_2_2` and still resolves no safetensors key. It belongs at the accessor-NAME
    /// composition site, which must carry (base, layer) as a pair instead of a string.
    fn base(&self, i: usize, e: usize) -> Result<String, StepRefusal> {
        match &self.l.bindings[e] {
            SourceBinding::Weight { id, .. } | SourceBinding::WeightScale { id, .. } => {
                let mut joined = self.facts.weight_paths[*id as usize].join("_");
                if let Some(pos) = joined.rfind('_')
                    && joined[pos + 1..].parse::<u32>().is_ok()
                {
                    joined.truncate(pos);
                }
                Ok(joined)
            }
            _ => Err(self.no(i, Refused::NotAWeight(e))),
        }
    }

    fn site(&self, i: usize, kind: WeightKind, e: usize) -> Result<Vec<WeightSlot>, StepRefusal> {
        Ok(vec![WeightSlot {
            kind,
            base: self.base(i, e)?,
        }])
    }

    /// The rotary table a rope reads: the sliding class's when any operand binds a local cos/sin.
    fn rotary(&self, i: usize) -> WeightSlot {
        let local = self.steps.args[i].iter().any(|r| match r {
            InputRef::Ext(e) => matches!(
                self.l.bindings[*e],
                SourceBinding::Cos { local: true } | SourceBinding::Sin { local: true }
            ),
            InputRef::Op(_) => false,
        });
        let base = if local { "rotary_local" } else { "rotary" };
        WeightSlot {
            kind: WeightKind::CosSin,
            base: base.to_string(),
        }
    }

    /// The additive offset the KV operand produced by `op` carries into the cache of `layer`,
    /// and — for a bias — the producing projection's accessor, which the writer's weight site
    /// must hold (`op_abi::rope_append_bias_slots`).
    fn kv_offset(
        &self,
        i: usize,
        op: usize,
        layer: LayerId,
    ) -> Result<(st::KvOffset, Option<WeightSlot>), StepRefusal> {
        let bias = scratchy_subtile::lower::kv_operand_bias(&self.l.input.ops, op);
        let Some((gemm, weight)) = bias.map_err(|e| self.no(i, Refused::KvBias(e)))? else {
            return Ok((st::KvOffset::Centered, None));
        };
        let e = self.weight_of(gemm)?;
        let projection = self.layer_with_path(e);
        if projection != layer {
            return Err(self.no(
                i,
                Refused::KvBiasFromLayer {
                    projection,
                    rope: layer,
                },
            ));
        }
        let storage = match weight {
            GemmWeight::Affine { .. } => st::BiasStorage::Affine,
            GemmWeight::Dense | GemmWeight::Fp8Dynamic => st::BiasStorage::Dense,
        };
        let linear = WeightSlot {
            kind: WeightKind::Linear,
            base: self.base(gemm, e)?,
        };
        Ok((st::KvOffset::LinearBias(storage), Some(linear)))
    }

    /// The ops of the construct op `i` was expanded from.
    fn group(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let g = self.l.op_expansion[i].map(|e| e.id);
        let of = move |j: &usize| g.is_some() && self.l.op_expansion[*j].map(|e| e.id) == g;
        (0..self.l.input.ops.len()).filter(of)
    }

    /// The geometry of the MoE block op `i` is a step of, read off the block's own ops.
    fn block(&self, i: usize) -> Result<st::MoeBlock, StepRefusal> {
        let (mut experts, mut k, mut inter, mut hidden) = (None, None, None, None);
        let (mut router, mut bundle, mut input) = (None, None, st::RouterInput::Raw);
        let (mut group_size, mut gate, mut up, mut down) = (None, None, None, None);
        for j in self.group(i) {
            match *self.op(j) {
                SubOp::RouterNorm { .. } => input = st::RouterInput::PreNormed,
                SubOp::RouterLogits {
                    experts: e,
                    router: r,
                } => (experts, router) = (Some(st::NumExperts(e.get())), Some(r)),
                SubOp::RouteTopK { k: t } => k = Some(st::TopK(t.get())),
                SubOp::ExpertMatmul {
                    proj,
                    n,
                    quant: q,
                    bundle: b,
                    ..
                } => {
                    let width = Some(self.expert_matmul(j)?.step.width.bits());
                    match proj {
                        ExpertProj::Gate => {
                            (inter, bundle) = (Some(Inter(n)), Some(b));
                            (group_size, gate) = (Some(Gs(q.group().get())), width);
                        }
                        ExpertProj::Up => up = width,
                        ExpertProj::Down => down = width,
                    }
                }
                SubOp::ExpertCombine { hidden: h, .. } => hidden = Some(W(h)),
                _ => {}
            }
        }
        let no = || self.no(i, Refused::IncompleteMoeBlock);
        Ok(st::MoeBlock {
            experts: experts.ok_or_else(no)?,
            top_k: k.ok_or_else(no)?,
            inter: inter.ok_or_else(no)?,
            hidden: hidden.ok_or_else(no)?,
            router: router.ok_or_else(no)?,
            bundle: bundle.ok_or_else(no)?,
            input,
            quant: st::ExpertQuant {
                group_size: group_size.ok_or_else(no)?,
                widths: st::ExpertWidths {
                    gate: gate.ok_or_else(no)?,
                    up: up.ok_or_else(no)?,
                    down: down.ok_or_else(no)?,
                },
            },
        })
    }

    /// A KV codec step's writer — the KV writer of `encode`'s construct, `encode` being the
    /// step's `KvEncode` — as the writer's own record states it. Its class is its attention's.
    fn codec_writer(&self, i: usize, encode: usize) -> Result<CodecWriter, StepRefusal> {
        let rope = |j: &usize| matches!(self.op(*j), SubOp::RopeAppend { .. });
        let no = || self.no(i, Refused::CodecUnanchored);
        let w = self.group(encode).find(rope).ok_or_else(no)?;
        // The writer's command — its last fold's — whether it runs it or the attention does.
        let f = self
            .folds
            .driven(self.steps.slot[w])
            .last()
            .ok_or_else(no)?;
        let e = self.fused(w, f)?;
        let (layer, pairing, class, offsets) = match e.step {
            MetalStep::RopeAppend(.., l, p, c, offsets, _) => (l, p, c, offsets),
            // Normed K/V: a norm adds nothing.
            MetalStep::RopeAppendNormed(.., l, p, c, _) => (l, p, c, st::KvOffsets::CENTERED),
            _ => return Err(no()),
        };
        Ok((layer, pairing, class, offsets, e.sites))
    }

    /// The scratch region the off-arena value op `j` writes (`op_abi::moe_write`).
    fn region(&self, j: usize) -> Result<MoeRegion, StepRefusal> {
        match moe_write(self.op(j)) {
            Some(MoeWrite::Region(r)) => Ok(r),
            Some(MoeWrite::OverOperand(k)) => self.region(self.step_arg(j, k.0)?),
            Some(MoeWrite::Arena) | None => Err(self.no(j, Refused::NoRegion)),
        }
    }

    /// Where MoE op `i`'s operand `k` rows are, and the arena slots that reads: the token rows
    /// (through the expert sort, which a gathered bake elides — its readers bind what it views),
    /// or a scratch region.
    fn moe_rows(&self, i: usize, k: u8) -> Result<(MoeRows, Vec<Slot>), StepRefusal> {
        let a = match self.arg(i, k)? {
            InputRef::Op(a) if metal_colour_rule(self.op(a)).residency == Residency::OffArena => a,
            _ => {
                let s = self.read(i, k)?;
                return Ok((MoeRows::Tokens(s), vec![s]));
            }
        };
        match self.region(a)? {
            MoeRegion::SortedRows => self.moe_rows(a, 0),
            r => Ok((MoeRows::Scratch(r), Vec::new())),
        }
    }

    /// Expert projection op `j`'s record parts.
    fn expert_matmul(&self, j: usize) -> Result<ExpertRecord, StepRefusal> {
        let SubOp::ExpertMatmul {
            proj,
            quant,
            bundle,
            ..
        } = *self.op(j)
        else {
            return Err(self.no(j, Refused::FusionShape));
        };
        let ((rows, reads), e) = (self.moe_rows(j, 0)?, self.source_arg(j, 2)?);
        let (layer, group_size) = (self.layer(e), Gs(quant.group().get()));
        let op_bits = Bits(quant.bits().get());
        let at = |width| st::ExpertMatmul {
            rows,
            layer,
            proj,
            group_size,
            width,
        };
        let raw = at(st::ExpertWidth::OpUniform(op_bits));
        // A Qwen-MoE projection's own width, as the quantization declares it per layer.
        let declared = match bundle {
            ExpertBundle::SharedFused => self.facts.moe_expert_bits,
            ExpertBundle::SwitchGlu | ExpertBundle::Fused => None,
        };
        let step = declared.map_or(raw, |w| {
            let widths = w(layer);
            let own = match proj {
                ExpertProj::Gate => widths[0],
                ExpertProj::Up => widths[1],
                ExpertProj::Down => widths[2],
            };
            at(st::ExpertWidth::Declared(own.unwrap_or(op_bits)))
        });
        Ok(ExpertRecord {
            step,
            raw: declared.map(|_| raw),
            reads,
            weight: (bundle.weight_kind(), e),
        })
    }

    /// Single-owner rule: a DSL-declared `<base>.shared_expert` subtree owns the shared expert of
    /// combine op `i`'s block; metal has no fused tail for one it does not.
    fn shared_expert_owned(&self, i: usize) -> Result<(), StepRefusal> {
        let SubOp::ExpertCombine { shared, .. } = *self.op(i) else {
            return Err(self.no(i, Refused::FusionShape));
        };
        if shared.0.is_some() {
            let bank = self
                .group(i)
                .find(|j| matches!(self.op(*j), SubOp::ExpertMatmul { .. }));
            let bank = bank.ok_or_else(|| self.no(i, Refused::IncompleteMoeBlock))?;
            let base = self.base(bank, self.source_arg(bank, 2)?)?;
            if !self.facts.dsl_shared_expert_bases.contains(&base) {
                return Err(self.no(i, Refused::UndeclaredSharedExpert { base }));
            }
        }
        Ok(())
    }

    /// A MoE step's record: the block's step, its arena reads and writes, and its weight.
    fn moe(
        &self,
        i: usize,
        step: MoeStep,
        reads: &[Slot],
        writes: &[Slot],
        weight: Option<(WeightKind, usize)>,
    ) -> Result<Emission, StepRefusal> {
        let site = match weight {
            Some((kind, e)) => self.site(i, kind, e)?,
            None => Vec::new(),
        };
        let mut e = em(MetalStep::Moe(self.block(i)?, step), reads, writes, site);
        e.sig.op_scratch = match moe_write(self.op(i)) {
            Some(MoeWrite::Arena) => Access::Read,
            _ => Access::Write,
        };
        Ok(e)
    }

    /// THE declared-alias invariant, checked once against the real colouring. `metal_colour_rule`
    /// says which operand a kernel writes over; the colourer must have given that operand and the
    /// op ONE buffer, or the kernel clobbers a buffer another op still owns — garbage tokens far
    /// from the cause (the failure that once cost a vocab-sized slot on gemma-class logits).
    fn alias_invariant(&self) -> Result<(), StepRefusal> {
        for i in 0..self.l.input.ops.len() {
            let Some(k) = metal_colour_rule(self.op(i)).in_place_operand() else {
                continue;
            };
            let (Some(InputRef::Op(src)), Ok(mine)) = (
                self.steps.args[i].get(k.index()).copied(),
                self.colours.of(self.l, i),
            ) else {
                continue;
            };
            if let Ok(theirs) = self.colours.of(self.l, src)
                && mine != theirs
            {
                let operand = k.0;
                return Err(self.no(
                    i,
                    Refused::Clobbers {
                        operand,
                        mine,
                        theirs,
                    },
                ));
            }
        }
        Ok(())
    }

    /// The head: the embedded hidden becomes the embed rows (and granite's embed multiplier,
    /// which sits between the embed and the splice). Returns the rows and the op they absorbed.
    fn head(&self) -> Result<(Vec<Emission>, Option<usize>), StepRefusal> {
        let ops = 0..self.l.input.ops.len();
        let norm = |i: usize| matches!(self.op(i), SubOp::RmsNorm { .. });
        if !ops.clone().any(norm) {
            return Err(canonical_refusal(Refused::NoFirstNorm));
        }
        let embedded = SourceBinding::EmbeddedHidden;
        if !self.l.bindings.contains(&embedded) {
            return Ok((Vec::new(), None));
        }
        let zero = Slot(0);
        let base = self.facts.embed_base.to_string();
        let (embed, kind) = match self.facts.embed_quant {
            Some((gs, bits)) => (
                MetalStep::AffineEmbed(zero, gs, bits),
                WeightKind::AffineQuantEmbedding,
            ),
            None => (MetalStep::Embed(zero), WeightKind::Embedding),
        };
        let mut rows = vec![em(embed, &[], &[zero], vec![WeightSlot { kind, base }])];
        let mut skip = None;
        if let Some(i0) = ops.clone().find(|i| !norm(*i))
            && let SubOp::ScalarMul { scale } = *self.op(i0)
            && matches!(self.steps.args[i0].first(), Some(InputRef::Ext(_)))
        {
            skip = Some(i0);
            // An identity multiply emits nothing.
            if scale != 1.0 {
                let out = self.colour(i0)?;
                let step = MetalStep::ScalarMul(zero, out, st::Scale(scale), self.width(i0));
                rows.push(em(step, &[zero], &[out], Vec::new()));
            }
        }
        rows.push(em(
            MetalStep::SpliceMmEmbeds(zero),
            &[zero],
            &[zero],
            Vec::new(),
        ));
        Ok((rows, skip))
    }

    /// Source op `i`'s record: nothing when a fold computes it elsewhere or it is elided.
    fn record(&self, i: usize, head: Option<usize>) -> Result<Option<Emission>, StepRefusal> {
        let mut e = match self.folds.role(self.steps.slot[i]) {
            StepRole::Absorbed { .. } | StepRole::Epilogue { .. } => None,
            _ if head == Some(i) => None,
            StepRole::Drives(f) => Some(self.fused(i, f)?),
            StepRole::Kept => self.kept(i)?,
        };
        if let Some(e) = &mut e {
            e.lm_head = self.lm_head(i);
            if let Some(x) = self.l.op_expansion[i] {
                e.sig.group = Some(x.id);
                e.gate = x.guard.and_then(|g| METAL_GUARD_GATES.gate(g));
            }
        }
        Ok(e)
    }

    /// The lm_head half: the forward's result matmul, and the sampled rows around it (the
    /// result's construct).
    fn lm_head(&self, i: usize) -> bool {
        let r = self.l.input.result;
        let id = |j: usize| self.l.op_expansion[j].map(|e| e.id);
        (i == r && matches!(self.op(i), SubOp::MatmulTile { .. }))
            || (id(r).is_some() && id(i) == id(r))
    }

    /// A fold's command: the kernel the fold names, over the steps its shape names.
    fn fused(&self, i: usize, f: &Fusion<MetalFusion>) -> Result<Emission, StepRefusal> {
        use FusedShape as Sh;
        use MetalFusion as F;
        match (f.kernel, f.shape) {
            (F::RopeAppend, Sh::Sibling { sibling }) => self.rope(i, self.op_at(i, sibling)?),
            (
                F::RopeAppendNormed,
                Sh::NormedRope {
                    q,
                    k,
                    v,
                    q_gain,
                    k_gain,
                },
            ) => {
                let (layer, pairing, class) = self.rope_fields(i)?;
                let [q, k, v] = [
                    self.colour(self.op_at(i, q)?)?,
                    self.colour(self.op_at(i, k)?)?,
                    self.colour(self.op_at(i, v)?)?,
                ];
                let mut sites = Vec::new();
                for g in [q_gain, k_gain].into_iter().flatten() {
                    let e = self.source_operand(i, g)?;
                    sites.push(WeightSlot {
                        kind: WeightKind::RmsNorm,
                        base: self.base(i, e)?,
                    });
                }
                sites.push(self.rotary(i));
                let pool = st::KvWrite::Pool;
                let step =
                    MetalStep::RopeAppendNormed(q, k, v, q, k, v, layer, pairing, class, pool);
                let mut e = em(step, &[q, k, v], &[q, k, v], sites);
                e.sig.kv_w = Some(layer);
                Ok(e)
            }
            (F::MeanSubRmsNorm | F::MeanSubRmsNormBiasAdd, Sh::CentredNorm { sub, bias, .. }) => {
                let out = self.colour(i)?;
                // The Sub's x operand: the pre-centring value (the embedded hidden at layer 0).
                let input = self.read(self.op_at(i, sub)?, 0)?;
                let layer = self.weight_layer(i);
                let e = self.weight_of(i)?;
                match (f.kernel, bias) {
                    // The bias-carrying LayerNorm accessor (vision); the bias rides its bundle, and
                    // the command writes the BiasAdd's buffer.
                    (F::MeanSubRmsNormBiasAdd, Some(b)) => {
                        let out = self.colour(self.op_at(i, b)?)?;
                        let step =
                            MetalStep::MeanSubRmsNormBiasAdd(input, out, layer, self.width(i));
                        Ok(em(
                            step,
                            &[input],
                            &[out],
                            self.site(i, WeightKind::LayerNorm, e)?,
                        ))
                    }
                    // The gain is an `RmsNorm` source like every other norm's.
                    (F::MeanSubRmsNorm, None) => {
                        let step = MetalStep::MeanSubRmsNorm(input, out, layer, self.width(i));
                        Ok(em(
                            step,
                            &[input],
                            &[out],
                            self.site(i, WeightKind::RmsNorm, e)?,
                        ))
                    }
                    _ => Err(self.no(i, Refused::FusionShape)),
                }
            }
            // `residual += delta; delta = rmsnorm(residual)`: the residual accumulates in place at
            // arg1, the normed value lands over arg0.
            (F::FusedAddRmsNorm, Sh::ResidualNorm { add }) => {
                let out = self.colour(i)?;
                let add = self.op_at(i, add)?;
                let x = self.colour(self.step_arg(add, 0)?)?;
                let res = self.read(add, 1)?;
                let layer = self.weight_layer(i);
                let step = match *self.op(i) {
                    SubOp::RmsNorm {
                        gain: g @ GainConvention::OnePlusScale,
                        ..
                    } => MetalStep::FusedAddRmsNormWithOffset(x, res, layer, offset(g)),
                    _ => MetalStep::FusedAddRmsNorm(x, res, layer, self.hidden, Rows(1)),
                };
                let site = self.site(i, WeightKind::RmsNorm, self.weight_of(i)?)?;
                let writes: Vec<Slot> = [res, out].into_iter().chain(self.gain_add(i)?).collect();
                Ok(em(step, &[x, res], &writes, site))
            }
            (
                F::SiluMul | F::GeluMul | F::FusedGateUpSiluMul | F::FusedGateUpGeluMul,
                Sh::Gated { .. },
            ) => self.gated(i, f.kernel),
            // `out = (rmsnorm(delta) + residual) * layer_scalar`.
            (
                F::NormAddScalarMul,
                Sh::NormAddScale {
                    delta,
                    residual,
                    gain,
                    scale,
                    ..
                },
            ) => {
                let out = self.colour(i)?;
                let delta = self.colour(self.op_at(i, delta)?)?;
                let res = self.read_operand(i, residual)?;
                let (gain, scale) = (
                    self.source_operand(i, gain)?,
                    self.source_operand(i, scale)?,
                );
                let step =
                    MetalStep::NormAddScalarMul(delta, res, out, self.layer(scale), self.hidden);
                let norm = |e| self.site(i, WeightKind::RmsNorm, e);
                let sites = [norm(gain)?, norm(scale)?].concat();
                Ok(em(step, &[delta, res], &[res, out], sites))
            }
            // `out = rmsnorm(delta) + residual`, the norm's own epsilon and gain convention.
            (
                F::NormAdd,
                Sh::NormAdd {
                    norm,
                    delta,
                    residual,
                    gain,
                },
            ) => {
                let nrm = self.op_at(i, norm)?;
                let SubOp::RmsNorm { eps, gain: g } = *self.op(nrm) else {
                    return Err(self.no(i, Refused::FusionShape));
                };
                let (out, x, res) = (
                    self.colour(i)?,
                    self.read_operand(i, delta)?,
                    self.read_operand(i, residual)?,
                );
                let e = self.source_operand(i, gain)?;
                let rows = st::RowNorm {
                    layer: self.layer(e),
                    eps: st::Eps(eps),
                    offset: offset(g),
                };
                let step = MetalStep::NormAdd(x, res, out, rows, self.hidden);
                let site = self.site(i, WeightKind::RmsNorm, e)?;
                Ok(em(step, &[x, res], &[out], site))
            }
            // The gate and up projections and `act(gate) * up`: the up reads the gate's rows and
            // layer, at the gate's group size.
            (F::MoeGateUpAct, Sh::ExpertGated { gate, up }) => {
                let SubOp::ExpertGatedAct { act } = *self.op(i) else {
                    return Err(self.no(i, Refused::FusionShape));
                };
                let g = self.expert_matmul(self.op_at(i, gate)?)?;
                let u = self.expert_matmul(self.op_at(i, up)?)?;
                let shared = |x: &st::ExpertMatmul| (x.rows, x.layer, x.group_size);
                if shared(&g.step) != shared(&u.step) {
                    return Err(self.no(i, Refused::FusionShape));
                }
                let (routing, router) = self.routing(i)?;
                let step = MoeStep::GateUpAct(g.step, u.step.width, act, routing);
                let mut e = self.moe(i, step, &g.reads, &[], Some(g.weight))?;
                e.sites.extend(router);
                if g.raw.is_some() || u.raw.is_some() {
                    let (gr, ur) = (g.raw.unwrap_or(g.step), u.raw.unwrap_or(u.step));
                    let raw = MoeStep::GateUpAct(gr, ur.width, act, routing);
                    e.raw = Some(MetalStep::Moe(self.block(i)?, raw));
                }
                Ok(e)
            }
            // The down projection, its unsort and the combine into the combine's buffer. The
            // command writes the down rows into the op scratch before it combines them.
            (F::MoeDownCombine, Sh::ExpertCombined { down, .. }) => {
                self.shared_expert_owned(i)?;
                let d = self.expert_matmul(self.op_at(i, down)?)?;
                let out = self.colour(i)?;
                let step = MoeStep::DownCombine(d.step, out, st::CombineEnds::default());
                let mut e = self.moe(i, step, &d.reads, &[out], Some(d.weight))?;
                e.sig.op_scratch = Access::Write;
                if let Some(raw) = d.raw {
                    let raw = MoeStep::DownCombine(raw, out, st::CombineEnds::default());
                    e.raw = Some(MetalStep::Moe(self.block(i)?, raw));
                }
                Ok(e)
            }
            // A gated command routing its token itself: the expert fold it extends.
            (F::MoeRouted, Sh::Routed { .. }) => {
                let driven = self.folds.driven(self.steps.slot[i]);
                let expert = driven.iter().rev().find(|f| f.kernel != F::MoeRouted);
                self.fused(i, expert.ok_or_else(|| self.no(i, Refused::FusionShape))?)
            }
            // The routing, as the program its folded steps spell.
            (F::MoeRoute, Sh::Route { .. }) => {
                let (program, weight) = self.route_program(i, f.shape)?;
                self.moe(i, MoeStep::Route(program), &[], &[], weight)
            }
            (F::NormedQmv | F::NormedRouter, Sh::NormedMatvec { .. })
            | (F::QmvEpilogue, Sh::MatvecEpilogue { .. }) => {
                let kept = self.kept(i)?;
                kept.ok_or_else(|| self.no(i, Refused::FusionShape))
            }
            (F::RowProgram, Sh::RowProgram { steps }) => self.row_program(i, &steps),
            // The attention's own record, running its writer's — the writer's last fold's.
            (F::RopedAttention, Sh::RopedAttention { writer }) => {
                let w = self.op_at(i, writer)?;
                let shape = || self.no(i, Refused::FusionShape);
                let last = self.folds.driven(writer).last().ok_or_else(shape)?;
                let rope = self.fused(w, last)?;
                let attention = self.kept(i)?.ok_or_else(shape)?;
                Ok(roped_attention(rope, attention))
            }
            // The combine's command — its expert fold's — computing the shared expert's gated rows
            // and the residual add as it stores each row, into the chain's last buffer.
            (
                F::CombineEpilogue,
                Sh::MatvecEpilogue {
                    gate_scale, add, ..
                },
            ) => {
                let shape = || self.no(i, Refused::FusionShape);
                let driven = self.folds.driven(self.steps.slot[i]);
                let base = driven.len().checked_sub(2).map(|b| &driven[b]);
                let mut e = self.fused(i, base.ok_or_else(shape)?)?;
                let mut ends = st::CombineEnds::default();
                let mut reads = Vec::new();
                if let Some((_, [shared, g])) = gate_scale {
                    let pair = (self.read_operand(i, shared)?, self.read_operand(i, g)?);
                    reads.extend([pair.0, pair.1]);
                    ends.gate_scale = Some(pair);
                }
                if let Some((_, residual)) = add {
                    reads.push(self.read_operand(i, residual)?);
                    ends.residual = true;
                }
                let last = add.map(|(a, _)| a).or(gate_scale.map(|(c, _)| c));
                let out = self.colour(self.op_at(i, last.ok_or_else(shape)?)?)?;
                let with = |step: &MetalStep| match step {
                    MetalStep::Moe(b, MoeStep::DownCombine(d, _, _)) => {
                        Ok(MetalStep::Moe(*b, MoeStep::DownCombine(*d, out, ends)))
                    }
                    _ => Err(shape()),
                };
                e.step = with(&e.step)?;
                e.raw = e.raw.as_ref().map(with).transpose()?;
                e.sig.reads.extend(reads);
                e.sig.writes = vec![out];
                Ok(e)
            }
            // The writer's command — its earlier fold's, else its own — writing the packed store too.
            (F::KvEncoded, Sh::Encoded { .. }) => {
                let driven = self.folds.driven(self.steps.slot[i]);
                let mut e = match driven.len().checked_sub(2).map(|b| &driven[b]) {
                    Some(base) => self.fused(i, base)?,
                    None => self
                        .kept(i)?
                        .ok_or_else(|| self.no(i, Refused::FusionShape))?,
                };
                let packed = st::KvWrite::PoolAndPacked;
                e.step = match e.step {
                    MetalStep::RopeAppend(q, k, v, qo, ko, vo, l, p, c, o, _) => {
                        MetalStep::RopeAppend(q, k, v, qo, ko, vo, l, p, c, o, packed)
                    }
                    MetalStep::RopeAppendNormed(q, k, v, qo, ko, vo, l, p, c, _) => {
                        MetalStep::RopeAppendNormed(q, k, v, qo, ko, vo, l, p, c, packed)
                    }
                    _ => return Err(self.no(i, Refused::FusionShape)),
                };
                Ok(e)
            }
            _ => Err(self.no(i, Refused::FusionShape)),
        }
    }

    /// The routing program top-k step `i`'s route fold `route` spells, with the router weight
    /// its per-expert scale reads.
    fn route_program(
        &self,
        i: usize,
        route: FusedShape,
    ) -> Result<(st::RouteProgram, Option<(WeightKind, usize)>), StepRefusal> {
        let FusedShape::Route {
            pre,
            tail: [scale, post, expert_scale],
            ..
        } = route
        else {
            return Err(self.no(i, Refused::FusionShape));
        };
        let op = |s| -> Result<(usize, &SubOp), StepRefusal> {
            let j = self.op_at(i, s)?;
            Ok((j, self.op(j)))
        };
        let scale = match scale.map(op).transpose()? {
            None => None,
            Some((_, &SubOp::RouteScale { scale })) => Some(st::Scale(scale)),
            Some(_) => return Err(self.no(i, Refused::FusionShape)),
        };
        let post = match post.map(op).transpose()? {
            None => st::RoutePost::None,
            Some((_, SubOp::RouteSoftmax)) => st::RoutePost::Softmax,
            Some((_, SubOp::RouteRenorm)) => st::RoutePost::Renorm,
            Some(_) => return Err(self.no(i, Refused::FusionShape)),
        };
        let weight = match expert_scale.map(op).transpose()? {
            None => None,
            Some((j, &SubOp::RouteExpertScale { router })) => {
                Some((router.weight_kind(), self.source_arg(j, 2)?))
            }
            Some(_) => return Err(self.no(i, Refused::FusionShape)),
        };
        let program = st::RouteProgram {
            pre_softmax: pre.is_some(),
            scale,
            post,
            expert_scale: weight.as_ref().map(|&(_, e)| self.layer(e)),
        };
        Ok((program, weight))
    }

    /// The routing expert command `i` computes itself (`MetalFusion::MoeRouted`), if it does, with
    /// the site of the router weight it reads.
    fn routing(
        &self,
        i: usize,
    ) -> Result<(Option<st::RouteProgram>, Vec<WeightSlot>), StepRefusal> {
        let driven = self.folds.driven(self.steps.slot[i]);
        let Some(top_k) = driven.iter().find_map(|f| match f.shape {
            FusedShape::Routed { top_k } => Some(top_k),
            _ => None,
        }) else {
            return Ok((None, Vec::new()));
        };
        let t = self.op_at(i, top_k)?;
        let route = self.folds.driven(top_k).iter().find_map(|f| match f.shape {
            FusedShape::Route { .. } => Some(f.shape),
            _ => None,
        });
        let route = route.ok_or_else(|| self.no(i, Refused::FusionShape))?;
        let (program, weight) = self.route_program(t, route)?;
        let site = match weight {
            Some((kind, e)) => self.site(t, kind, e)?,
            None => Vec::new(),
        };
        Ok((Some(program), site))
    }

    /// A row program over its `steps` (tape order, the driver `i` last): each external row loaded
    /// once, a register per step, and a store of every step read outside the program — the driver,
    /// and the members the fold made its epilogues.
    fn row_program(
        &self,
        i: usize,
        steps: &[Option<SlotId>; ROW_PROGRAM_STEPS],
    ) -> Result<Emission, StepRefusal> {
        use st::RowInstr as R;
        let members = steps.iter().flatten().map(|s| self.op_at(i, *s));
        let members = members.collect::<Result<Vec<usize>, _>>()?;
        let no = || self.no(i, Refused::FusionShape);
        let width = W(self.graph.shape(self.graph.op_output[i]).cols);
        let mut prog = st::RowProgram {
            instrs: [None; st::ROW_PROGRAM_INSTRS],
            inputs: [None; 4],
            outputs: [None; 3],
            gains: [None; 8],
            scalars: [None; 2],
            width,
        };
        let mut instrs: Vec<R> = Vec::new();
        let mut reg_of: HashMap<usize, u8> = HashMap::new();
        let mut loaded: Vec<(Slot, u8)> = Vec::new();
        let (mut next, mut sites) = (0u8, Vec::new());
        let (mut gains, mut scalars) = (0usize, 0usize);
        for &m in &members {
            // Operand `k` of `m` in a register: a member's, else its row loaded once.
            let mut operand =
                |k: u8, instrs: &mut Vec<R>, next: &mut u8| -> Result<u8, StepRefusal> {
                    if let InputRef::Op(a) = self.arg(m, k)?
                        && let Some(&r) = reg_of.get(&a)
                    {
                        return Ok(r);
                    }
                    let slot = self.read(m, k)?;
                    if let Some(&(_, r)) = loaded.iter().find(|(s, _)| *s == slot) {
                        return Ok(r);
                    }
                    let input = u8::try_from(loaded.len()).map_err(|_| no())?;
                    *prog.inputs.get_mut(usize::from(input)).ok_or_else(no)? = Some(slot);
                    instrs.push(R::Load { dst: *next, input });
                    loaded.push((slot, *next));
                    *next += 1;
                    Ok(*next - 1)
                };
            let a = operand(0, &mut instrs, &mut next)?;
            let add = matches!(self.op(m), SubOp::Elementwise(EwKind::Add));
            let b = if add {
                operand(1, &mut instrs, &mut next)?
            } else {
                a
            };
            let dst = next;
            let instr = match *self.op(m) {
                SubOp::Elementwise(EwKind::Add) => R::Add { dst, a, b },
                SubOp::RmsNorm { eps, gain } => {
                    let e = self.weight_of(m)?;
                    *prog.gains.get_mut(gains).ok_or_else(no)? = Some(self.layer(e));
                    sites.extend(self.site(m, WeightKind::RmsNorm, e)?);
                    gains += 1;
                    R::Norm {
                        dst,
                        a,
                        gain: (gains - 1) as u8,
                        eps: st::Eps(eps),
                        offset: offset(gain),
                    }
                }
                SubOp::ScalarWeightMul => {
                    let e = self.weight_of(m)?;
                    *prog.scalars.get_mut(scalars).ok_or_else(no)? = Some(self.layer(e));
                    sites.extend(self.site(m, WeightKind::RmsNorm, e)?);
                    scalars += 1;
                    R::ScaleWeight {
                        dst,
                        a,
                        scalar: (scalars - 1) as u8,
                    }
                }
                SubOp::ScalarMul { scale } => R::Scale {
                    dst,
                    a,
                    scale: st::Scale(scale),
                },
                _ => return Err(no()),
            };
            instrs.push(instr);
            reg_of.insert(m, dst);
            next += 1;
        }
        let mut writes = Vec::new();
        for &m in &members {
            let written = m == i
                || matches!(
                    self.folds.role(self.steps.slot[m]),
                    StepRole::Epilogue { .. }
                );
            if !written {
                continue;
            }
            let output = u8::try_from(writes.len()).map_err(|_| no())?;
            let slot = self.colour(m)?;
            *prog.outputs.get_mut(usize::from(output)).ok_or_else(no)? = Some(slot);
            instrs.push(R::Store {
                output,
                a: reg_of[&m],
            });
            writes.push(slot);
        }
        if instrs.len() > st::ROW_PROGRAM_INSTRS || usize::from(next) > st::ROW_PROGRAM_REGISTERS {
            return Err(no());
        }
        for (k, instr) in instrs.into_iter().enumerate() {
            prog.instrs[k] = Some(instr);
        }
        let reads: Vec<Slot> = loaded.iter().map(|(s, _)| *s).collect();
        Ok(em(
            MetalStep::RowProgram(Box::new(prog)),
            &reads,
            &writes,
            sites,
        ))
    }

    fn rope_fields(&self, i: usize) -> Result<(LayerId, RopeFormTag, AttnMask), StepRefusal> {
        match *self.op(i) {
            SubOp::RopeAppend {
                layer,
                attn,
                pairing,
                ..
            } => Ok((LayerId(layer), pairing, attn)),
            _ => Err(self.no(i, Refused::FusionShape)),
        }
    }

    /// The rope + paged KV write: q from the folded query rotation, k and v its own operands.
    fn rope(&self, i: usize, rotate: usize) -> Result<Emission, StepRefusal> {
        let (layer, pairing, class) = self.rope_fields(i)?;
        let q = self.colour(self.step_arg(rotate, 0)?)?;
        let (k_op, v_op) = (self.step_arg(i, 0)?, self.step_arg(i, 3)?);
        let (k, v) = (self.colour(k_op)?, self.colour(v_op)?);
        let (k_off, k_linear) = self.kv_offset(i, k_op, layer)?;
        let (v_off, v_linear) = self.kv_offset(i, v_op, layer)?;
        let offsets = st::KvOffsets { k: k_off, v: v_off };
        let pool = st::KvWrite::Pool;
        let step = MetalStep::RopeAppend(q, k, v, q, k, v, layer, pairing, class, offsets, pool);
        let sites = rope_append_weight_site(self.rotary(i), k_linear, v_linear);
        let mut e = em(step, &[q, k, v], &[q, k, v], sites);
        e.sig.kv_w = Some(layer);
        Ok(e)
    }

    /// `mul(act(gate), up)`: the fused gate/up projection kernel when the projections folded,
    /// else the activation-multiply over the separately projected gate and up.
    fn gated(&self, i: usize, kernel: MetalFusion) -> Result<Emission, StepRefusal> {
        use MetalFusion as F;
        let act = self.step_arg(i, 0)?;
        let gate = self.step_arg(act, 0)?;
        let up = self.step_arg(i, 1)?;
        let out = self.colour(i)?;
        let (gate_slot, up_slot) = (self.colour(gate)?, self.colour(up)?);
        match kernel {
            F::FusedGateUpSiluMul | F::FusedGateUpGeluMul
                if matches!(self.matmul(gate, None)?.step, MetalStep::AffineQmm(_)) =>
            {
                // The gate matvec's command, writing the activation's rows; the up's weight after
                // the gate's.
                let (mut g, mut u) = (self.matmul(gate, None)?, self.matmul(up, None)?);
                let (MetalStep::AffineQmm(mut mm), MetalStep::AffineQmm(um)) = (g.step, u.step)
                else {
                    return Err(self.no(i, Refused::FusionShape));
                };
                if (mm.input, mm.n, mm.k, mm.group_size, mm.bits)
                    != (um.input, um.n, um.k, um.group_size, um.bits)
                {
                    return Err(self.no(i, Refused::FusionShape));
                }
                mm.output = out;
                let act = match kernel {
                    F::FusedGateUpGeluMul => st::GatedAct::Gelu,
                    _ => st::GatedAct::Silu,
                };
                g.sites.append(&mut u.sites);
                Ok(em(
                    MetalStep::AffineGatedQmv(mm, act),
                    &[mm.input],
                    &[out],
                    g.sites,
                ))
            }
            F::FusedGateUpSiluMul | F::FusedGateUpGeluMul => {
                // The gate projection's input and weight layer carry the fused command.
                let input = self.read(gate, 0)?;
                let layer = self.weight_layer(gate);
                let step = match kernel {
                    F::FusedGateUpGeluMul => MetalStep::FusedGateUpGeluMul(input, out, layer),
                    _ => MetalStep::FusedGateUpSiluMul(input, out, layer),
                };
                let g = self.base(gate, self.weight_of(gate)?)?;
                let u = self.base(up, self.weight_of(up)?)?;
                let site = vec![WeightSlot {
                    kind: WeightKind::Linear,
                    base: format!("{g}__fused__{u}"),
                }];
                Ok(em(step, &[input], &[gate_slot, up_slot, out], site))
            }
            F::GeluMul => {
                let step = MetalStep::GeluMul(gate_slot, up_slot, out);
                Ok(em(step, &[gate_slot, up_slot], &[out], Vec::new()))
            }
            _ => {
                let step = MetalStep::SiluMul(gate_slot, up_slot, out, self.intermediate);
                Ok(em(step, &[gate_slot, up_slot], &[out], Vec::new()))
            }
        }
    }

    /// An RMSNorm on its own: the first norm reads the embedded hidden; a per-head norm reads a
    /// `[m·heads, head_dim]` view; any other takes its operand's true width.
    fn norm(&self, i: usize, gain: GainConvention) -> Result<Emission, StepRefusal> {
        let out = self.colour(i)?;
        let layer = self.weight_layer(i);
        let site = self.site(i, WeightKind::RmsNorm, self.weight_of(i)?)?;
        let (input, width, rows, writes) = match self.arg(i, 0)? {
            InputRef::Ext(e) => (self.embedded(i, 0, e)?, self.hidden, 1, vec![out]),
            InputRef::Op(a) => {
                let (width, rows) = match *self.op(a) {
                    SubOp::Reshape {
                        rows: RowScale::Times(k),
                        cols,
                    } => (W(cols), k.get()),
                    _ => (W(self.graph.shape(self.graph.op_output[i]).cols), 1),
                };
                // The (w + offset) gain add rides the norm's writes, ahead of its output.
                let writes = self.gain_add(i)?.into_iter().chain([out]).collect();
                (self.colour(a)?, width, rows, writes)
            }
        };
        let step = match gain {
            GainConvention::OnePlusScale => {
                MetalStep::ScalarOffsetRmsNorm(input, out, layer, offset(gain), width, Rows(rows))
            }
            GainConvention::Scale => MetalStep::RmsNorm(input, out, layer, width, Rows(rows)),
        };
        Ok(em(step, &[input], &writes, site))
    }

    /// A matmul, dense or MLX-affine — or a `step` of the sampled rows around one: the construct's
    /// matmul as it records alone, over this step's buffers. Every sampled step's site is the
    /// matmul's: each is realized from how the matmul lowers.
    fn matmul(&self, i: usize, step: Option<st::SampleRowsStep>) -> Result<Emission, StepRefusal> {
        use st::SampleRowsStep as R;
        let is_matmul = |j: &usize| matches!(self.op(*j), SubOp::MatmulTile { .. });
        let mm = match step {
            Some(_) => self.group(i).find(is_matmul),
            None => Some(i),
        };
        let mm = mm.ok_or_else(|| self.no(i, Refused::NoSampledMatmul))?;
        let SubOp::MatmulTile { n, weight } = *self.op(mm) else {
            return Err(self.no(i, Refused::FusionShape));
        };
        let (mut out, mut input, e) = (self.colour(mm)?, self.read(mm, 0)?, self.weight_of(mm)?);
        let layer = self.layer_with_path(e);
        let k = self.graph.tensors[e].rows;
        let (nd, kd) = (st::NDim(n), st::KDim(k));
        let mut site = self.site(mm, WeightKind::Linear, e)?;
        // What the matmul's folds put around its dot: the norm its input passes through (it reads
        // the norm's input and gain), and the residual add its rows feed (it writes the add's
        // buffer, which holds the residual).
        let (mut ends, mut reads) = (st::QmvEnds::default(), Vec::new());
        for f in self.folds.driven(self.steps.slot[mm]) {
            match f.shape {
                FusedShape::NormedMatvec { norm } => {
                    let nrm = self.op_at(mm, norm)?;
                    let SubOp::RmsNorm { eps, gain } = *self.op(nrm) else {
                        return Err(self.no(mm, Refused::FusionShape));
                    };
                    input = self.read(nrm, 0)?;
                    let layer = self.weight_layer(nrm);
                    let (eps, offset) = (st::Eps(eps), offset(gain));
                    ends.norm = Some(st::RowNorm { layer, eps, offset });
                    site.extend(self.site(nrm, WeightKind::RmsNorm, self.weight_of(nrm)?)?);
                }
                FusedShape::MatvecEpilogue {
                    bias,
                    scale,
                    gate_scale,
                    add,
                } => {
                    // A matvec's store computes no gate scale.
                    if gate_scale.is_some() {
                        return Err(self.no(mm, Refused::FusionShape));
                    }
                    ends.bias = bias.map(|_| match weight {
                        GemmWeight::Affine { .. } => st::BiasStorage::Affine,
                        GemmWeight::Dense | GemmWeight::Fp8Dynamic => st::BiasStorage::Dense,
                    });
                    if let Some(sc) = scale {
                        let SubOp::ScalarMul { scale } = *self.op(self.op_at(mm, sc)?) else {
                            return Err(self.no(mm, Refused::FusionShape));
                        };
                        ends.scale = Some(st::Scale(scale));
                    }
                    // The command writes the chain's last buffer: the add's, over the residual.
                    let last = add.map(|(a, _)| a).or(scale).or(bias);
                    let last = last.ok_or_else(|| self.no(mm, Refused::FusionShape))?;
                    out = self.colour(self.op_at(mm, last)?)?;
                    if let Some((_, residual)) = add {
                        reads.push(self.read_operand(mm, residual)?);
                        ends.residual = true;
                    }
                }
                _ => {}
            }
        }
        reads.insert(0, input);
        let plain = match weight {
            GemmWeight::Dense if ends == st::QmvEnds::default() => {
                MetalStep::Gemm(input, out, layer, nd, kd)
            }
            GemmWeight::Dense => return Err(self.no(mm, Refused::FusionShape)),
            GemmWeight::Fp8Dynamic => return Err(self.no(mm, Refused::Fp8Gemm)),
            GemmWeight::Affine { affine } => MetalStep::AffineQmm(st::AffineMatmul {
                input,
                output: out,
                layer,
                n: nd,
                k: kd,
                group_size: Gs(affine.group().get()),
                bits: Bits(affine.bits().get()),
                vector_limit: st::QmvBatchLimit(affine_qmm_vector_limit(k, n)),
                ends,
            }),
        };
        let (Some(step), MetalStep::AffineQmm(g)) = (step, &plain) else {
            return match step {
                None => Ok(em(plain, &reads, &[out], site)),
                Some(_) => Err(self.no(i, Refused::SampledNotAffine)),
            };
        };
        let (reads, writes): (&[Slot], _) = match step {
            R::Gather => (&[input], [input]),
            R::Matmul => (&[input], [out]),
            R::Scatter => (&[out], [out]),
            R::AllRows => (&[input, out], [out]),
        };
        Ok(em(MetalStep::SampleRows(*g, step), reads, &writes, site))
    }

    /// A step no fold touches, lowered on its own.
    fn kept(&self, i: usize) -> Result<Option<Emission>, StepRefusal> {
        use EwKind as E;
        use MetalStep as S;
        use SubOp as L;
        let out = || self.colour(i);
        type Unary<'a> = &'a dyn Fn(Slot, Slot, st::ActivationWidth) -> MetalStep;
        let unary = |f: Unary| -> Result<Emission, StepRefusal> {
            let (input, out) = (self.read(i, 0)?, out()?);
            let step = f(input, out, self.width(i));
            Ok(em(step, &[input], &[out], Vec::new()))
        };
        Ok(Some(match *self.op(i) {
            // A view: no kernel runs, so the barrier walk neither fences on it nor bookkeeps it.
            L::Reshape { rows, .. } => {
                let (input, out) = (self.read(i, 0)?, out()?);
                let (mult, div) = match rows {
                    RowScale::Times(k) => (k.get(), 1),
                    RowScale::Over(k) => (1, k.get()),
                };
                let step = S::Reshape(input, out, Rows(mult), st::RowsDivisor(div));
                let mut e = em(step, &[input], &[out], Vec::new());
                e.sig.metadata = true;
                e
            }
            L::EmbeddingGather { indices_kind } => {
                let indices = match indices_kind {
                    0 => st::GatherIndices::WindowIndex,
                    1 => st::GatherIndices::ReverseIndices,
                    value => {
                        let field = "indices_kind";
                        return Err(self.no(i, Refused::OutOfDomain { field, value }));
                    }
                };
                let (input, out) = (self.read(i, 0)?, out()?);
                em(
                    S::EmbeddingGather(input, out, indices, self.width(i)),
                    &[input],
                    &[out],
                    Vec::new(),
                )
            }
            L::GateScale => {
                let ins = [self.read(i, 0)?, self.read(i, 1)?, self.read(i, 2)?];
                let out = out()?;
                em(
                    S::GateScale(ins[0], ins[1], ins[2], out),
                    &ins,
                    &[out],
                    Vec::new(),
                )
            }
            L::LoadPixels { .. } => em(S::LoadPixels(out()?), &[], &[out()?], Vec::new()),
            L::LoadPosEmbeds { .. } => em(S::LoadPosEmbeds(out()?), &[], &[out()?], Vec::new()),
            L::Elementwise(E::Gelu) => unary(&S::Gelu)?,
            L::Elementwise(E::QuickGelu) => unary(&S::QuickGelu)?,
            L::Elementwise(E::GeluErf) => unary(&S::GeluErf)?,
            L::TanhSoftCap => unary(&S::TanhSoftCap)?,
            L::RmsNorm { gain, .. } => self.norm(i, gain)?,
            L::MatmulTile { .. } => {
                let sampled = self
                    .group(i)
                    .any(|j| matches!(self.op(j), L::SampleRowsGather));
                self.matmul(i, sampled.then_some(st::SampleRowsStep::Matmul))?
            }
            L::SampleRowsGather => self.matmul(i, Some(st::SampleRowsStep::Gather))?,
            L::SampleRowsScatter => self.matmul(i, Some(st::SampleRowsStep::Scatter))?,
            L::AllRowsMatmul => self.matmul(i, Some(st::SampleRowsStep::AllRows))?,
            L::RopeAppend { .. } => return Err(self.no(i, Refused::NoFoldedRotate)),
            L::AttnDecode { mask, .. } => {
                let out = out()?;
                let q = self.colour(self.step_arg(i, 0)?)?;
                // The layer and pairing come from the rope append whose cache it reads.
                let rope = self.step_arg(i, 3)?;
                let (layer, pairing) = match *self.op(rope) {
                    L::RopeAppend { layer, pairing, .. } => (LayerId(layer), pairing),
                    _ => (LayerId(0), RopeFormTag::NeoX),
                };
                // Prefill runs the paged kernel with K arriving pre-roped: no rotary. Decode's
                // CosSin is the GLOBAL rotary whatever the class (only the rope uses the local).
                let neox = RopeFormTag::NeoX;
                let rotary = || {
                    let base = "rotary".to_string();
                    vec![WeightSlot {
                        kind: WeightKind::CosSin,
                        base,
                    }]
                };
                use AttnMask::{Causal, SlidingWindow as Sliding};
                let (step, sites) = match (self.l.input.ops[i].m > 1, mask) {
                    (true, Sliding) => {
                        (S::SlidingAttentionPrefillPaged(q, out, layer, neox), vec![])
                    }
                    (true, Causal) => (S::AttentionPrefillPaged(q, out, layer, neox), vec![]),
                    (false, Sliding) => (
                        S::SlidingAttentionViaCache(q, out, layer, pairing),
                        rotary(),
                    ),
                    (false, Causal) => (S::AttentionViaCache(q, out, layer, pairing), rotary()),
                };
                let v = match self.arg(i, 4) {
                    Ok(InputRef::Op(v)) => self.colour(v).ok(),
                    _ => None,
                };
                let reads: Vec<Slot> = [Some(q), self.colour(rope).ok(), v]
                    .into_iter()
                    .flatten()
                    .collect();
                let mut e = em(step, &reads, &[out], sites);
                e.sig.kv_r = Some(layer);
                // A coded attention reads its K/V staged, off decode steps.
                let staged = |j: usize| matches!(self.op(j), L::KvStage { .. });
                if self.group(i).any(staged) {
                    e.sig.codec_staging = Access::Read;
                }
                e
            }
            // The KV codec: each step's operands are its writer's, its site the writer's.
            L::KvStage { operand } => {
                let (layer, _, class, offsets, sites) =
                    self.codec_writer(i, self.step_arg(i, 0)?)?;
                let mut e = em(S::KvStage(operand, layer, class, offsets), &[], &[], sites);
                // The layer's packed store in, the staging buffer out.
                e.sig.kv_r = Some(layer);
                e.sig.codec_staging = Access::Write;
                e
            }
            L::RotateRows { rows } => {
                let x = self.read(i, 0)?;
                em(S::RotateRows(x, rows), &[x], &[x], Vec::new())
            }
            L::AttnPackedKv => {
                // The layer's decode attention: its writer's layer and pairing.
                let (layer, pairing, class, offsets, sites) =
                    self.codec_writer(i, self.step_arg(i, 2)?)?;
                let (q, out) = (self.read(i, 0)?, out()?);
                let step = S::AttnPackedKv(q, out, layer, pairing, class, offsets);
                let mut e = em(step, &[q, out], &[out], sites);
                e.sig.kv_r = Some(layer);
                e
            }
            L::Elementwise(E::Mul) => return Err(self.no(i, Refused::NoGatedFold)),
            L::SiluMul => return Err(self.no(i, Refused::SplitSiluMul)),
            // Identity — the op's buffer IS its operand's (in place by contract).
            L::ScalarMul { scale: 1.0 } => return Ok(None),
            L::ScalarMul { scale } => unary(&|a, b, w| S::ScalarMul(a, b, st::Scale(scale), w))?,
            // The accessor is the UPSTREAM gemm's LinearLayer (the bias rides on it): base name,
            // layer and storage all come from that gemm.
            L::Elementwise(E::BiasAdd) => {
                let out = out()?;
                let up = self.step_arg(i, 0)?;
                let input = self.colour(up)?;
                let L::MatmulTile { n, weight } = *self.op(up) else {
                    let upstream = self.op(up).name();
                    return Err(self.no(i, Refused::BiasUpstream(upstream)));
                };
                let e = self.weight_of(up)?;
                let storage = match weight {
                    GemmWeight::Affine { .. } => st::BiasStorage::Affine,
                    _ => st::BiasStorage::Dense,
                };
                let layer = self.layer_with_path(e);
                let step = S::MetalBiasAdd(input, out, layer, st::NDim(n), storage);
                em(
                    step,
                    &[input],
                    &[out],
                    self.site(up, WeightKind::Linear, e)?,
                )
            }
            // In place: `Add(delta, out)` accumulates into the out buffer, which aliases the other
            // operand.
            L::Elementwise(E::Add) => {
                let out = out()?;
                let (a, b) = (self.read(i, 0)?, self.read(i, 1)?);
                let other = if out == b { a } else { b };
                let step = S::Add(other, out, self.width(i));
                em(step, &[a, b], &[out], Vec::new())
            }
            L::ScalarWeightMul => {
                let (out, input, e) = (out()?, self.read(i, 0)?, self.weight_of(i)?);
                let step = S::ScalarWeightMul(input, out, self.layer(e));
                em(
                    step,
                    &[input],
                    &[out],
                    self.site(i, WeightKind::RmsNorm, e)?,
                )
            }
            L::GateSplit { .. } => {
                let (out, input, gate) = (out()?, self.read(i, 0)?, self.second(i)?);
                em(
                    S::GateSplit(input, out, gate),
                    &[input],
                    &[out, gate],
                    Vec::new(),
                )
            }
            // The gate came from a GateSplit's SECOND output.
            L::GateApply => {
                let (out, attn) = (out()?, self.read(i, 0)?);
                let gate = match self.arg(i, 1)? {
                    InputRef::Op(a) => self.second(a)?,
                    InputRef::Ext(e) => self.embedded(i, 1, e)?,
                };
                em(
                    S::GateApply(attn, gate, out),
                    &[attn, gate],
                    &[out],
                    Vec::new(),
                )
            }
            L::VisionRope => {
                let (q, k) = (self.read(i, 0)?, self.read(i, 1)?);
                let (q_out, k_out) = (out()?, self.second(i)?);
                em(
                    S::VisionRope(q, k, q_out, k_out),
                    &[q, k],
                    &[q_out, k_out],
                    Vec::new(),
                )
            }
            L::VarlenAttention { cu_kind } => {
                let out = out()?;
                let q = self.read(i, 0)?;
                // K roped by a VisionRope is that rope's SECOND output.
                let k = match self.arg(i, 1)? {
                    InputRef::Op(a) if matches!(self.op(a), L::VisionRope) => self.second(a)?,
                    _ => self.read(i, 1)?,
                };
                let v = self.read(i, 2)?;
                let segments = match cu_kind {
                    0 => st::CuSeqlens::Batch,
                    1 => st::CuSeqlens::VisionFull,
                    2 => st::CuSeqlens::VisionWindow,
                    value => {
                        let field = "cu_kind";
                        return Err(self.no(i, Refused::OutOfDomain { field, value }));
                    }
                };
                em(
                    S::VarlenAttention(q, k, v, out, segments),
                    &[q, k, v],
                    &[out],
                    Vec::new(),
                )
            }
            L::EncoderAttn { .. } => {
                let out = out()?;
                let [q, k, v] = [self.read(i, 0)?, self.read(i, 1)?, self.read(i, 2)?];
                em(
                    S::EncoderAttention(q, k, v, out),
                    &[q, k, v],
                    &[out],
                    Vec::new(),
                )
            }
            L::GatedDeltaNet => {
                let out = out()?;
                let ins = [
                    self.read(i, 0)?,
                    self.read(i, 1)?,
                    self.read(i, 2)?,
                    self.read(i, 3)?,
                ];
                let e = self.weight_of(i)?;
                let step = S::GatedDeltaNet(ins[0], ins[1], ins[2], ins[3], out, self.layer(e));
                let site = self.site(i, WeightKind::GatedDeltaNet, e)?;
                let mut e = em(step, &ins, &[out], site);
                // Its four commands pass their intermediates through the op scratch.
                e.sig.op_scratch = Access::Write;
                e
            }
            L::RouterNorm { eps, router } => {
                let (x, e) = (self.read(i, 0)?, self.source_arg(i, 1)?);
                let step = MoeStep::RouterNorm(x, self.layer(e), st::Eps(eps));
                self.moe(i, step, &[x], &[], Some((router.weight_kind(), e)))?
            }
            L::RouterLogits { router, .. } => {
                let ((mut rows, mut reads), e) = (self.moe_rows(i, 0)?, self.source_arg(i, 1)?);
                // Its folded pre-norm: it reads the norm's rows, normalizing them as it loads.
                let mut norm = None;
                for f in self.folds.driven(self.steps.slot[i]) {
                    let FusedShape::NormedMatvec { norm: n } = f.shape else {
                        return Err(self.no(i, Refused::FusionShape));
                    };
                    let SubOp::RouterNorm { eps, .. } = *self.op(self.op_at(i, n)?) else {
                        return Err(self.no(i, Refused::FusionShape));
                    };
                    (rows, reads) = self.moe_rows(self.op_at(i, n)?, 0)?;
                    norm = Some(st::Eps(eps));
                }
                let step = MoeStep::RouterLogits(rows, self.layer(e), norm);
                self.moe(i, step, &reads, &[], Some((router.weight_kind(), e)))?
            }
            L::RouteSoftmax => {
                let scores = match self.region(self.step_arg(i, 0)?)? {
                    MoeRegion::RouterLogits => st::MoeScores::Router,
                    MoeRegion::TopKScores => st::MoeScores::TopK,
                    r => return Err(self.no(i, Refused::SoftmaxOver(r))),
                };
                self.moe(i, MoeStep::Softmax(scores), &[], &[], None)?
            }
            L::RouteArgsort => self.moe(i, MoeStep::Argsort, &[], &[], None)?,
            L::RouteTopK { .. } => self.moe(i, MoeStep::TopK, &[], &[], None)?,
            L::RouteGatherScores => self.moe(i, MoeStep::GatherScores, &[], &[], None)?,
            L::RouteScale { scale } => {
                self.moe(i, MoeStep::Scale(st::Scale(scale)), &[], &[], None)?
            }
            L::RouteRenorm => self.moe(i, MoeStep::Renorm, &[], &[], None)?,
            L::RouteExpertScale { router } => {
                let e = self.source_arg(i, 2)?;
                let step = MoeStep::ExpertScale(self.layer(e));
                self.moe(i, step, &[], &[], Some((router.weight_kind(), e)))?
            }
            L::ExpertSort { .. } => {
                let x = self.read(i, 0)?;
                self.moe(i, MoeStep::Sort(x), &[x], &[], None)?
            }
            L::ExpertMatmul { .. } => {
                let x = self.expert_matmul(i)?;
                let step = MoeStep::ExpertMatmul(x.step);
                let mut em = self.moe(i, step, &x.reads, &[], Some(x.weight))?;
                if let Some(raw) = x.raw {
                    em.raw = Some(MetalStep::Moe(self.block(i)?, MoeStep::ExpertMatmul(raw)));
                }
                em
            }
            L::ExpertGatedAct { act } => self.moe(i, MoeStep::GatedAct(act), &[], &[], None)?,
            L::ExpertUnsort => self.moe(i, MoeStep::Unsort, &[], &[], None)?,
            L::ExpertCombine { .. } => {
                self.shared_expert_owned(i)?;
                let out = out()?;
                self.moe(i, MoeStep::Combine(out), &[], &[out], None)?
            }
            // Standalone unit norm (the unfused rope path): width and rows from the input view.
            L::RmsNormUnit { .. } => {
                let (out, a) = (out()?, self.step_arg(i, 0)?);
                let (width, rows) = match *self.op(a) {
                    L::Reshape {
                        rows: RowScale::Times(k),
                        cols,
                    } => (W(cols), k.get()),
                    _ => (self.hidden, 1),
                };
                let input = self.colour(a)?;
                em(
                    S::RmsNormUnit(input, out, width, Rows(rows)),
                    &[input],
                    &[out],
                    Vec::new(),
                )
            }
            L::Mean
            | L::Elementwise(E::Sub | E::Silu)
            | L::RopeRotate { .. }
            | L::KvEncode { .. } => {
                return Err(self.no(i, Refused::Escaped));
            }
            L::SumReduce { .. } | L::RmsNormReduce { .. } | L::RmsNormApply { .. } => {
                return Err(self.no(i, Refused::TileLevel));
            }
        }))
    }
}

/// The constant a `(1 + w)` gain adds to the stored weight.
fn offset(gain: GainConvention) -> st::GainOffset {
    st::GainOffset(gain.offset())
}

/// The MLX qmv batch limit for an affine-quantized matmul, from the target's table — fixed at M4
/// whatever the generation class the bake targets.
fn affine_qmm_vector_limit(k: u32, n: u32) -> u32 {
    use scratchy_target_metal::targets::AppleSiliconGen;
    scratchy_target_metal::quantized::get_qmv_batch_limit(k, n, AppleSiliconGen::M4)
}

/// Which MoE bits a layout carries: the repacked per-projection widths, or each op's raw width.
///
/// ⚠️ THE ASYMMETRY IS THE BASELINE'S: the rolled and unrolled layouts are repacked, a peeled
/// candidate is not — so on a quantized Qwen-MoE a peeled roll never proves.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MoeBits {
    Repacked,
    Raw,
}

/// A layout of the records: the backbone rows (loop markers included) and the lm_head rows,
/// each with its hazard signatures and weight sites.
pub struct Assembled {
    pub rows: Vec<StepRow>,
    sigs: Vec<HazardSig>,
    pub sites: Vec<Vec<WeightSlot>>,
    pub lm_rows: Vec<StepRow>,
    lm_sigs: Vec<HazardSig>,
    pub lm_sites: Vec<Vec<WeightSlot>>,
}

impl Assembled {
    /// The barrier flags of the backbone and the lm_head, walked as ONE stream: the lm_head's
    /// first row fences against what the backbone left pending.
    pub fn flags(&self) -> (Vec<bool>, Vec<bool>) {
        let mut all = hazard_flags(self.sigs.iter().chain(&self.lm_sigs));
        let lm = all.split_off(self.sigs.len());
        (all, lm)
    }
}

/// Lay the records out along `items`: head rows first, each step's record at its position, a
/// `Loop` marker at each loop's opening once its body length is known.
pub fn assemble(
    items: &[TapeItem],
    records: &StepRecords,
    moe: MoeBits,
) -> Result<Assembled, StepRefusal> {
    let mut out = Assembled {
        rows: Vec::new(),
        sigs: Vec::new(),
        sites: Vec::new(),
        lm_rows: Vec::new(),
        lm_sigs: Vec::new(),
        lm_sites: Vec::new(),
    };
    let row = |e: &Emission| match (moe, &e.raw) {
        (MoeBits::Raw, Some(raw)) => StepRow::Step(raw.clone(), e.gate),
        _ => StepRow::Step(e.step.clone(), e.gate),
    };
    for e in &records.head {
        out.rows.push(row(e));
        out.sigs.push(e.sig.clone());
        out.sites.push(e.sites.clone());
    }
    // ⛔ A STACK, NOT AN `Option`: gemma-4 nests a loop over its five sliding layers inside the
    // loop over its six-layer cell.
    let mut open: Vec<(usize, u32, u32)> = Vec::new();
    for item in items {
        match item {
            TapeItem::OpenLoop { iters, stride, .. } => {
                open.push((out.rows.len(), *iters, *stride))
            }
            TapeItem::CloseLoop { .. } => {
                let (at, iters, stride) = open
                    .pop()
                    .ok_or(canonical_refusal(Refused::LoopUnbalanced))?;
                let body = out.rows.len() - at;
                if body > 0 {
                    let marker = StepRow::Loop {
                        iters: st::LoopIters(iters),
                        body: st::BodyLen(body as u32),
                        stride: st::LayerStride(stride),
                    };
                    out.rows.insert(at, marker);
                    let metadata = HazardSig {
                        metadata: true,
                        ..HazardSig::default()
                    };
                    out.sigs.insert(at, metadata);
                    out.sites.insert(at, Vec::new());
                }
            }
            TapeItem::Step(s) => {
                let Some(e) = &records.per_op[s.source_op.0] else {
                    continue;
                };
                let (rows, sigs, sites) = match e.lm_head {
                    true => (&mut out.lm_rows, &mut out.lm_sigs, &mut out.lm_sites),
                    false => (&mut out.rows, &mut out.sigs, &mut out.sites),
                };
                rows.push(row(e));
                sigs.push(e.sig.clone());
                sites.push(e.sites.clone());
            }
        }
    }
    Ok(out)
}

/// THE MTL4 barrier flags: a row fences when it reads a buffer pending a write (RAW), writes one
/// pending a write or read (WAW/WAR), or touches a KV layer or the op scratch pending a
/// conflicting access. The first dispatch never fences; a fence clears everything pending;
/// metadata rows are invisible. The rows one construct expanded to (`HazardSig::group`) fence as
/// ONE: the first on the union of their accesses, every later one unconditionally.
pub fn hazard_flags<'a>(sigs: impl Iterator<Item = &'a HazardSig>) -> Vec<bool> {
    let sigs: Vec<&HazardSig> = sigs.collect();
    let mut pending = Pending::default();
    let (mut flags, mut first, mut i) = (Vec::with_capacity(sigs.len()), true, 0);
    while i < sigs.len() {
        if sigs[i].metadata {
            flags.push(false);
            i += 1;
            continue;
        }
        let g = sigs[i].group;
        let rest = sigs[i + 1..].iter();
        let end = i + 1 + rest.take_while(|s| g.is_some() && s.group == g).count();
        let need = !first && sigs[i..end].iter().any(|s| pending.conflicts(s));
        if need {
            pending = Pending::default();
        }
        sigs[i..end].iter().for_each(|s| pending.add(s));
        first = false;
        flags.push(need);
        flags.extend(std::iter::repeat_n(true, end - i - 1));
        i = end;
    }
    flags
}

/// What the walk has seen written and read since the last fence.
#[derive(Default)]
struct Pending {
    w: HashSet<Slot>,
    r: HashSet<Slot>,
    kv_w: HashSet<LayerId>,
    kv_r: HashSet<LayerId>,
    op_scratch: Access,
    codec_staging: Access,
}

impl Pending {
    fn conflicts(&self, s: &HazardSig) -> bool {
        let arena = s.reads.iter().any(|x| self.w.contains(x))
            || s.writes
                .iter()
                .any(|x| self.w.contains(x) || self.r.contains(x));
        let kv = s.kv_r.is_some_and(|l| self.kv_w.contains(&l))
            || s.kv_w
                .is_some_and(|l| self.kv_w.contains(&l) || self.kv_r.contains(&l));
        let one = |access: Access, pending: Access| match access {
            Access::Untouched => false,
            Access::Read => pending == Access::Write,
            Access::Write => pending != Access::Untouched,
        };
        let scratch = one(s.op_scratch, self.op_scratch);
        let staging = one(s.codec_staging, self.codec_staging);
        arena || kv || scratch || staging
    }

    fn add(&mut self, s: &HazardSig) {
        self.w.extend(s.writes.iter().copied());
        self.r.extend(s.reads.iter().copied());
        self.kv_w.extend(s.kv_w);
        self.kv_r.extend(s.kv_r);
        self.op_scratch = self.op_scratch.max(s.op_scratch);
        self.codec_staging = self.codec_staging.max(s.codec_staging);
    }
}

/// THE ROLL PROOF: `candidate` with every loop expanded — a body row `i·stride` layers on per
/// iteration, nesting summed — must be exactly `unrolled`: rows, barrier flags, and weight sites
/// (a rolled body binds the same source families every iteration; only the layer advances).
pub fn proves(
    candidate: &Assembled,
    flags: &[bool],
    unrolled: &Assembled,
    unrolled_flags: &[bool],
) -> Result<(), String> {
    fn expand(rows: &[StepRow], at: usize, base: u32, out: &mut Vec<(StepRow, usize)>) {
        let mut k = 0;
        while k < rows.len() {
            match &rows[k] {
                &StepRow::Loop {
                    iters,
                    body,
                    stride,
                } => {
                    let b = k + 1..k + 1 + body.0 as usize;
                    for it in 0..iters.0 {
                        expand(&rows[b.clone()], at + b.start, base + it * stride.0, out);
                    }
                    k = b.end;
                }
                StepRow::Step(s, gate) => {
                    out.push((StepRow::Step(s.clone().advanced(base), *gate), at + k));
                    k += 1;
                }
            }
        }
    }
    let mut expanded = Vec::new();
    expand(&candidate.rows, 0, 0, &mut expanded);
    if expanded.len() != unrolled.rows.len() {
        return Err(format!(
            "expanded len {} != original {}",
            expanded.len(),
            unrolled.rows.len()
        ));
    }
    for (i, ((a, _), b)) in expanded.iter().zip(&unrolled.rows).enumerate() {
        if a != b {
            return Err(format!("row {i}: original={b:?} expanded={a:?}"));
        }
    }
    for (i, ((_, k), b)) in expanded.iter().zip(unrolled_flags).enumerate() {
        if flags[*k] != *b {
            return Err(format!(
                "barrier row {i}: original={b} expanded={}",
                flags[*k]
            ));
        }
    }
    for (i, ((_, k), b)) in expanded.iter().zip(&unrolled.sites).enumerate() {
        if candidate.sites[*k] != *b {
            let a = &candidate.sites[*k];
            return Err(format!("site row {i}: original={b:?} expanded={a:?}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(reads: &[u32], writes: &[u32], group: Option<u32>) -> HazardSig {
        HazardSig {
            reads: reads.iter().map(|s| Slot(*s)).collect(),
            writes: writes.iter().map(|s| Slot(*s)).collect(),
            group: group.map(ExpansionId),
            ..HazardSig::default()
        }
    }

    /// A construct's rows fence as ONE: the first on every member's accesses, the rest always.
    #[test]
    fn a_constructs_rows_fence_as_one() {
        // Alone, row 1 has nothing pending to fence on; row 2 reads row 0's write.
        let alone = [
            sig(&[], &[1], None),
            sig(&[2], &[], None),
            sig(&[1], &[3], None),
        ];
        assert_eq!(hazard_flags(alone.iter()), [false, false, true]);
        let one = [
            sig(&[], &[1], None),
            sig(&[2], &[], Some(0)),
            sig(&[1], &[3], Some(0)),
        ];
        assert_eq!(hazard_flags(one.iter()), [false, true, true]);
    }

    /// The op scratch is one location: a write fences on any pending access, a read on a write.
    #[test]
    fn the_op_scratch_is_one_location() {
        let scratch = |op_scratch| HazardSig {
            op_scratch,
            ..HazardSig::default()
        };
        let rows = [Access::Read, Access::Read, Access::Write, Access::Read].map(scratch);
        assert_eq!(hazard_flags(rows.iter()), [false, false, true, true]);
    }
}
