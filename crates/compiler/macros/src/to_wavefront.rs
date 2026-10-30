// SPDX-License-Identifier: Apache-2.0
//! PD-wavefront task **T2b** — the macro→wavefront bridge.
//!
//! Translate a *solved* decode FUF (`Fuf` + solver [`Assignment`]) into a
//! [`scratchy_subtile::lower::LoweringInput`] and hand it to
//! [`scratchy_subtile::lower::lower`], so a real Llama-3.2-1B forward
//! flows through the host-validated subtile pipeline (DAG eval + tape
//! player + wavefront scheduler).
//!
//! This is the one piece that *cannot* be `cargo test`-ed on a Mac (the
//! macro crate's test suite is cuda-coupled), so it is kept deliberately
//! thin and mechanical: every structural decision it makes — embed
//! becomes a read-only `Source`, `rope_append` splits into a Q-side
//! `RopeRotate` + a K-side `RopeAppend` (the GPU cache-write) with the
//! un-roped V aliasing the V-proj output, `attention` reads the prefix KV
//! cache as `Source` segments + the new token as `Sub` edges, `lm_head` is
//! the result — is mirrored exactly
//! by the Mac-testable `full_forward_bit_exact` test in
//! `scratchy_subtile::lower`. The bridge resolves shapes and wires
//! edges; the decomposition *semantics* it targets are already proven
//! bit-exact vs `cpu_golden` there.
//!
//! Coarse granularity (one subtile per op) per the plan: split-K /
//! N-block tiling and the inter-op slicing it needs are layered on at
//! the scheduler, not here.
//!
//! Scope: the Llama-3.2 decode op set (`Embed`, `RmsNorm`, `Gemm`,
//! `RopeAppend`, `Attention`, `Silu`, `Mul`, `Add`). Anything else
//! (MoE, MLA, sliding-window / interleaved rope, vision, TP collectives)
//! is surfaced as [`BridgeError::UnsupportedOp`] — never silently
//! skipped. The drive treats any error as "no wavefront lowering for
//! this model" and continues; this never gates a normal build.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};

use scratchy_subtile::handoff::{BundleTensor, Expansion, ExpansionId};
use scratchy_subtile::lower::{
    AffineInt4, ArchOp, ExpertQuant, GemmWeight, InputRef, LoweringInput, OpDesc,
};
use scratchy_subtile::model_geometry::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
use scratchy_subtile::subtile_ir::{
    AttnMask, DraftConvStage, EwKind, ExpertBody, ExpertBundle, ExpertProj, GainConvention,
    GatedAct, NumExperts, RopeFormTag, RouterBundle, RowScale, SharedExpertBound, SourceShape,
    SubOp, TopK,
};

use crate::assignment::Assignment;
use crate::classified::{ExternKind, OpKind, UnrollIndex};
use crate::codegen::weight_kind_accessor_method;
use crate::config::ModelParams;
use crate::fuf::{Fuf, FufInput, FufNode, TileId};
use crate::quantization::StorageFormat;
use crate::shape::Inferred;
use crate::weight_vocab::{
    WeightKind, WeightSlot, attention_scale_for, eval_shape_with, gemm_nk_from_fuf,
};

/// Why a FUF couldn't be lowered to a wavefront `LoweringInput`. Always
/// surfaced (logged by the drive), never papered over.
#[derive(Debug)]
pub enum BridgeError {
    /// FUF op kind outside the coarse Llama-3.2 decode set.
    UnsupportedOp { tile: TileId, op: OpKind },
    /// A tile's (or an upstream tile's) output didn't close to a
    /// concrete shape under `bounds` — shape inference left a `Var`, or
    /// a dim referenced a missing config bound.
    UnresolvedShape { tile: TileId, what: &'static str },
    /// A tile referenced a `(TileId, slot)` the walk hadn't produced —
    /// FUF not in ascending-id topological order, or a bad slot.
    DanglingInput { tile: TileId, dep: TileId, slot: u8 },
    /// An op's inputs didn't match the arity / kinds the bridge expects
    /// for that `OpKind`.
    MalformedOp {
        tile: TileId,
        op: OpKind,
        detail: &'static str,
    },
    /// A required model scalar (e.g. `rms_norm_eps`) was absent.
    MissingScalar { key: &'static str },
    /// A required integer bound (e.g. `head_dim`) was absent.
    MissingBound { key: &'static str },
    /// The config's kv-head count does not divide its query-head count, so the model has no GQA
    /// grouping at all. Refused HERE, at the one place the three numbers are read, rather than
    /// papered over downstream by a `max(1)` that yields a group size whose attention reads another
    /// head's keys — fluent wrong output, never a fault.
    NoHeadGrouping {
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    },
    /// The FUF produced no result op (empty, or the last tile wasn't a
    /// value-producing op).
    NoResult,
    /// An RmsNorm's folded `w + offset` gain is neither the `w` nor the `(1 + w)` convention, and no
    /// kernel implements a third — rounding it to the nearest would scale every normalized
    /// activation by the wrong constant.
    UnknownGainConvention { tile: TileId, offset: f32 },
    /// A Gemm's weight carries a quantization `StorageFormat` the SDSC/wavefront lowering does not yet
    /// emit (fp8 needs the weight_scale wired as a 3rd Gemm input + 1-byte staging; int4/ggml unwired).
    /// A build-time refusal — NOT a silent mis-lower of a packed weight as dense fp16.
    UnsupportedWeight { tile: TileId, detail: &'static str },
    /// METAL HAS NO KERNEL for this shape at all — not a gap in the
    /// bridge, a gap in the backend. Distinct from
    /// [`Self::UnsupportedWeight`] because the two are handled
    /// differently: this one is TOLERATED the way the runtime-lowering
    /// path tolerated it (the preset bakes an empty variant list and
    /// the pool refuses at load), while a bridge gap must stay fatal
    /// or the front-end swap silently drops coverage the solver had.
    NoMetalRealization { tile: TileId, detail: &'static str },
    /// A MoE's config declares an expert activation no expert body realizes.
    UnrealizedExpertActivation { tile: TileId, act: String },
}

impl BridgeError {
    /// See [`Self::NoMetalRealization`].
    pub fn is_no_metal_realization(&self) -> bool {
        matches!(self, Self::NoMetalRealization { .. })
    }
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOp { tile, op } => {
                write!(
                    f,
                    "tile {} op {op:?} is outside the coarse decode set",
                    tile.0
                )
            }
            Self::UnresolvedShape { tile, what } => {
                write!(
                    f,
                    "tile {} {what} did not close to a concrete shape",
                    tile.0
                )
            }
            Self::DanglingInput { tile, dep, slot } => write!(
                f,
                "tile {} reads ({}, slot {}) which was not produced yet",
                tile.0, dep.0, slot
            ),
            Self::MalformedOp { tile, op, detail } => {
                write!(f, "tile {} op {op:?}: {detail}", tile.0)
            }
            Self::MissingScalar { key } => write!(f, "model scalar `{key}` missing"),
            Self::MissingBound { key } => write!(f, "config bound `{key}` missing"),
            Self::NoHeadGrouping {
                num_q_heads,
                num_kv_heads,
                head_dim,
            } => write!(
                f,
                "config declares num_attention_heads={num_q_heads}, \
                 num_key_value_heads={num_kv_heads}, head_dim={head_dim} — the kv-head count does \
                 not divide the query-head count, so this model has no GQA grouping to emit"
            ),
            Self::NoResult => write!(f, "FUF produced no result op"),
            Self::UnknownGainConvention { tile, offset } => write!(
                f,
                "tile {} rmsnorm gain offset {offset} is neither the w nor the (1+w) convention, \
                 and no kernel implements a third",
                tile.0
            ),
            Self::NoMetalRealization { tile, detail } => {
                write!(f, "tile {}: {detail}", tile.0)
            }
            Self::UnrealizedExpertActivation { tile, act } => write!(
                f,
                "tile {} moe_block: the config declares expert activation `{act}`, which no \
                 expert body realizes (a gated `silu` bank or an ungated `relu2` one)",
                tile.0
            ),
            Self::UnsupportedWeight { tile, detail } => {
                write!(
                    f,
                    "tile {} gemm weight scheme not lowered: {detail}",
                    tile.0
                )
            }
        }
    }
}

impl std::error::Error for BridgeError {}

/// What real tensor each wavefront `Source` is bound to at run time —
/// parallel to [`LoweringInput::sources`]. The bridge anonymizes sources
/// to bare shapes; this manifest is how the Tier-B host executor (and
/// later the GPU tape player) maps `SourceId(i)` back to a concrete
/// buffer. `Weight` carries the raw [`crate::classified::WeightId`] +
/// unrolled index (resolve to a tensor name via the same `Program`).
pub use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};

/// Lightweight stats for the macro-time dump.
#[derive(Debug, Clone)]
pub struct BridgeStats {
    pub fuf_tiles: usize,
    pub subgraphs: usize,
    pub sources: usize,
    pub ops: usize,
    /// `(op kind name, count)`, sorted by name.
    pub op_histogram: Vec<(&'static str, usize)>,
    /// Source bindings bucketed: weights, prefix-KV pairs, and the
    /// fixed singletons (embed/cos/sin).
    pub weight_sources: usize,
    pub prefix_sources: usize,
}

/// One producer of a FUF `(tile, slot)`: either a wavefront op output,
/// or a graph `Source` (embed becomes a source; a `rope_append`'s
/// un-roped V slot aliases the V-proj op output, which is itself an
/// `Op`, so that case stays `Op` here).
#[derive(Clone, Copy)]
enum Producer {
    Op(usize),
    Ext(usize),
}

impl Producer {
    fn as_input_ref(self) -> InputRef {
        match self {
            Producer::Op(j) => InputRef::Op(j),
            Producer::Ext(e) => InputRef::Ext(e),
        }
    }
}

/// Mutable bridge state threaded through the FUF walk.
struct Builder<'a> {
    fuf: &'a Fuf,
    inferred: &'a Inferred,
    model: &'a ModelParams,
    /// Folded `w + scalar` gain offsets, keyed by the ADD tile id.
    /// A consuming rmsnorm carries the value as `gain_offset`; the
    /// declared-offset case (kernel const) records 0.0.
    gain_offsets: HashMap<u32, f32>,
    /// Per rmsnorm OP index: the folded gain-Add's tile id.
    norm_gain_add_tiles: HashMap<usize, u32>,
    /// The GLOBAL attention class's geometry on hybrid arches whose
    /// global dims differ from the base (Gemma4 hd512); `None` when
    /// uniform — `OpKind::Attention` ops use it when present.
    geom_global: Option<ModelAttnGeometry>,
    bounds: BTreeMap<String, u64>, // model bounds + num_tokens=1
    sources: Vec<SourceShape>,
    bindings: Vec<SourceBinding>,
    ops: Vec<OpDesc>,
    /// FUF `(tile_id, slot)` → its producer in the wavefront graph.
    produced: HashMap<(u32, u8), Producer>,
    /// Dedup weight sources by `(WeightId, index)`.
    weight_src: HashMap<(u32, Option<UnrollIndex>), usize>,
    /// Shared cos / sin source indices, keyed by the rope's FULL column width
    /// (`heads * head_dim`). The rotary row is provided pre-tiled to that width
    /// by the host worker — the sendnn graph multiplies activations by it
    /// directly (an in-graph `Concat`-tile of an identical `[1, head_dim]` row
    /// crashes the DeepTools partitioner; see `guard_known_crash_patterns`).
    /// GQA needs two widths (Q: `n_q_heads*hd`, K: `n_kv_heads*hd`). The eval /
    /// ktir / metal consumers read the first `head_dim` and broadcast, so a
    /// wider source is backward-compatible for them.
    cos_sin: BTreeMap<(u32, bool), (usize, usize)>,
    /// Prefix KV-cache source pair `(prefix_k, prefix_v)` per kv-cache
    /// extern index (one per layer).
    prefix: HashMap<(u64, u32), (usize, usize)>,
    /// The model's head geometry, minted once at the bounds parse below — one value, with its GQA
    /// grouping already proven, rather than three integers this builder could pair up wrongly.
    geom: Option<ModelAttnGeometry>,
    eps: f32,
    scale: f32,
    m: u32,
    /// Modeled prefix length for the KV-cache `Source` rows (structural;
    /// the host eval / GPU player binds the real length at run time).
    prefix_len: u32,
    /// Parallel to `ops`: the construct each op was expanded from.
    op_expansion: Vec<Option<Expansion>>,
    /// The construct being expanded, while its ops are pushed.
    expanding: Option<ExpansionId>,
    expansions: u32,
}

/// Which config key names the routed experts, and which routing math orders the top-k — the
/// discriminator `moe_block`'s expansion reads off the model bounds. Mirrors the ISel impls'
/// `applies_to` gates (Mixtral / Qwen / GLM-4), so bridge and solver can never disagree on the
/// family a config belongs to.
enum RouterFlavor {
    /// `num_local_experts` — top-k then softmax over the chosen (Mixtral).
    Mixtral,
    /// `num_experts` — softmax over every expert, then top-k (Qwen; renorm per config).
    QwenShared,
    /// `n_routed_experts` — sigmoid + e_score_correction_bias keys pick the top-k, the unbiased
    /// sigmoids gathered as scores (GLM-4 / DeepSeek-V3 `noaux_tc` at `n_group = 1`).
    GlmSigmoid,
}

/// The expert half of a MoE block, which every router shares.
struct Experts {
    experts: NumExperts,
    k: TopK,
    inter: u32,
    hidden: u32,
    quant: ExpertQuant,
    bundle: ExpertBundle,
    body: ExpertBody,
    shared: SharedExpertBound,
}

impl<'a> Builder<'a> {
    fn push_op(&mut self, op: ArchOp, inputs: Vec<InputRef>) -> usize {
        let idx = self.ops.len();
        self.ops.push(OpDesc {
            op,
            m: self.m,
            inputs,
        });
        let guard = None;
        let expansion = self.expanding.map(|id| Expansion { id, guard });
        self.op_expansion.push(expansion);
        idx
    }

    fn push(&mut self, op: ArchOp, inputs: &[usize]) -> usize {
        self.push_op(op, inputs.iter().map(|&i| InputRef::Op(i)).collect())
    }

    /// Open the next construct's expansion: every op pushed until it closes shares its id.
    fn expand(&mut self) {
        self.expanding = Some(ExpansionId(self.expansions));
        self.expansions += 1;
    }

    /// Each row's `k` highest-scoring experts, from the router's `logits`.
    fn route_top_k(&mut self, logits: usize, k: TopK) -> usize {
        let sorted = self.push(SubOp::RouteArgsort, &[logits]);
        self.push(SubOp::RouteTopK { k }, &[sorted])
    }

    /// The expert half: the `(token, expert)` pairs of `x` sorted by expert, projected by
    /// `bank` and activated as the expert body says, projected back, restored to token order,
    /// summed by `scores`. Closes the expansion; returns the sum.
    ///
    /// `bank` resolves one projection's WEIGHT and SCALE sources — the two are separate
    /// launch tensors (a bundle holds both), so the matmul binds them as operands 2 and 3.
    fn experts(
        &mut self,
        x: InputRef,
        indices: usize,
        scores: usize,
        bank: &dyn Fn(ExpertProj) -> (InputRef, InputRef),
        e: Experts,
    ) -> usize {
        let (k, quant, bundle) = (e.k, e.quant, e.bundle);
        let matmul = |proj, n| SubOp::ExpertMatmul {
            proj,
            n,
            k,
            quant,
            bundle,
        };
        let sort = SubOp::ExpertSort {
            experts: e.experts,
            k,
            bundle,
        };
        let pairs = self.push_op(sort, vec![x, InputRef::Op(indices)]);
        let rows =
            |r: usize, w: InputRef, s: InputRef| vec![InputRef::Op(r), InputRef::Op(pairs), w, s];
        let act = match e.body {
            ExpertBody::Gated(act) => {
                let (gw, gs) = bank(ExpertProj::Gate);
                let gate = self.push_op(matmul(ExpertProj::Gate, e.inter), rows(pairs, gw, gs));
                let (uw, us) = bank(ExpertProj::Up);
                let up = self.push_op(matmul(ExpertProj::Up, e.inter), rows(pairs, uw, us));
                self.push(SubOp::ExpertGatedAct { act }, &[gate, up])
            }
            ExpertBody::Ungated(act) => {
                let (uw, us) = bank(ExpertProj::Up);
                let up = self.push_op(matmul(ExpertProj::Up, e.inter), rows(pairs, uw, us));
                self.push(SubOp::ExpertAct { act }, &[up])
            }
        };
        let (dw, ds) = bank(ExpertProj::Down);
        let down = self.push_op(matmul(ExpertProj::Down, e.hidden), rows(act, dw, ds));
        let tokens = self.push(SubOp::ExpertUnsort, &[down, pairs]);
        let (hidden, shared) = (e.hidden, e.shared);
        let sum = self.push(SubOp::ExpertCombine { hidden, shared }, &[tokens, scores]);
        self.expanding = None;
        sum
    }

    fn push_source(&mut self, rows: u32, cols: u32, binding: SourceBinding) -> usize {
        let e = self.sources.len();
        self.sources.push(SourceShape { rows, cols });
        self.bindings.push(binding);
        e
    }

    /// Resolve a Tile / Weight input to an `InputRef`. Externs and
    /// scalars are op-specific and must be handled by the caller.
    fn resolve(&mut self, tile: TileId, inp: &FufInput) -> Result<InputRef, BridgeError> {
        match inp {
            FufInput::Tile { id, slot } => match self.produced.get(&(id.0, *slot)) {
                Some(p) => Ok(p.as_input_ref()),
                None => Err(BridgeError::DanglingInput {
                    tile,
                    dep: *id,
                    slot: *slot,
                }),
            },
            FufInput::Weight { id, index, .. } => {
                let e = self.weight_source(tile, id.0, *index)?;
                Ok(InputRef::Ext(e))
            }
            FufInput::Extern { .. } => Err(BridgeError::MalformedOp {
                tile,
                op: self.fuf.get(tile).op,
                detail: "extern input not valid here",
            }),
            FufInput::Scalar(_) => Err(BridgeError::MalformedOp {
                tile,
                op: self.fuf.get(tile).op,
                detail: "scalar input not valid here",
            }),
        }
    }

    /// Get-or-create the dedup'd source for a weight reference.
    fn weight_source(
        &mut self,
        tile: TileId,
        id: u32,
        index: Option<UnrollIndex>,
    ) -> Result<usize, BridgeError> {
        if let Some(&e) = self.weight_src.get(&(id, index)) {
            return Ok(e);
        }
        let shape = self
            .inferred
            .weights
            .get(&crate::classified::WeightId(id))
            .ok_or(BridgeError::UnresolvedShape {
                tile,
                what: "weight (no inferred shape)",
            })?;
        let dims = eval_shape_with(shape, &self.bounds).ok_or(BridgeError::UnresolvedShape {
            tile,
            what: "weight shape has unresolved dim",
        })?;
        let (rows, cols) = match dims.as_slice() {
            [n] => (1u32, *n as u32),
            [r, c] => (*r as u32, *c as u32),
            // Expert bundles are rank-3 `[experts, n, k]`; the tape
            // source is an opaque bundle — fold the leading dims.
            [e, r, c] => ((*e * *r) as u32, *c as u32),
            other => {
                eprintln!("[to-wavefront] tile {} weight dims = {other:?}", tile.0);
                return Err(BridgeError::UnresolvedShape {
                    tile,
                    what: "weight rank not 1, 2 or 3",
                });
            }
        };
        let e = self.push_source(rows, cols, SourceBinding::Weight { id, index });
        self.weight_src.insert((id, index), e);
        Ok(e)
    }

    /// The single Tile or Weight input matching `pred`, resolved.
    fn input_at(&mut self, tile: TileId, idx: usize) -> Result<InputRef, BridgeError> {
        let inp = self
            .fuf
            .get(tile)
            .inputs
            .get(idx)
            .ok_or(BridgeError::MalformedOp {
                tile,
                op: self.fuf.get(tile).op,
                detail: "missing input at expected index",
            })?
            .clone();
        self.resolve(tile, &inp)
    }

    /// Output dims `(rows, cols)` of one tile slot, resolved to concrete
    /// integers. Rank-1 shapes are treated as `[1, n]`.
    fn out_cols(&self, tile: TileId, slot: usize, what: &'static str) -> Result<u32, BridgeError> {
        let node = self.fuf.get(tile);
        let shape = node
            .outputs
            .get(slot)
            .ok_or(BridgeError::UnresolvedShape { tile, what })?;
        let dims = eval_shape_with(shape, &self.bounds)
            .ok_or(BridgeError::UnresolvedShape { tile, what })?;
        dims.last()
            .copied()
            .map(|c| c as u32)
            .ok_or(BridgeError::UnresolvedShape { tile, what })
    }

    /// Shared cos / sin sources pre-tiled to the rope's full column width
    /// `cols` (`heads * head_dim`), memoized per width. The worker fills them by
    /// tiling the per-position rotary row across heads; the sendnn graph reads
    /// them whole (no in-graph `Concat`-tile, which crashes the partitioner).
    fn cos_sin(&mut self, cols: u32, local: bool) -> (usize, usize) {
        if let Some(cs) = self.cos_sin.get(&(cols, local)) {
            return *cs;
        }
        let cos = self.push_source(self.m, cols, SourceBinding::Cos { local });
        let sin = self.push_source(self.m, cols, SourceBinding::Sin { local });
        self.cos_sin.insert((cols, local), (cos, sin));
        (cos, sin)
    }

    /// The per-layer prefix-KV source pair at the CALLING CLASS's kv width.
    ///
    /// ⭐ THE WIDTH IS AN ARGUMENT, NOT A MODEL FACT. A hybrid model (gemma-4) has TWO attention
    /// classes — its sliding layers run (nkvh=8, hd=256) and its global layers (nkvh=1, hd=512) —
    /// so "the model's kv width" does not exist; each call site passes its own class's, read off
    /// the same class geometry the `AttnDecode` it feeds gets. Memoized per `(layer, width)`:
    /// one layer's K and V caches share a width, and a second call from the SAME class finds its
    /// pair; a wrong pairing would be caught by the SubtileIR shape validator, which refuses a
    /// cache region wider than its source.
    fn prefix_for(&mut self, layer: u64, kvdim: u32) -> (usize, usize) {
        if let Some(pp) = self.prefix.get(&(layer, kvdim)) {
            return *pp;
        }
        let pk = self.push_source(self.prefix_len, kvdim, SourceBinding::PrefixK { layer });
        let pv = self.push_source(self.prefix_len, kvdim, SourceBinding::PrefixV { layer });
        self.prefix.insert((layer, kvdim), (pk, pv));
        (pk, pv)
    }
}

/// The sliding window extent every sliding-attention tile reads — the one
/// source every target already trusts (`model.bounds["sliding_window"]`).
/// A sliding tile on a model with no such bound is a config error, named
/// here so it cannot become a default window that silently attends the
/// wrong span.
fn sliding_window_bound(
    tile: TileId,
    model: &ModelParams,
    op: OpKind,
) -> Result<std::num::NonZeroU32, BridgeError> {
    Ok(
        std::num::NonZeroU32::new(model.bounds.get("sliding_window").copied().ok_or(
            BridgeError::MalformedOp {
                tile,
                op,
                detail: "a sliding-attention tile, but the model config carries no \
                         `sliding_window` bound",
            },
        )? as u32)
        .expect(
            "`sliding_window` = 0 is not a window; NonZeroU32::new returns None \
             only for 0, and a config with 0 is malformed",
        ),
    )
}

/// Pull the kv-cache extern's layer index out of an attention /
/// rope_append input list. Returns the first `Extern{KvCache, Some(i)}`.
fn kv_cache_index(node: &FufNode) -> Option<u64> {
    node.inputs.iter().find_map(|inp| match inp {
        FufInput::Extern {
            kind: ExternKind::KvCache,
            index,
        } => index.map(|i| i.0),
        _ => None,
    })
}

/// A MoE block's expert and top-k counts; a block routing to none is a malformed config.
fn moe_counts(tile: TileId, experts: u32, k: u32) -> Result<(NumExperts, TopK), BridgeError> {
    let nz = std::num::NonZeroU32::new;
    match (nz(experts), nz(k)) {
        (Some(e), Some(k)) => Ok((NumExperts::new(e), TopK::new(k))),
        _ => Err(BridgeError::UnresolvedShape {
            tile,
            what: "a MoE block with zero experts or zero top-k",
        }),
    }
}

/// The resident prefix-KV cache capacity (prefix `Source` rows == length-mask
/// columns) a decode/prefill bundle is baked against. A model's decode and
/// prefill bundles model the SAME on-device cache, so a build threads ONE value
/// to every [`lower_decode_to_wavefront`] call.
///
/// Encoded as a distinct `NonZeroU32` newtype so the capacity can never be a
/// bare placeholder — two states that both produced silent runtime failures are
/// now UNCONSTRUCTIBLE:
///   - `prefix_len = 0` made the SubtileIR prefix `Source` invalid (`NonZeroU32`
///     rules it out at the type level);
///   - a small literal capacity for the PREFILL bundle (the old
///     `else { 1 }`) baked a `[1, 1]` KTIR length mask, so the worker rejected
///     any prompt longer than one token (`positions 0..N exceed bundle prefix
///     capacity 1`). The capacity is no longer an arbitrary `u32` a caller can
///     set to `1` (or to an `m`/row count) — it is a `PrefixCapacity`, and every
///     bundle of one model receives the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixCapacity(std::num::NonZeroU32);

impl PrefixCapacity {
    /// Wrap a proven-nonzero capacity. The caller holds the `NonZeroU32`, so
    /// there is no fallible/`unwrap` boundary here (the zero case is excluded by
    /// the argument type, not by a runtime check).
    pub fn new(rows: std::num::NonZeroU32) -> Self {
        Self(rows)
    }
    /// The prefix `Source` row count == the length-mask column width.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

/// Translate a solved decode FUF into a wavefront `LoweringInput`.
///
/// `asn` is the `num_tokens = 1` (decode) [`Assignment`]; `bounds` is the
/// per-model integer bounds the solver used (tp-sharded if tp>1).
/// `prefix_len` ([`PrefixCapacity`]) models the KV-cache prefix rows for the
/// attention `Source` segments (structural only — the real length binds at run
/// time); the decode and prefill bundles of one model MUST share it.
pub fn lower_decode_to_wavefront(
    fuf: &Fuf,
    // `None` = tape-authoritative mode (M2b step 3): the walk needs no
    // solver assignment — `Some` adds the coverage sanity check the
    // instruction-selection era provided for free.
    asn: Option<&Assignment>,
    inferred: &Inferred,
    bounds: &BTreeMap<String, u64>,
    model: &ModelParams,
    prefix_len: PrefixCapacity,
    num_tokens: u32,
) -> Result<LoweredDecode, BridgeError> {
    // Unwrap the typed capacity to the raw row count once, at the boundary; the
    // rest of the bridge keeps using a plain `u32`.
    let prefix_len = prefix_len.get();
    // The SAME eps resolver instruction selection uses (rms_norm_eps →
    // layer_norm_eps → vision_norm_eps → the 1e-5 default) — one
    // source of truth, always resolvable.
    let eps = crate::codegen::rms_norm_eps(model);
    // Encoder/LayerNorm-class canonicals may omit head_dim; the scale
    // is only consumed at attention ops — default NaN and refuse there.
    let scale = if model.bounds.contains_key("head_dim")
        || model.scalars.contains_key("query_pre_attn_scalar")
    {
        attention_scale_for(model)
    } else {
        f32::NAN
    };
    let mut b = bounds.clone();
    // num_tokens (= query rows m): 1 = decode, >1 = batched prefill. The op row
    // dim threads through OpDesc.m / the EmbeddedHidden + cos_sin sources — NOT
    // the solver Assignment (which only tiles N/K, num_tokens-independent) — so
    // overriding the bound + m here yields a valid m-row graph.
    let m = num_tokens.max(1);
    b.insert("num_tokens".into(), m as u64);
    // ⭐ THE PARSE BOUNDARY from the model's `config.json` to the head geometry as a value. The
    // three numbers are read here, once, and immediately fused into one `ModelAttnGeometry` whose
    // mint divides out the GQA group size and refuses a kv-head count that does not divide the
    // query-head count. Downstream — the tape, the lowering, the device addresses — never sees them
    // apart again, and never re-divides.
    // Attention geometry is OPTIONAL at entry: vision canonicals carry
    // no head_dim and never reach the decode-attention arms (which
    // refuse on None).
    let geom_pair: Option<(ModelAttnGeometry, Option<ModelAttnGeometry>)> =
        if b.contains_key("head_dim") {
            let head_dim = HeadDim::new(b["head_dim"] as u32);
            let num_q_heads = QueryHeads::new(b.get("num_attention_heads").copied().ok_or(
                BridgeError::MissingBound {
                    key: "num_attention_heads",
                },
            )? as u32);
            let num_kv_heads = KvHeads::new(b.get("num_key_value_heads").copied().ok_or(
                BridgeError::MissingBound {
                    key: "num_key_value_heads",
                },
            )? as u32);
            let geom = ModelAttnGeometry::mint(num_q_heads, num_kv_heads, head_dim).ok_or(
                BridgeError::NoHeadGrouping {
                    num_q_heads: num_q_heads.get(),
                    num_kv_heads: num_kv_heads.get(),
                    head_dim: head_dim.get(),
                },
            )?;
            let geom_global = {
                let g_hd = b.get("global_head_dim").copied().map(|v| v as u32);
                let g_kv = b
                    .get("num_global_key_value_heads")
                    .copied()
                    .map(|v| v as u32);
                match (g_hd, g_kv) {
                    (Some(g_hd), Some(g_kv))
                        if g_hd != head_dim.get() || g_kv != num_kv_heads.get() =>
                    {
                        Some(
                            ModelAttnGeometry::mint(
                                num_q_heads,
                                KvHeads::new(g_kv),
                                HeadDim::new(g_hd),
                            )
                            .ok_or(BridgeError::NoHeadGrouping {
                                num_q_heads: num_q_heads.get(),
                                num_kv_heads: g_kv,
                                head_dim: g_hd,
                            })?,
                        )
                    }
                    _ => None,
                }
            };
            Some((geom, geom_global))
        } else {
            None
        };
    let (geom, geom_global) = match geom_pair {
        Some((g, gg)) => (Some(g), gg),
        None => (None, None),
    };
    let mut bx = Builder {
        fuf,
        model,
        gain_offsets: HashMap::new(),
        norm_gain_add_tiles: HashMap::new(),
        geom_global,
        inferred,
        bounds: b,
        sources: Vec::new(),
        bindings: Vec::new(),
        ops: Vec::new(),
        produced: HashMap::new(),
        weight_src: HashMap::new(),
        cos_sin: BTreeMap::new(),
        prefix: HashMap::new(),
        geom,
        eps,
        scale,
        m,
        prefix_len,
        op_expansion: Vec::new(),
        expanding: None,
        expansions: 0,
    };

    // The result is the last value-producing op (the lm_head GEMM).
    let mut result: Option<usize> = None;
    // The op an `export_hidden` names: the hidden state the forward exports.
    let mut exported: Option<usize> = None;

    // Topological walk over `fuf.nodes`. The bridge's resolver looks
    // up each input's producer in `bx.produced`, so a tile must be
    // visited only AFTER every tile-input it references. Metal's
    // post-fusion FUF happens to be stored in ascending-id topo order
    // (which is why the original `for node in &fuf.nodes` loop
    // sufficed), but cuda's fusion pass reorders ids — so we must
    // sort. Standard Kahn: unmet-input count per tile, queue of
    // ready tiles, process in order, decrement successors. Result is
    // a permutation of `fuf.nodes` that respects every tile→tile
    // edge regardless of how fusion happened to assign ids.
    let topo: Vec<TileId> = {
        let n = fuf.nodes.len();
        let mut indeg: Vec<u32> = vec![0; n];
        let mut succs: Vec<Vec<u32>> = vec![Vec::new(); n];
        for node in &fuf.nodes {
            let me = node.id.0 as usize;
            for inp in &node.inputs {
                if let FufInput::Tile { id, .. } = inp {
                    let dep = id.0 as usize;
                    indeg[me] += 1;
                    succs[dep].push(node.id.0);
                }
            }
        }
        let mut queue: std::collections::VecDeque<u32> =
            (0..n as u32).filter(|&i| indeg[i as usize] == 0).collect();
        let mut out: Vec<TileId> = Vec::with_capacity(n);
        while let Some(t) = queue.pop_front() {
            out.push(TileId(t));
            for &s in &succs[t as usize] {
                indeg[s as usize] -= 1;
                if indeg[s as usize] == 0 {
                    queue.push_back(s);
                }
            }
        }
        if out.len() != n {
            // Cycle in the FUF — that's structurally impossible
            // (the FUF is a DAG built from straight-line DSL +
            // unrolled `for`), so surface it loudly.
            return Err(BridgeError::MalformedOp {
                tile: TileId(0),
                op: fuf.nodes[0].op,
                detail: "FUF has a cycle; cannot topologically order",
            });
        }
        out
    };

    for tile_id in topo {
        let node = fuf.get(tile_id);
        let tile = node.id;
        // Every tile must be claimed by the solve — otherwise codegen
        // would have errored. A coverage gap here is a real bug, so
        // surface it rather than lower a tile the solver rejected.
        if asn.is_some_and(|a| a.subgraph_of(tile).is_none()) {
            return Err(BridgeError::MalformedOp {
                tile,
                op: node.op,
                detail: "tile not covered by the assignment",
            });
        }

        match node.op {
            // Embed is a host-side row gather, not a megakernel op: the
            // runtime supplies the embedded hidden state, so the embed
            // tile's output is a read-only graph Source.
            OpKind::Embed => {
                let cols = bx.out_cols(tile, 0, "embed output")?;
                let e = bx.push_source(m, cols, SourceBinding::EmbeddedHidden);
                bx.produced.insert((tile.0, 0), Producer::Ext(e));
            }
            // MmEmbedSplice is the multimodal placeholder splice the
            // tp-lowering pass inserts after every Embed to overwrite
            // placeholder positions with vision-encoder rows. For
            // text-only forwards (no `ctx.embed_patches`) it's a
            // runtime no-op — the megakernel sees its input slot
            // verbatim. Pass the upstream (Embed's output) through to
            // the splice's output slot so downstream tiles read the
            // same `EmbeddedHidden` source. No op is emitted
            // — this op is structurally absent from the megakernel.
            OpKind::MmEmbedSplice => {
                let upstream = bx.input_at(tile, 0)?;
                let producer = match upstream {
                    InputRef::Op(j) => Producer::Op(j),
                    InputRef::Ext(e) => Producer::Ext(e),
                };
                bx.produced.insert((tile.0, 0), producer);
            }
            OpKind::RmsNorm => {
                let x = bx.input_at(tile, 0)?;
                // The gain input may route through a folded `w + scalar`
                // (Gemma's (1+w)); the fold recorded the offset per tile.
                let gain_tile = match node.inputs.get(1) {
                    Some(crate::fuf::FufInput::Tile { id, .. }) => Some(id.0),
                    _ => None,
                };
                // Plus the model's declared zero-centred offset (`rms_norm_zero_centered` on a
                // dense checkpoint, `norm_weight_runtime_offset`): every norm's, carried on the
                // op like a DSL fold's so a fused norm kernel (row program, normed matvec) bakes
                // it as a standalone one does.
                let gain_offset = gain_tile
                    .and_then(|t| bx.gain_offsets.get(&t).copied())
                    .unwrap_or(0.0)
                    + crate::codegen::norm_weight_runtime_offset(bx.model);
                // The (1+w) convention is CARRIED, not dropped and not folded into the weights at
                // load: folding would make the loaded gain disagree with the checkpoint, so every
                // consumer that re-reads it (dumps, the numeric reference, a second target) would
                // see a different tensor.
                let gain = GainConvention::from_offset(gain_offset).ok_or(
                    BridgeError::UnknownGainConvention {
                        tile,
                        offset: gain_offset,
                    },
                )?;
                let w = bx.input_at(tile, 1)?;
                let idx = bx.push_op(SubOp::RmsNorm { eps, gain }, vec![x, w]);
                if gain_offset != 0.0
                    && let Some(t) = gain_tile
                {
                    bx.norm_gain_add_tiles.insert(idx, t);
                }
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Reshape => {
                // A VIEW: `[m, rows_mult*cols]` seen as `[m*rows_mult,
                // cols]` (qwen3 per-head qk-norm). The target shape
                // comes from the tile's inferred output at the walk's
                // bounds; rows must be a whole multiple of m.
                let x = bx.input_at(tile, 0)?;
                let dims = eval_shape_with(&node.outputs[0], &bx.bounds).ok_or(
                    BridgeError::UnresolvedShape {
                        tile,
                        what: "reshape target shape",
                    },
                )?;
                let (rows, cols) = match dims.as_slice() {
                    [r, c] => (*r as u32, *c as u32),
                    [c] => (1u32, *c as u32),
                    _ => {
                        return Err(BridgeError::UnresolvedShape {
                            tile,
                            what: "reshape rank not 1 or 2",
                        });
                    }
                };
                // rows = m * k or m / k: multiplying views (per-head)
                // have m | rows; the patch merger DIVIDES (rows | m).
                let factor = |k: u32| std::num::NonZeroU32::new(k);
                let rows = if rows != 0 && rows.is_multiple_of(bx.m) {
                    factor(rows / bx.m).map(RowScale::Times)
                } else if rows != 0 && bx.m.is_multiple_of(rows) {
                    factor(bx.m / rows).map(RowScale::Over)
                } else {
                    None
                }
                .ok_or(BridgeError::UnresolvedShape {
                    tile,
                    what: "reshape rows neither multiply nor divide num_tokens",
                })?;
                let idx = bx.push_op(SubOp::Reshape { rows, cols }, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::BiasAdd => {
                let x = bx.input_at(tile, 0)?;
                let b = bx.input_at(tile, 1)?;
                let idx = bx.push_op(SubOp::Elementwise(EwKind::BiasAdd), vec![x, b]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Gemm => {
                let (n, k) = gemm_nk_from_fuf(fuf, node, &bx.bounds).ok_or(
                    BridgeError::UnresolvedShape {
                        tile,
                        what: "gemm n/k",
                    },
                )?;
                let act = bx.input_at(tile, 0)?;
                let w = bx.input_at(tile, 1)?;
                let _ = k; // k is now derived from in0_cols at lower-time
                // The weight's quantization scheme is KNOWN here at expansion time (`weight_storage_of`);
                // map it to the typed `GemmWeight` rather than laundering it through a bool. Dense is the
                // only scheme the SDSC/wavefront lowering emits today; fp8 needs the weight_scale wired as
                // a 3rd Gemm input + 1-byte staging (the coupled pass), and int4/ggml are unwired — refuse
                // at BUILD (guard) rather than mis-lower a packed weight as dense fp16.
                let weight = match crate::weight_vocab::weight_storage_of(node) {
                    None | Some(crate::quantization::StorageFormat::Dense) => GemmWeight::Dense,
                    // fp8 W8A8: the arity-3 branch in `lower_subtile_tape_to_superdsc` does the per-token
                    // qfp8ch quant + matmulfp8 (fp8×fp8→fp16) + `a_scale·w_scale` dequant. Feed it the
                    // per-channel `weight_scale` as the 3rd input; the worker loads the 1-byte weight + scale.
                    Some(crate::quantization::StorageFormat::Fp8 { .. }) => GemmWeight::Fp8Dynamic,
                    // MLX-affine packed weights — the format every 4-bit metal checkpoint
                    // ships. `k` is the activation's column count, checked here against the
                    // group so a weight whose groups do not tile K cannot be lowered.
                    Some(crate::quantization::StorageFormat::Affine { bits, group_size }) => {
                        let affine = AffineInt4::mint(*bits, *group_size, k).ok_or(
                            BridgeError::UnsupportedWeight {
                                tile,
                                detail: "affine weight: bits unsupported, or group_size does \
                                         not tile k",
                            },
                        )?;
                        GemmWeight::Affine { affine }
                    }
                    Some(_) => {
                        return Err(BridgeError::UnsupportedWeight {
                            tile,
                            detail: "non-affine int4/ggml weight not yet lowered on the \
                                     SDSC/wavefront path",
                        });
                    }
                };
                // Dense = arity-2 `[act, W]`. fp8 = arity-3 `[act, W_fp8, w_scale]`: the scale sibling
                // reuses the weight's `(id, index)` (worker resolves `{prefix}.weight_scale`), shape `[1,n]`.
                let inputs = match weight {
                    GemmWeight::Fp8Dynamic => {
                        let we = match w {
                            InputRef::Ext(e) => e,
                            _ => {
                                return Err(BridgeError::UnsupportedWeight {
                                    tile,
                                    detail: "fp8 gemm weight input is not an external Weight source",
                                });
                            }
                        };
                        let (wid, widx) = match &bx.bindings[we] {
                            SourceBinding::Weight { id, index } => (*id, *index),
                            _ => {
                                return Err(BridgeError::UnsupportedWeight {
                                    tile,
                                    detail: "fp8 gemm weight source is not a Weight binding",
                                });
                            }
                        };
                        let ws = bx.push_source(
                            1,
                            n,
                            SourceBinding::WeightScale {
                                id: wid,
                                index: widx,
                            },
                        );
                        vec![act, InputRef::Ext(we), InputRef::Ext(ws)]
                    }
                    _ => vec![act, w],
                };
                let idx = bx.push_op(SubOp::MatmulTile { n, weight }, inputs);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // rope_append(q, k, v, positions, rotary, kv_cache[layer]) →
            // (q', k', v): the Q slot rotates in place (`RopeRotate`); the K
            // slot is the GPU cache-write `RopeAppend` (rotate K + write the
            // rotated K / un-rotated V to the paged cache for `layer`), so it
            // takes the un-roped V as a fourth input. The un-roped V (slot 2)
            // still aliases the V-proj output for the attention dataflow edge;
            // the serializer maps attention onto the runtime paged cache, so
            // the new K/V are NOT a separate cache round-trip.
            OpKind::RopeAppend | OpKind::RopeAppendInterleaved => {
                let q = bx.input_at(tile, 0)?;
                let k = bx.input_at(tile, 1)?;
                // slot 2 (v) is the third input's producer, aliased.
                let v_inp = node.inputs.get(2).ok_or(BridgeError::MalformedOp {
                    tile,
                    op: node.op,
                    detail: "rope_append missing v input",
                })?;
                let v = bx.resolve(tile, v_inp)?;
                let layer = kv_cache_index(node).ok_or(BridgeError::MalformedOp {
                    tile,
                    op: node.op,
                    detail: "rope_append missing kv_cache extern index",
                })? as u32;
                // Cos/sin pre-tiled to each rope's full width (GQA: Q and K
                // differ). The worker fills them; the sendnn graph multiplies by
                // them directly (no in-graph Concat-tile -> no partitioner crash).
                let q_cols = bx.out_cols(tile, 0, "rope q cols")?;
                let k_cols = bx.out_cols(tile, 1, "rope k cols")?;
                let local_rope = node.inputs.iter().any(|inp| {
                    matches!(
                        inp,
                        FufInput::Extern {
                            kind: ExternKind::RotaryLocal,
                            ..
                        }
                    )
                });
                // ⭐ THE HEAD DIM IS THE ROTARY CLASS'S OWN, NOT THE MODEL'S. A hybrid-attention
                // arch (gemma-4) binds TWO rotary externs: `rotary_local` names the BASE class
                // (`bx.geom`) and plain `rotary` names the GLOBAL one (`bx.geom_global`, head_dim
                // 512 where the base is 256). This used to stamp `bx.geom.hd()` on every rope
                // node, so a global layer's 512-wide head rotated as two 256-wide heads with a
                // 256-half pairing — a rotation the consumer's own geometry contradicts. The
                // class selection mirrors the attention arm's (`AttnMask::Causal ⇒ geom_global`):
                // the base class never has a global twin (its geometry IS the model's), and a
                // global twin exists exactly when the two classes differ.
                let head_dim = if local_rope {
                    bx.geom
                } else {
                    bx.geom_global.or(bx.geom)
                }
                .ok_or(BridgeError::MissingBound { key: "head_dim" })?
                .hd();
                let (cos_q, sin_q) = bx.cos_sin(q_cols, local_rope);
                let (cos_k, sin_k) = bx.cos_sin(k_cols, local_rope);
                // E.12 — RopeAppend writes rotated K and V into the
                // paged KV cache. Pull the per-layer PrefixK / PrefixV
                // source indices via the cached `prefix_for(layer, k_cols)`
                // helper (same indices Attention will receive later). The
                // WIDTH is this layer's own k-rope width — a hybrid model's
                // classes have different kv widths, so the cache the append
                // writes must be THIS class's.
                let (pk, pv) = bx.prefix_for(layer as u64, k_cols);
                let qi = bx.push_op(
                    SubOp::rope_rotate(head_dim),
                    vec![q, InputRef::Ext(cos_q), InputRef::Ext(sin_q)],
                );
                let pairing = match node.op {
                    OpKind::RopeAppendInterleaved => RopeFormTag::Interleaved,
                    _ => RopeFormTag::NeoX,
                };
                let ki = bx.push_op(
                    // `Causal` by default; the consuming attention arm backpatches
                    // `SlidingWindow` for sliding layers.
                    SubOp::rope_append(head_dim, layer, AttnMask::Causal, pairing),
                    vec![
                        k,
                        InputRef::Ext(cos_k),
                        InputRef::Ext(sin_k),
                        v,
                        InputRef::Ext(pk),
                        InputRef::Ext(pv),
                    ],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(qi));
                bx.produced.insert((tile.0, 1), Producer::Op(ki));
                bx.produced.insert(
                    (tile.0, 2),
                    match v {
                        InputRef::Op(j) => Producer::Op(j),
                        InputRef::Ext(e) => Producer::Ext(e),
                    },
                );
            }
            // kv_append(k, v, kv_cache[layer]) → (k, v): the un-rotated KV
            // writer, `[K, V, K_cache, V_cache]`. Slot 0 is the writer (the
            // attention's new-K edge, as a rope writer's is); V passes
            // through aliasing its projection, as under rope_append.
            OpKind::KvAppend => {
                let head_dim = bx
                    .geom
                    .ok_or(BridgeError::MissingBound { key: "head_dim" })?
                    .hd();
                let k = bx.input_at(tile, 0)?;
                let v = bx.input_at(tile, 1)?;
                let layer = kv_cache_index(node).ok_or(BridgeError::MalformedOp {
                    tile,
                    op: node.op,
                    detail: "kv_append missing kv_cache extern index",
                })? as u32;
                // The prefix cache at THIS layer's own K width — a hybrid
                // model's classes have different kv widths, so the cache the
                // append writes must be THIS class's (the same rule the rope
                // arm's `prefix_for(layer, k_cols)` follows).
                let k_cols = bx.out_cols(tile, 0, "kv_append k cols")?;
                let (pk, pv) = bx.prefix_for(layer as u64, k_cols);
                let ki = bx.push_op(
                    // `Causal` by default; the consuming attention arm backpatches
                    // `SlidingWindow` for sliding layers.
                    SubOp::kv_append(head_dim, layer, AttnMask::Causal),
                    vec![k, v, InputRef::Ext(pk), InputRef::Ext(pv)],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(ki));
                bx.produced.insert(
                    (tile.0, 1),
                    match v {
                        InputRef::Op(j) => Producer::Op(j),
                        InputRef::Ext(e) => Producer::Ext(e),
                    },
                );
            }
            // attention(q', k', v, kv_cache, block_table): AttnDecode
            // reading Q_rot, prefix-cache Source segments, then the new
            // token's (K_rot, V) as Sub edges.
            OpKind::Attention
            | OpKind::SlidingAttention
            | OpKind::SinkAttention
            | OpKind::SinkSlidingAttention => {
                if bx.scale.is_nan() {
                    return Err(BridgeError::MissingBound { key: "head_dim" });
                }
                let is_sink = matches!(
                    node.op,
                    OpKind::SinkAttention | OpKind::SinkSlidingAttention
                );
                let q = bx.input_at(tile, 0)?;
                let k = bx.input_at(tile, 1)?;
                let v = bx.input_at(tile, 2)?;
                // gpt-oss attention sinks: the per-layer `[num_heads]`
                // weight at arg 3 rides to the kernel as a TRAILING
                // weight-source input — an extra softmax column, never
                // a (K, V) segment pair.
                let sinks = if is_sink {
                    let inp = node
                        .inputs
                        .get(3)
                        .cloned()
                        .ok_or(BridgeError::MalformedOp {
                            tile,
                            op: node.op,
                            detail: "sink_attention without a sinks weight input",
                        })?;
                    Some(bx.resolve(tile, &inp)?)
                } else {
                    None
                };
                let Some(layer) = kv_cache_index(node) else {
                    // Cache-less encoder form: attention(q, k, v).
                    if is_sink {
                        return Err(BridgeError::MalformedOp {
                            tile,
                            op: node.op,
                            detail: "sink attention requires a kv_cache — decoder-only op",
                        });
                    }
                    if bx.scale.is_nan() {
                        return Err(BridgeError::MissingBound { key: "head_dim" });
                    }
                    let k = bx.input_at(tile, 1)?;
                    let v = bx.input_at(tile, 2)?;
                    let idx = bx.push_op(
                        SubOp::EncoderAttn {
                            geom: bx
                                .geom
                                .ok_or(BridgeError::MissingBound { key: "head_dim" })?,
                            scale: bx.scale,
                        },
                        vec![q, k, v],
                    );
                    bx.produced.insert((tile.0, 0), Producer::Op(idx));
                    result = Some(idx);
                    continue;
                };
                // valid_len = the modeled prefix-cache rows: the RopeAppend
                // above wrote the new token into the cache at decode_position,
                // so the cache `[prefix_len, kv]` already spans prefix ++ new
                // = `prefix_len` valid positions. The eval_node prefix slice
                // uses `valid_len - 1` (the read-only prefix rows; the new
                // row arrives via the separate k/v segments). The GPU mask
                // binds the real length from the runtime DecodePosition arg.
                let valid_len = bx.prefix_len;
                // ⭐ THE WINDOW SIZE, from the model's own bounds (the
                // helper's doc states the law). A placeholder `1` on the
                // causal arms never escapes the mask construction below.
                let window = if matches!(
                    node.op,
                    OpKind::SlidingAttention | OpKind::SinkSlidingAttention
                ) {
                    sliding_window_bound(tile, bx.model, node.op)?
                } else {
                    // Causal: unused below; a placeholder value that never escapes.
                    std::num::NonZeroU32::new(1).expect("1 is nonzero")
                };
                let mask = match node.op {
                    OpKind::SlidingAttention | OpKind::SinkSlidingAttention => {
                        AttnMask::SlidingWindow { window }
                    }
                    _ => AttnMask::Causal,
                };
                let base_geom = bx
                    .geom
                    .ok_or(BridgeError::MissingBound { key: "head_dim" })?;
                let geom = match mask {
                    AttnMask::Causal => bx.geom_global.unwrap_or(base_geom),
                    AttnMask::SlidingWindow { .. } => base_geom,
                };
                // The prefix cache at THIS layer's class width — a hybrid model's
                // sliding and global layers have different kv widths, and the cache
                // this attention reads is the one its own class's RopeAppend wrote.
                let (pk, pv) = bx.prefix_for(layer, geom.kv_width());
                if let AttnMask::SlidingWindow { window } = mask {
                    // The k-rope for a sliding layer targets the LOCAL
                    // geometry class.
                    if let InputRef::Op(ri) = k
                        && let SubOp::RopeAppend { attn, .. } | SubOp::KvAppend { attn, .. } =
                            &mut bx.ops[ri].op
                    {
                        *attn = AttnMask::SlidingWindow { window };
                    }
                }
                let mut attn_inputs = vec![q, InputRef::Ext(pk), InputRef::Ext(pv), k, v];
                if let Some(s) = sinks {
                    attn_inputs.push(s);
                }
                let idx = bx.push_op(
                    SubOp::attn_decode(geom, scale, valid_len, mask),
                    attn_inputs,
                );
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // rope_query(q, positions, rotary): the query's rotation alone — no K, no cache write.
            OpKind::RopeQuery => {
                let head_dim = bx
                    .geom
                    .ok_or(BridgeError::MissingBound { key: "head_dim" })?
                    .hd();
                let q = bx.input_at(tile, 0)?;
                let q_cols = bx.out_cols(tile, 0, "rope_query q cols")?;
                let local_rope = node.inputs.iter().any(|inp| {
                    matches!(
                        inp,
                        FufInput::Extern {
                            kind: ExternKind::RotaryLocal,
                            ..
                        }
                    )
                });
                let (cos_q, sin_q) = bx.cos_sin(q_cols, local_rope);
                let idx = bx.push_op(
                    SubOp::rope_rotate(head_dim),
                    vec![q, InputRef::Ext(cos_q), InputRef::Ext(sin_q)],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // shared_attention(q, kv_cache[layer], block_table): AttnDecode over the cache's
            // prefix segments alone — the step appends no K/V.
            OpKind::SharedAttention | OpKind::SharedSlidingAttention => {
                if bx.scale.is_nan() {
                    return Err(BridgeError::MissingBound { key: "head_dim" });
                }
                let q = bx.input_at(tile, 0)?;
                let layer = kv_cache_index(node).ok_or(BridgeError::MalformedOp {
                    tile,
                    op: node.op,
                    detail: "shared attention missing kv_cache extern index",
                })?;
                // ⭐ THE WINDOW SIZE — the same bound the attention arm
                // reads (`sliding_window_bound`); a placeholder `1` on the
                // causal arm never escapes the mask construction below.
                let window = if matches!(node.op, OpKind::SharedSlidingAttention) {
                    sliding_window_bound(tile, bx.model, node.op)?
                } else {
                    // Causal: unused below; a placeholder value that never escapes.
                    std::num::NonZeroU32::new(1).expect("1 is nonzero")
                };
                let mask = match node.op {
                    OpKind::SharedSlidingAttention => AttnMask::SlidingWindow { window },
                    _ => AttnMask::Causal,
                };
                let base_geom = bx
                    .geom
                    .ok_or(BridgeError::MissingBound { key: "head_dim" })?;
                let geom = match mask {
                    AttnMask::Causal => bx.geom_global.unwrap_or(base_geom),
                    AttnMask::SlidingWindow { .. } => base_geom,
                };
                // The prefix cache at THIS layer's class width — a hybrid
                // model's sliding and global layers have different kv widths,
                // and the cache this attention reads is its own class's.
                let (pk, pv) = bx.prefix_for(layer, geom.kv_width());
                let idx = bx.push_op(
                    SubOp::attn_decode(geom, bx.scale, bx.prefix_len, mask),
                    vec![q, InputRef::Ext(pk), InputRef::Ext(pv)],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // export_hidden(x): x itself, named the forward's exported hidden state.
            OpKind::ExportHidden => {
                let x = bx.input_at(tile, 0)?;
                let InputRef::Op(j) = x else {
                    return Err(BridgeError::MalformedOp {
                        tile,
                        op: node.op,
                        detail: "export_hidden of a source, not an op's output",
                    });
                };
                bx.produced.insert((tile.0, 0), Producer::Op(j));
                exported = Some(j);
            }
            // capture_hidden(x): x's rows into its layer's block of the rows a drafter reads.
            // Unroll emits a body's nodes in program order, so the captures' ids ascend with their
            // layers: a capture's slot is its rank among them, its layer's place in
            // `draft_capture_layers`.
            OpKind::CaptureHidden => {
                let x = bx.input_at(tile, 0)?;
                let captures: Vec<TileId> = (fuf.nodes.iter())
                    .filter(|n| n.op == OpKind::CaptureHidden)
                    .map(|n| n.id)
                    .collect();
                let declared = (bx.model.layer_sets.get(crate::config::DRAFT_CAPTURE_LAYERS))
                    .map_or(0, |s| s.members().len());
                if captures.len() != declared {
                    return Err(BridgeError::MalformedOp {
                        tile,
                        op: node.op,
                        detail: "capture_hidden ran at a layer count other than \
                                 draft_capture_layers'",
                    });
                }
                let slot = captures.iter().filter(|&&c| c.0 < tile.0).count() as u32;
                let slots = captures.len() as u32;
                let idx = bx.push_op(SubOp::CaptureHidden { slot, slots }, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // draft_conv(x, dynamic, base[, residual]): one stage of Splash's dynamic convolution.
            OpKind::DraftConv => {
                let stage = match node.inputs.len() {
                    3 => DraftConvStage::Prepare,
                    _ => DraftConvStage::Residual,
                };
                let inputs = (0..node.inputs.len())
                    .map(|i| bx.input_at(tile, i))
                    .collect::<Result<Vec<_>, _>>()?;
                let idx = bx.push_op(SubOp::DraftConv { stage }, inputs);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // draft_attention(qkv, q_norm, k_norm, positions) and
            // draft_context_kv(kv, k_norm, positions): the drafter's ring ops at the layer their
            // norms index. Their kernels rotate at each row's position from the model's rotary
            // base.
            OpKind::DraftAttention | OpKind::DraftContextKv => {
                let layer = node
                    .inputs
                    .iter()
                    .find_map(|inp| match inp {
                        FufInput::Weight {
                            index: Some(UnrollIndex(l)),
                            ..
                        } => Some(*l as u32),
                        _ => None,
                    })
                    .ok_or(BridgeError::MalformedOp {
                        tile,
                        op: node.op,
                        detail: "draft ring op without a layer-indexed norm",
                    })?;
                let norms = match node.op {
                    OpKind::DraftAttention => 1..3,
                    _ => 1..2,
                };
                let mut inputs = vec![bx.input_at(tile, 0)?];
                for i in norms {
                    inputs.push(bx.input_at(tile, i)?);
                }
                let op = match node.op {
                    OpKind::DraftAttention => SubOp::DraftAttention {
                        layer,
                        width: bx.out_cols(tile, 0, "draft attention cols")?,
                    },
                    _ => SubOp::DraftContextKv { layer },
                };
                let idx = bx.push_op(op, inputs);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // draft_window_store(context, positions) and draft_window_load(context, positions):
            // the drafter's context window. Their positions are the forward's, a runtime binding.
            OpKind::DraftWindowStore | OpKind::DraftWindowLoad => {
                let context = bx.input_at(tile, 0)?;
                let op = match node.op {
                    OpKind::DraftWindowStore => SubOp::DraftWindowStore,
                    _ => SubOp::DraftWindowLoad,
                };
                let idx = bx.push_op(op, vec![context]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            // draft_select(logits, selector, predecessor_codebook, successor_codebook).
            OpKind::DraftSelect => {
                let inputs = (0..4)
                    .map(|i| bx.input_at(tile, i))
                    .collect::<Result<Vec<_>, _>>()?;
                let idx = bx.push_op(SubOp::DraftSelect, inputs);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Silu => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(SubOp::Elementwise(EwKind::Silu), vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Gelu => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(SubOp::Elementwise(EwKind::Gelu), vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::RmsNormUnit => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(SubOp::RmsNormUnit { eps }, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::ScalarWeightMul => {
                let x = bx.input_at(tile, 0)?;
                let w = node
                    .inputs
                    .iter()
                    .find_map(|inp| match inp {
                        FufInput::Weight { .. } => Some(bx.resolve(tile, inp)),
                        _ => None,
                    })
                    .ok_or(BridgeError::MalformedOp {
                        tile,
                        op: OpKind::ScalarWeightMul,
                        detail: "scalar_weight_mul without a weight input",
                    })??;
                let idx = bx.push_op(SubOp::ScalarWeightMul, vec![x, w]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Moe => {
                // The router's rows, and the experts' — a latent MoE's own (`moe_block(x,
                // latent, moe[layer])`), else the router's.
                let x = bx.input_at(tile, 0)?;
                let tiles = node
                    .inputs
                    .iter()
                    .filter(|i| matches!(i, FufInput::Tile { .. }));
                let expert_x = match tiles.count() {
                    2 => bx.input_at(tile, 1)?,
                    _ => x,
                };
                // The bundle is a STRUCT weight (`mlp[layer]` — router +
                // experts + optional shared under one base), not a
                // tensor: it has no inferred shape, so it gets an opaque
                // 1x1 source carrying only its binding identity.
                let w = node
                    .inputs
                    .iter()
                    .find_map(|inp| match inp {
                        FufInput::Weight { id, index, .. } => Some((id.0, *index)),
                        _ => None,
                    })
                    .map(|(id, index)| {
                        InputRef::Ext(bx.push_source(1, 1, SourceBinding::Weight { id, index }))
                    })
                    .ok_or(BridgeError::MalformedOp {
                        tile,
                        op: OpKind::Moe,
                        detail: "moe_block without a weight input",
                    })?;
                let b = &bx.bounds;
                // Variant discriminators mirror the metal impls'
                // applies_to: Mixtral = num_local_experts; Qwen-shared =
                // num_experts; GLM-4 = n_routed_experts with Qwen-style
                // on-disk naming (switch_mlp + shared_experts), routed
                // sigmoid+bias (DeepSeek's noaux_tc at n_group=1).
                let (router_flavor, num_experts) = if let Some(&n) = b.get("num_local_experts") {
                    (RouterFlavor::Mixtral, n as u32)
                } else if let Some(&n) = b.get("n_routed_experts") {
                    (RouterFlavor::GlmSigmoid, n as u32)
                } else if let Some(&n) = b.get("num_experts") {
                    (RouterFlavor::QwenShared, n as u32)
                } else {
                    return Err(BridgeError::MissingBound { key: "num_experts" });
                };
                let top_k =
                    b.get("num_experts_per_tok")
                        .copied()
                        .ok_or(BridgeError::MissingBound {
                            key: "num_experts_per_tok",
                        })? as u32;
                let moe_inter = b
                    .get("moe_intermediate_size")
                    .or_else(|| b.get("intermediate_size"))
                    .copied()
                    .ok_or(BridgeError::MissingBound {
                        key: "moe_intermediate_size",
                    })? as u32;
                let shared_inter = b
                    .get("shared_expert_intermediate_size")
                    .copied()
                    .unwrap_or(0) as u32;
                let norm_topk = b.get("norm_topk_prob").copied().unwrap_or(0) != 0;
                // The experts' body is the config's declared activation's: never a default the
                // checkpoint did not state (a relu² bank read as gated SiLU computes garbage).
                let body = bx.model.expert_activation.body().map_err(|act| {
                    let act = act.to_string();
                    BridgeError::UnrealizedExpertActivation { tile, act }
                })?;
                // noaux_tc's `routed_scaling_factor` multiplies the renormalized scores last
                // (mlx `group_expert_select`); 1.0 (GLM-4.5) is the identity and emits nothing.
                let routed_scale = bx
                    .model
                    .scalars
                    .get("routed_scaling_factor")
                    .map(|&f| f as f32)
                    .filter(|&f| f != 1.0);
                let (group_size, bits) = match crate::weight_vocab::weight_storage_of(node) {
                    Some(crate::quantization::StorageFormat::Affine { group_size, bits }) => {
                        (*group_size, *bits)
                    }
                    _ => {
                        return Err(BridgeError::NoMetalRealization {
                            tile,
                            detail: "MoE without affine expert storage has no metal \
                                     realization (dense MoE is unclaimed on metal)",
                        });
                    }
                };
                let (experts, k) = moe_counts(tile, num_experts, top_k)?;
                // Qwen order: softmax over every expert, then top-k (renormalised when the
                // config says so); Mixtral: top-k, then softmax over the chosen; GLM-4 /
                // Nemotron-H: sigmoid+bias keys pick the top-k, the UNBIASED sigmoids gathered,
                // renormalised and scaled by the routed scaling factor.
                let (router, bundle) = match router_flavor {
                    RouterFlavor::QwenShared | RouterFlavor::GlmSigmoid => {
                        (RouterBundle::SharedFused, ExpertBundle::SharedFused)
                    }
                    RouterFlavor::Mixtral => (RouterBundle::Fused, ExpertBundle::Fused),
                };
                bx.expand();
                let lg = bx.push_op(SubOp::RouterLogits { experts, router }, vec![x, w]);
                let shared = std::num::NonZeroU32::new(shared_inter)
                    .filter(|_| !matches!(router_flavor, RouterFlavor::Mixtral));
                let lg = match router_flavor {
                    RouterFlavor::QwenShared => bx.push(SubOp::RouteSoftmax, &[lg]),
                    RouterFlavor::GlmSigmoid => bx.push_op(
                        SubOp::RouteSigmoidBias { router },
                        vec![InputRef::Op(lg), w],
                    ),
                    RouterFlavor::Mixtral => lg,
                };
                let indices = bx.route_top_k(lg, k);
                let scores = bx.push(SubOp::RouteGatherScores, &[lg, indices]);
                let scores = match (router_flavor, norm_topk) {
                    (RouterFlavor::QwenShared, true) => bx.push(SubOp::RouteRenorm, &[scores]),
                    (RouterFlavor::QwenShared, false) => scores,
                    (RouterFlavor::GlmSigmoid, norm) => {
                        let scores = match norm {
                            true => bx.push(SubOp::RouteRenorm, &[scores]),
                            false => scores,
                        };
                        match routed_scale {
                            Some(scale) => bx.push(SubOp::RouteScale { scale }, &[scores]),
                            None => scores,
                        }
                    }
                    (RouterFlavor::Mixtral, _) => bx.push(SubOp::RouteSoftmax, &[scores]),
                };
                let hidden = bx.out_cols(tile, 0, "moe_block output")?;
                let experts = Experts {
                    experts,
                    k,
                    inter: moe_inter,
                    hidden,
                    quant: ExpertQuant::declared(group_size, bits),
                    bundle,
                    body,
                    shared: SharedExpertBound(shared),
                };
                // The fused MoE bundle is ONE source (its affine scales resolve by tensor
                // role at the target, not as separate operands), so both of a projection's
                // weight operands bind it.
                let bank = &|_proj: ExpertProj| (w, w);
                let idx = bx.experts(expert_x, indices, scores, bank, experts);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GateSplit => {
                let x = bx.input_at(tile, 0)?;
                let cols = bx.out_cols(tile, 0, "gate_split q cols")?;
                let idx = bx.push_op(SubOp::GateSplit { half_cols: cols }, vec![x]);
                // TWO outputs: slot 0 = q, slot 1 = gate.
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                bx.produced.insert((tile.0, 1), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GateApply => {
                let attn = bx.input_at(tile, 0)?;
                let gate = bx.input_at(tile, 1)?;
                let idx = bx.push_op(SubOp::GateApply, vec![attn, gate]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Concat => {
                let a = bx.input_at(tile, 0)?;
                let b = bx.input_at(tile, 1)?;
                let cols = bx.out_cols(tile, 0, "concat cols")?;
                let idx = bx.push_op(SubOp::Concat { cols }, vec![a, b]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GateScale => {
                let routed = bx.input_at(tile, 0)?;
                let shared = bx.input_at(tile, 1)?;
                let g = bx.input_at(tile, 2)?;
                let idx = bx.push_op(SubOp::GateScale, vec![routed, shared, g]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GatedDeltaNet | OpKind::Mamba2 => {
                let (op, mut inputs) = if node.op == OpKind::Mamba2 {
                    let inner = bx.out_cols(tile, 0, "mamba2 d_inner")?;
                    (SubOp::Mamba2 { inner }, vec![bx.input_at(tile, 0)?])
                } else {
                    let ins = (0..4).map(|i| bx.input_at(tile, i));
                    (SubOp::GatedDeltaNet, ins.collect::<Result<Vec<_>, _>>()?)
                };
                let w = node
                    .inputs
                    .iter()
                    .find_map(|inp| match inp {
                        FufInput::Weight { id, index, .. } => Some((id.0, *index)),
                        _ => None,
                    })
                    .map(|(id, index)| {
                        InputRef::Ext(bx.push_source(1, 1, SourceBinding::Weight { id, index }))
                    })
                    .ok_or(BridgeError::MalformedOp {
                        tile,
                        op: node.op,
                        detail: "recurrent mixer without a weight bundle",
                    })?;
                inputs.push(w);
                let idx = bx.push_op(op, inputs);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GemmaMoe => {
                let router_in = bx.input_at(tile, 0)?;
                let expert_in = bx.input_at(tile, 1)?;
                // TWO opaque struct bundles: router + switch_glu experts,
                // in DSL argument order.
                let mut bundles = node.inputs.iter().filter_map(|inp| match inp {
                    FufInput::Weight {
                        id, index, storage, ..
                    } => Some((id.0, *index, storage.clone())),
                    _ => None,
                });
                let (rw, ew) = match (bundles.next(), bundles.next()) {
                    (Some(r), Some(e)) => (r, e),
                    _ => {
                        return Err(BridgeError::MalformedOp {
                            tile,
                            op: OpKind::GemmaMoe,
                            detail: "gemma_moe needs router + experts weight bundles",
                        });
                    }
                };
                // ⭐ ONE SOURCE PER BUNDLE TENSOR, NOT ONE PER BUNDLE. A MoE bundle is ONE
                // `Weights` field (one accessor, one load) holding SEVERAL tensors, and a
                // launch source must carry exactly one staged tensor — so the block binds as
                // many sources as it has tensors, each naming the bundle's `(id, index)` plus
                // WHICH tensor (`BundleTensor`). The single-tensor `Weight` binding this arm
                // used to push had shape `[1,1]` and bound NOTHING a worker could stage, which
                // is why the spyre weight-bindings walk refused the kind.
                //
                // Shapes are the STACKED ON-DISK extents (the expert axis folded into rows,
                // the same fold `weight_source` gives a rank-3 `[E, r, c]`), stated from the
                // model bounds the block already read. The router's projection is FUF-layout
                // `[k=hidden, n=E]` like every other gemm weight; its two gains are rows of a
                // `[1, ·]` vector like every other norm gain.
                let bundle_src = |bx: &mut Builder,
                                  id: u32,
                                  index: Option<UnrollIndex>,
                                  tensor: BundleTensor,
                                  rows: u32,
                                  cols: u32| {
                    bx.push_source(
                        rows,
                        cols,
                        SourceBinding::BundleTensor { id, index, tensor },
                    )
                };
                let router_src = |bx: &mut Builder, tensor: BundleTensor, rows: u32, cols: u32| {
                    bundle_src(bx, rw.0, rw.1, tensor, rows, cols)
                };
                let expert_src = |bx: &mut Builder, tensor: BundleTensor, rows: u32, cols: u32| {
                    bundle_src(bx, ew.0, ew.1, tensor, rows, cols)
                };
                let b = &bx.bounds;
                let num_experts = b
                    .get("num_experts")
                    .or_else(|| b.get("num_local_experts"))
                    .copied()
                    .ok_or(BridgeError::MissingBound { key: "num_experts" })?
                    as u32;
                let top_k =
                    b.get("num_experts_per_tok")
                        .copied()
                        .ok_or(BridgeError::MissingBound {
                            key: "num_experts_per_tok",
                        })? as u32;
                let moe_inter = b
                    .get("moe_intermediate_size")
                    .or_else(|| b.get("intermediate_size"))
                    .copied()
                    .ok_or(BridgeError::MissingBound {
                        key: "moe_intermediate_size",
                    })? as u32;
                // The EXPERT bundle's storage — the SECOND weight edge in
                // DSL order (router first, switch_glu second); mixed-bit
                // presets quantize the router differently. Mirrors the
                // instruction-selection helper's experts-edge rule, with
                // its same (64, 4) fallback.
                let (group_size, bits) = match ew.2 {
                    crate::quantization::StorageFormat::Affine { group_size, bits } => {
                        (group_size, bits)
                    }
                    _ => (64, 4),
                };
                // Routes off its own pre-norm of `router_in`: logits → top-k → scores at a
                // `hidden^-0.5` temperature → softmax → × per-expert scale. GeGLU experts over
                // `expert_in`.
                let (experts, k) = moe_counts(tile, num_experts, top_k)?;
                let hidden = bx.out_cols(tile, 0, "gemma_moe output")?;
                let gemma = RouterBundle::Gemma;
                // The bundle's tensors as launch sources, in block order: the router's
                // projection (FUF `[k=hidden, n=E]`), its two gains, then each expert
                // projection's codes and per-channel scale (stacked on-disk extents
                // `[E·out, in]` / `[E·out, 1]`).
                let e_rows = num_experts;
                let r_gate = router_src(&mut bx, BundleTensor::RouterGate, hidden, e_rows);
                let r_scale = router_src(&mut bx, BundleTensor::RouterScale, 1, hidden);
                let r_per_expert =
                    router_src(&mut bx, BundleTensor::RouterPerExpertScale, 1, e_rows);
                let w_src = |bx: &mut Builder, t: BundleTensor, out: u32, inp: u32| {
                    expert_src(bx, t, e_rows * out, inp)
                };
                let s_src = |bx: &mut Builder, t: BundleTensor, out: u32| {
                    expert_src(bx, t, e_rows * out, 1)
                };
                let gate_w = w_src(&mut bx, BundleTensor::ExpertGateW, moe_inter, hidden);
                let gate_s = s_src(&mut bx, BundleTensor::ExpertGateS, moe_inter);
                let up_w = w_src(&mut bx, BundleTensor::ExpertUpW, moe_inter, hidden);
                let up_s = s_src(&mut bx, BundleTensor::ExpertUpS, moe_inter);
                let down_w = w_src(&mut bx, BundleTensor::ExpertDownW, hidden, moe_inter);
                let down_s = s_src(&mut bx, BundleTensor::ExpertDownS, hidden);
                bx.expand();
                let norm = SubOp::RouterNorm { eps, router: gemma };
                let xr = bx.push_op(norm, vec![router_in, InputRef::Ext(r_scale)]);
                let logits = SubOp::RouterLogits {
                    experts,
                    router: gemma,
                };
                let lg = bx.push_op(logits, vec![InputRef::Op(xr), InputRef::Ext(r_gate)]);
                let indices = bx.route_top_k(lg, k);
                let scores = bx.push(SubOp::RouteGatherScores, &[lg, indices]);
                let scale = (hidden as f32).powf(-0.5);
                let scores = bx.push(SubOp::RouteScale { scale }, &[scores]);
                let scores = bx.push(SubOp::RouteSoftmax, &[scores]);
                let scale = SubOp::RouteExpertScale { router: gemma };
                let (s, i) = (InputRef::Op(scores), InputRef::Op(indices));
                let scores = bx.push_op(scale, vec![s, i, InputRef::Ext(r_per_expert)]);
                let experts = Experts {
                    experts,
                    k,
                    inter: moe_inter,
                    hidden,
                    quant: ExpertQuant::declared(group_size, bits),
                    bundle: ExpertBundle::SwitchGlu,
                    body: ExpertBody::Gated(GatedAct::Gelu),
                    shared: SharedExpertBound(None),
                };
                let bank = &|proj: ExpertProj| {
                    let (w, s) = match proj {
                        ExpertProj::Gate => (gate_w, gate_s),
                        ExpertProj::Up => (up_w, up_s),
                        ExpertProj::Down => (down_w, down_s),
                    };
                    (InputRef::Ext(w), InputRef::Ext(s))
                };
                let idx = bx.experts(expert_in, indices, scores, bank, experts);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GptOssMoe => {
                let x = bx.input_at(tile, 0)?;
                // One opaque STRUCT bundle (`mlp[layer]` — router gate +
                // router bias + experts + expert linear biases under one
                // base), like `moe_block`.
                let w = node
                    .inputs
                    .iter()
                    .find_map(|inp| match inp {
                        FufInput::Weight { id, index, .. } => Some((id.0, *index)),
                        _ => None,
                    })
                    .map(|(id, index)| {
                        InputRef::Ext(bx.push_source(1, 1, SourceBinding::Weight { id, index }))
                    })
                    .ok_or(BridgeError::MalformedOp {
                        tile,
                        op: OpKind::GptOssMoe,
                        detail: "gptoss_moe without a weight input",
                    })?;
                let b = &bx.bounds;
                let num_experts =
                    b.get("num_local_experts")
                        .copied()
                        .ok_or(BridgeError::MissingBound {
                            key: "num_local_experts",
                        })? as u32;
                let top_k =
                    b.get("num_experts_per_tok")
                        .copied()
                        .ok_or(BridgeError::MissingBound {
                            key: "num_experts_per_tok",
                        })? as u32;
                let moe_inter = b
                    .get("moe_intermediate_size")
                    .or_else(|| b.get("intermediate_size"))
                    .copied()
                    .ok_or(BridgeError::MissingBound {
                        key: "moe_intermediate_size",
                    })? as u32;
                let (group_size, bits) = match crate::weight_vocab::weight_storage_of(node) {
                    Some(crate::quantization::StorageFormat::Affine { group_size, bits }) => {
                        (*group_size, *bits)
                    }
                    _ => {
                        return Err(BridgeError::NoMetalRealization {
                            tile,
                            detail: "gptoss_moe without affine expert storage has no metal \
                                     realization (dense MoE is unclaimed on metal)",
                        });
                    }
                };
                // The SwiGLU-OAI limit is BAKED as 7.0 into the kernels
                // (`swiglu_oai_mul_f`, the GATED_ACT==2 epilogue); gpt-oss is
                // the only consumer, so a config carrying any other limit is
                // refused here at build time rather than silently computed
                // with the wrong clamp. The upgrade path if a second consumer
                // appears is a `CanonicalParams` scalar.
                let limit = bx.model.scalars.get("swiglu_limit").copied().ok_or(
                    BridgeError::MalformedOp {
                        tile,
                        op: OpKind::GptOssMoe,
                        detail: "gptoss_moe requires a `swiglu_limit` config scalar",
                    },
                )?;
                if limit != 7.0 {
                    return Err(BridgeError::MalformedOp {
                        tile,
                        op: OpKind::GptOssMoe,
                        detail: "gptoss_moe kernels bake swiglu_limit = 7.0; this config \
                                 carries another value",
                    });
                }
                let (experts, k) = moe_counts(tile, num_experts, top_k)?;
                let router = RouterBundle::GptOss;
                bx.expand();
                // gpt-oss order: linear router WITH bias → top-4 over the
                // biased logits → the biased values read back as the gathered
                // scores → softmax over the top-4 (no renorm) — the Mixtral
                // arm's exact post-gather sequence.
                let lg = bx.push_op(SubOp::RouterLogits { experts, router }, vec![x, w]);
                let lg = bx.push_op(SubOp::RouteBias { router }, vec![InputRef::Op(lg), w]);
                let indices = bx.route_top_k(lg, k);
                let scores = bx.push(SubOp::RouteGatherScores, &[lg, indices]);
                let scores = bx.push(SubOp::RouteSoftmax, &[scores]);
                let hidden = bx.out_cols(tile, 0, "gptoss_moe output")?;
                let experts = Experts {
                    experts,
                    k,
                    inter: moe_inter,
                    hidden,
                    quant: ExpertQuant::declared(group_size, bits),
                    bundle: ExpertBundle::GptOss,
                    body: ExpertBody::Gated(GatedAct::SwigluOai),
                    shared: SharedExpertBound(None),
                };
                // The fused MoE bundle is ONE source (its affine scales resolve by tensor
                // role at the target, not as separate operands), so both of a projection's
                // weight operands bind it — the `moe_block` arm's own law.
                let bank = &|_proj: ExpertProj| (w, w);
                let idx = bx.experts(x, indices, scores, bank, experts);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::TanhSoftCap => {
                let x = bx.input_at(tile, 0)?;
                // The cap is a MODEL CONSTANT (`final_logit_softcapping`, HF
                // naming) — the only models that emit a `tanh_softcap(...)` tile
                // are the capping ones, so it is present whenever this arm runs.
                // Read at the same site every other model-const scalar is (the
                // parsed `scalars` table, which hoists nested `text_config`),
                // and carried on the op so the tape holds it — metal/cuda bake
                // it into kernels and spyre binds it as a registry const.
                let cap = bx
                    .model
                    .scalars
                    .get("final_logit_softcapping")
                    .copied()
                    .ok_or(BridgeError::MissingScalar {
                        key: "final_logit_softcapping",
                    })? as f32;
                let idx = bx.push_op(SubOp::TanhSoftCap { cap }, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Mul => {
                // Granite (and Gemma's embed scale) multiply an activation by
                // a compile-time constant: `x * scalar(c)`. The FUF carries
                // the constant as a bare `FufInput::Scalar`, which resolves to
                // no tile and no source — so a scalar operand collapses the
                // binary `Mul` into a unary `ScalarMul { scale }` over the
                // tensor operand (mirrors the interpreter IR's
                // `Instruction::ScalarMul(_, _, f32)`). A plain tensor×tensor
                // `Mul` (e.g. SwiGLU gate·up) keeps both operands.
                let (scale, tensor_idx) = {
                    let inputs = &bx.fuf.get(tile).inputs;
                    let scale = inputs.iter().find_map(|i| match i {
                        // FUF scalars are f64; the lowered op / sengraph const
                        // are f32 / f16, matching `ir::Instruction::ScalarMul`.
                        FufInput::Scalar(v) => Some(*v as f32),
                        _ => None,
                    });
                    let tensor_idx = inputs
                        .iter()
                        .position(|i| !matches!(i, FufInput::Scalar(_)));
                    (scale, tensor_idx)
                };
                match scale {
                    Some(scale) => {
                        let tensor_idx = tensor_idx.ok_or(BridgeError::MalformedOp {
                            tile,
                            op: OpKind::Mul,
                            detail: "scalar * scalar Mul should have been \
                                     constant-folded before lowering",
                        })?;
                        let x = bx.input_at(tile, tensor_idx)?;
                        let idx = bx.push_op(SubOp::ScalarMul { scale }, vec![x]);
                        bx.produced.insert((tile.0, 0), Producer::Op(idx));
                        result = Some(idx);
                    }
                    None => {
                        let a = bx.input_at(tile, 0)?;
                        let b2 = bx.input_at(tile, 1)?;
                        let idx = bx.push_op(SubOp::Elementwise(EwKind::Mul), vec![a, b2]);
                        bx.produced.insert((tile.0, 0), Producer::Op(idx));
                        result = Some(idx);
                    }
                }
            }
            OpKind::LoadRows(source) => {
                // A synthesized source tile whose width the FUF already carries.
                let width = bx.out_cols(tile, 0, "host-staged rows cols")?;
                let idx = bx.push_op(SubOp::LoadRows { source, width }, vec![]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::VisionRope => {
                let q = bx.input_at(tile, 0)?;
                let k = bx.input_at(tile, 1)?;
                let idx = bx.push_op(SubOp::VisionRope, vec![q, k]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                bx.produced.insert((tile.0, 1), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::QuickGelu => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(SubOp::Elementwise(EwKind::QuickGelu), vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GeluErf => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(SubOp::Elementwise(EwKind::GeluErf), vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Relu2 => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(SubOp::Elementwise(EwKind::Relu2), vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::EmbeddingGather => {
                // The index table is a runtime input (an extern), not a
                // tape operand: which table it is IS the op's identity,
                // exactly like VarlenAttention's cu_seqlens kind below.
                let x = bx.input_at(tile, 0)?;
                let indices_kind: u8 = node
                    .inputs
                    .iter()
                    .find_map(|inp| match inp {
                        FufInput::Extern { kind, .. } => match kind {
                            ExternKind::WindowIndex => Some(0),
                            ExternKind::ReverseIndices => Some(1),
                            _ => None,
                        },
                        _ => None,
                    })
                    .ok_or(BridgeError::MalformedOp {
                        tile,
                        op: OpKind::EmbeddingGather,
                        detail: "embedding_gather without a window/reverse index extern",
                    })?;
                let idx = bx.push_op(SubOp::EmbeddingGather { indices_kind }, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::VarlenAttention => {
                let q = bx.input_at(tile, 0)?;
                let k = bx.input_at(tile, 1)?;
                let v = bx.input_at(tile, 2)?;
                let cu_kind: u8 = node
                    .inputs
                    .iter()
                    .find_map(|inp| match inp {
                        FufInput::Extern { kind, .. } => match kind {
                            ExternKind::CuSeqlens => Some(0),
                            ExternKind::CuSeqlensFull => Some(1),
                            ExternKind::CuSeqlensWindow => Some(2),
                            _ => None,
                        },
                        _ => None,
                    })
                    .ok_or(BridgeError::MalformedOp {
                        tile,
                        op: OpKind::VarlenAttention,
                        detail: "varlen attention without a cu_seqlens extern",
                    })?;
                let idx = bx.push_op(SubOp::VarlenAttention { cu_kind }, vec![q, k, v]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Mean => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(SubOp::Mean, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Sub => {
                let a = bx.input_at(tile, 0)?;
                let b2 = bx.input_at(tile, 1)?;
                let idx = bx.push_op(SubOp::Elementwise(EwKind::Sub), vec![a, b2]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Add => {
                // Gemma's `(1 + w)` norm-gain convention rides the DSL
                // as `Add(weight, scalar)`: the Add tile ALIASES its
                // weight input on the tape — nothing is emitted — and the
                // consuming rmsnorm carries the offset in its gain
                // convention (the `RmsNorm` arm).
                let scalar = node.inputs.iter().find_map(|i| match i {
                    crate::fuf::FufInput::Scalar(v) => Some(*v),
                    _ => None,
                });
                if let Some(v) = scalar {
                    let declared = crate::codegen::norm_weight_runtime_offset(bx.model) as f64;
                    // The declared offset the `RmsNorm` arm adds to every
                    // norm → 0 here, so it counts once. Any other offset is
                    // DATA the consuming rmsnorm carries (Gemma's
                    // per-instruction offset vocabulary).
                    let tape_offset = if (v - declared).abs() <= f64::EPSILON {
                        0.0
                    } else {
                        v as f32
                    };
                    bx.gain_offsets.insert(tile.0, tape_offset);
                    let w = node
                        .inputs
                        .iter()
                        .find(|i| !matches!(i, crate::fuf::FufInput::Scalar(_)))
                        .ok_or(BridgeError::MalformedOp {
                            tile,
                            op: OpKind::Add,
                            detail: "scalar Add with no tensor operand",
                        })?;
                    let aliased = bx.resolve(tile, w)?;
                    let producer = match aliased {
                        InputRef::Op(i) => Producer::Op(i),
                        InputRef::Ext(e) => Producer::Ext(e),
                    };
                    bx.produced.insert((tile.0, 0), producer);
                } else {
                    let a = bx.input_at(tile, 0)?;
                    let b2 = bx.input_at(tile, 1)?;
                    let idx = bx.push_op(SubOp::Elementwise(EwKind::Add), vec![a, b2]);
                    bx.produced.insert((tile.0, 0), Producer::Op(idx));
                    result = Some(idx);
                }
            }
            other => return Err(BridgeError::UnsupportedOp { tile, op: other }),
        }
    }

    let result = result.ok_or(BridgeError::NoResult)?;
    debug_assert_eq!(bx.sources.len(), bx.bindings.len());
    // Invert the producer map into per-op tile provenance. An op can
    // appear under several (tile, slot) keys — its own tile AND alias
    // slots on downstream tiles (the rope's un-roped V aliases the
    // V-proj output). The op's OWN tile is the smallest key; alias
    // keys must not claim provenance (HashMap iteration order made the
    // V gemm rank as the rope's tile and broke the wave-order sort).
    let mut op_tiles: Vec<Option<(u32, u8)>> = vec![None; bx.ops.len()];
    for (&(tile, slot), producer) in &bx.produced {
        if let Producer::Op(idx) = producer {
            match op_tiles[*idx] {
                None => op_tiles[*idx] = Some((tile, slot)),
                Some(cur) if (tile, slot) < cur => op_tiles[*idx] = Some((tile, slot)),
                _ => {}
            }
        }
    }
    Ok(LoweredDecode {
        input: LoweringInput {
            sources: bx.sources,
            ops: bx.ops,
            result,
            exported,
        },
        bindings: bx.bindings,
        op_tiles,
        norm_gain_add_tiles: bx.norm_gain_add_tiles,
        op_expansion: bx.op_expansion,
        unnamed_reads: Vec::new(),
    })
}

/// A per-arch weight locator recovered from a lowered decode bucket's
/// `weight_slots`: the `(bucket, op_idx, slot)` triple the runtime
/// `WeightAccessors::<kind>_at` match table keys on, plus the [`WeightKind`]
/// selecting the accessor bundle. The unrolled `layer` is supplied
/// per-source (it is the FUF's former loop-var), so this is layer-agnostic.
#[derive(Clone, Debug)]
pub struct WeightLocInfo {
    pub bucket: u32,
    pub op_idx: u32,
    pub slot: u32,
    pub kind: WeightKind,
}

/// Build `accessor-base-name → WeightLocInfo` from a decode bucket's
/// backbone + lm_head `weight_slots`, reproducing
/// [`crate::codegen::emit_weight_accessors_impl`]'s per-(op_idx, kind)
/// slot-ordinal walk EXACTLY — both route the kind→method key through
/// [`weight_kind_accessor_method`] — so a wavefront weight source resolves
/// to the SAME `(bucket, op_idx, slot)` match arm the non-mega decode path
/// uses. `bb_bucket_id` is the backbone tape-index (`2*ci`); lm_head is
/// `bb_bucket_id + 1`.
///
/// First writer wins per base: within one (loop-compressed) decode body
/// each accessor base occurs once per `(op_idx, kind)`, and every arm for a
/// given base calls the same layer-parametric `self.<base>(layer)` getter,
/// so any one arm is interchangeable given the right `layer`.
pub fn build_base_to_loc(
    backbone: &[Vec<WeightSlot>],
    lm_head: &[Vec<WeightSlot>],
    bb_bucket_id: u32,
) -> HashMap<String, WeightLocInfo> {
    let mut map: HashMap<String, WeightLocInfo> = HashMap::new();
    let mut walk = |slots_arr: &[Vec<WeightSlot>], bucket: u32| {
        for (op_idx, slots) in slots_arr.iter().enumerate() {
            // Per-(op_idx, method) ordinal — identical to the runtime match
            // table's `slot` axis.
            let mut counts: HashMap<&'static str, u32> = HashMap::new();
            for slot in slots {
                let key = weight_kind_accessor_method(&slot.kind);
                let n = counts.entry(key).or_insert(0);
                let ordinal = *n;
                *n += 1;
                map.entry(slot.base.to_string())
                    .or_insert_with(|| WeightLocInfo {
                        bucket,
                        op_idx: op_idx as u32,
                        slot: ordinal,
                        kind: slot.kind.clone(),
                    });
            }
        }
    };
    walk(backbone, bb_bucket_id);
    walk(lm_head, bb_bucket_id + 1);
    map
}

/// Recover `(group_size, bits)` for a weight from its FUF `storage`
/// annotation (the per-model `annotate_storage_formats` pass). `None` for
/// dense / unquantized weights.
fn affine_gs_bits(fuf: &Fuf, id: u32, index: Option<UnrollIndex>) -> Option<(u32, u32)> {
    for node in &fuf.nodes {
        for inp in &node.inputs {
            if let FufInput::Weight {
                id: wid,
                index: widx,
                storage,
            } = inp
                && wid.0 == id
                && *widx == index
            {
                return match storage {
                    StorageFormat::Affine { bits, group_size } => Some((*group_size, *bits)),
                    _ => None,
                };
            }
        }
    }
    None
}

/// Outcome of resolving the wavefront weight sources to real runtime
/// locators — the macro-time dump reports this to verify every weight keys
/// a real `WeightAccessors` match arm (the whole point of macro-emission
/// increment 2).
#[derive(Debug, Default, Clone)]
pub struct ResolutionReport {
    /// Total `SourceBinding::Weight` sources.
    pub weights_total: usize,
    /// How many resolved to a real `WeightLoc` (base found in the decode
    /// bucket's `weight_slots`).
    pub weights_resolved: usize,
    /// Accessor base names that did NOT resolve (sorted, deduped) — each
    /// fell back to a placeholder locator. Empty ⇒ every weight resolved.
    pub unresolved: Vec<String>,
}

/// Compute dump stats for a lowered forward (the drive logs these).
pub fn stats(fuf: &Fuf, asn: &Assignment, lowered: &LoweredDecode) -> BridgeStats {
    let input = &lowered.input;
    let mut hist: HashMap<&'static str, usize> = HashMap::new();
    for od in &input.ops {
        // Names come from THE op registry, not a second table that
        // can drift from it.
        let name = od.op.name();
        *hist.entry(name).or_default() += 1;
    }
    let mut op_histogram: Vec<(&'static str, usize)> = hist.into_iter().collect();
    op_histogram.sort_by_key(|(n, _)| *n);
    let weight_sources = lowered
        .bindings
        .iter()
        .filter(|b| matches!(b, SourceBinding::Weight { .. }))
        .count();
    let prefix_sources = lowered
        .bindings
        .iter()
        .filter(|b| {
            matches!(
                b,
                SourceBinding::PrefixK { .. } | SourceBinding::PrefixV { .. }
            )
        })
        .count();
    BridgeStats {
        fuf_tiles: fuf.len(),
        subgraphs: asn.num_subgraphs(),
        sources: input.sources.len(),
        ops: input.ops.len(),
        op_histogram,
        weight_sources,
        prefix_sources,
    }
}
