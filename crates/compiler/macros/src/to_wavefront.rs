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

use scratchy_subtile::lower::{AffineInt4, GemmWeight, InputRef, LoweredOp, LoweringInput, OpDesc};
use scratchy_subtile::model_geometry::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
use scratchy_subtile::subtile_ir::SourceShape;

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
            Self::NoMetalRealization { tile, detail } => {
                write!(f, "tile {}: {detail}", tile.0)
            }
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
    /// `(LoweredOp kind name, count)`, sorted by name.
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
    prefix: HashMap<u64, (usize, usize)>,
    /// The model's head geometry, minted once at the bounds parse below — one value, with its GQA
    /// grouping already proven, rather than three integers this builder could pair up wrongly.
    geom: Option<ModelAttnGeometry>,
    eps: f32,
    scale: f32,
    m: u32,
    /// Modeled prefix length for the KV-cache `Source` rows (structural;
    /// the host eval / GPU player binds the real length at run time).
    prefix_len: u32,
}

impl<'a> Builder<'a> {
    fn push_op(&mut self, op: LoweredOp, inputs: Vec<InputRef>) -> usize {
        let idx = self.ops.len();
        self.ops.push(OpDesc {
            op,
            m: self.m,
            inputs,
        });
        idx
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

    fn prefix_for(&mut self, layer: u64) -> (usize, usize) {
        if let Some(pp) = self.prefix.get(&layer) {
            return *pp;
        }
        let kvdim = self
            .geom
            .expect(
                "prefix_for is only reachable from KV-cache arms, which refuse \
                 on canonicals without attention geometry",
            )
            .kv_width();
        let pk = self.push_source(self.prefix_len, kvdim, SourceBinding::PrefixK { layer });
        let pv = self.push_source(self.prefix_len, kvdim, SourceBinding::PrefixV { layer });
        self.prefix.insert(layer, (pk, pv));
        (pk, pv)
    }
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
    };

    // The result is the last value-producing op (the lm_head GEMM).
    let mut result: Option<usize> = None;

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
            // same `EmbeddedHidden` source. No `LoweredOp` is emitted
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
                let gain_offset = gain_tile
                    .and_then(|t| bx.gain_offsets.get(&t).copied())
                    .unwrap_or(0.0);
                let w = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::RmsNorm { eps, gain_offset }, vec![x, w]);
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
                // rows = m * mult / div: multiplying views (per-head)
                // have m | rows; the patch merger DIVIDES (rows | m).
                let (rows_mult, rows_div) = if rows != 0 && rows.is_multiple_of(bx.m) {
                    (rows / bx.m, 1)
                } else if rows != 0 && bx.m.is_multiple_of(rows) {
                    (1, bx.m / rows)
                } else {
                    return Err(BridgeError::UnresolvedShape {
                        tile,
                        what: "reshape rows neither multiply nor divide num_tokens",
                    });
                };
                let idx = bx.push_op(
                    LoweredOp::Reshape {
                        rows_mult,
                        rows_div,
                        cols,
                    },
                    vec![x],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::BiasAdd => {
                let x = bx.input_at(tile, 0)?;
                let b = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::BiasAdd, vec![x, b]);
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
                let idx = bx.push_op(LoweredOp::Gemm { n, weight }, inputs);
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
                let head_dim = bx
                    .geom
                    .ok_or(BridgeError::MissingBound { key: "head_dim" })?
                    .hd();
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
                let (cos_q, sin_q) = bx.cos_sin(q_cols, local_rope);
                let (cos_k, sin_k) = bx.cos_sin(k_cols, local_rope);
                // E.12 — RopeAppend writes rotated K and V into the
                // paged KV cache. Pull the per-layer PrefixK / PrefixV
                // source indices via the cached `prefix_for(layer)`
                // helper (same indices Attention will receive later).
                let (pk, pv) = bx.prefix_for(layer as u64);
                let qi = bx.push_op(
                    LoweredOp::RopeRotate { head_dim },
                    vec![q, InputRef::Ext(cos_q), InputRef::Ext(sin_q)],
                );
                let ki = bx.push_op(
                    LoweredOp::RopeAppend {
                        head_dim,
                        layer,
                        // Default; the consuming attention arm backpatches
                        // `false` for sliding layers.
                        is_global: true,
                        interleaved: node.op == OpKind::RopeAppendInterleaved,
                    },
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
            // attention(q', k', v, kv_cache, block_table): AttnDecode
            // reading Q_rot, prefix-cache Source segments, then the new
            // token's (K_rot, V) as Sub edges.
            OpKind::Attention | OpKind::SlidingAttention => {
                if bx.scale.is_nan() {
                    return Err(BridgeError::MissingBound { key: "head_dim" });
                }
                let q = bx.input_at(tile, 0)?;
                let k = bx.input_at(tile, 1)?;
                let v = bx.input_at(tile, 2)?;
                let Some(layer) = kv_cache_index(node) else {
                    // Cache-less encoder form: attention(q, k, v).
                    if bx.scale.is_nan() {
                        return Err(BridgeError::MissingBound { key: "head_dim" });
                    }
                    let k = bx.input_at(tile, 1)?;
                    let v = bx.input_at(tile, 2)?;
                    let idx = bx.push_op(
                        LoweredOp::EncoderAttn {
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
                let (pk, pv) = bx.prefix_for(layer);
                // valid_len = the modeled prefix-cache rows: the RopeAppend
                // above wrote the new token into the cache at decode_position,
                // so the cache `[prefix_len, kv]` already spans prefix ++ new
                // = `prefix_len` valid positions. The eval_node prefix slice
                // uses `valid_len - 1` (the read-only prefix rows; the new
                // row arrives via the separate k/v segments). The GPU mask
                // binds the real length from the runtime DecodePosition arg.
                let valid_len = bx.prefix_len;
                let sliding = node.op == OpKind::SlidingAttention;
                let base_geom = bx
                    .geom
                    .ok_or(BridgeError::MissingBound { key: "head_dim" })?;
                let geom = if !sliding {
                    bx.geom_global.unwrap_or(base_geom)
                } else {
                    base_geom
                };
                if sliding {
                    // The k-rope for a sliding layer targets the LOCAL
                    // geometry class.
                    if let InputRef::Op(ri) = k
                        && let LoweredOp::RopeAppend { is_global, .. } = &mut bx.ops[ri].op
                    {
                        *is_global = false;
                    }
                }
                let idx = bx.push_op(
                    LoweredOp::AttnDecode {
                        geom,
                        scale,
                        valid_len,
                        sliding,
                    },
                    vec![q, InputRef::Ext(pk), InputRef::Ext(pv), k, v],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Silu => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(LoweredOp::Silu, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Gelu => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(LoweredOp::Gelu, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::RmsNormUnit => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(LoweredOp::RmsNormUnit { eps }, vec![x]);
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
                let idx = bx.push_op(LoweredOp::ScalarWeightMul, vec![x, w]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Moe => {
                let x = bx.input_at(tile, 0)?;
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
                // num_experts (neither on deepseek's n_routed_experts).
                let (qwen_shared, num_experts) = if let Some(&n) = b.get("num_local_experts") {
                    (false, n as u32)
                } else if let Some(&n) = b.get("num_experts") {
                    (true, n as u32)
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
                let idx = bx.push_op(
                    LoweredOp::Moe {
                        qwen_shared,
                        num_experts,
                        top_k,
                        moe_inter,
                        shared_inter,
                        norm_topk,
                        group_size,
                        bits,
                    },
                    vec![x, w],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GateSplit => {
                let x = bx.input_at(tile, 0)?;
                let cols = bx.out_cols(tile, 0, "gate_split q cols")?;
                let idx = bx.push_op(LoweredOp::GateSplit { half_cols: cols }, vec![x]);
                // TWO outputs: slot 0 = q, slot 1 = gate.
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                bx.produced.insert((tile.0, 1), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GateApply => {
                let attn = bx.input_at(tile, 0)?;
                let gate = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::GateApply, vec![attn, gate]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GateScale => {
                let routed = bx.input_at(tile, 0)?;
                let shared = bx.input_at(tile, 1)?;
                let g = bx.input_at(tile, 2)?;
                let idx = bx.push_op(LoweredOp::GateScale, vec![routed, shared, g]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GatedDeltaNet => {
                let qkv = bx.input_at(tile, 0)?;
                let z = bx.input_at(tile, 1)?;
                let a = bx.input_at(tile, 2)?;
                let b2 = bx.input_at(tile, 3)?;
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
                        op: OpKind::GatedDeltaNet,
                        detail: "gated_delta_net without a weight bundle",
                    })?;
                let idx = bx.push_op(LoweredOp::GatedDeltaNet, vec![qkv, z, a, b2, w]);
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
                let rsrc = bx.push_source(
                    1,
                    1,
                    SourceBinding::Weight {
                        id: rw.0,
                        index: rw.1,
                    },
                );
                let esrc = bx.push_source(
                    1,
                    1,
                    SourceBinding::Weight {
                        id: ew.0,
                        index: ew.1,
                    },
                );
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
                let idx = bx.push_op(
                    LoweredOp::GemmaMoe {
                        num_experts,
                        top_k,
                        moe_inter,
                        group_size,
                        bits,
                    },
                    vec![
                        router_in,
                        expert_in,
                        InputRef::Ext(rsrc),
                        InputRef::Ext(esrc),
                    ],
                );
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::TanhSoftCap => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(LoweredOp::TanhSoftCap, vec![x]);
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
                        let idx = bx.push_op(LoweredOp::ScalarMul { scale }, vec![x]);
                        bx.produced.insert((tile.0, 0), Producer::Op(idx));
                        result = Some(idx);
                    }
                    None => {
                        let a = bx.input_at(tile, 0)?;
                        let b2 = bx.input_at(tile, 1)?;
                        let idx = bx.push_op(LoweredOp::Mul, vec![a, b2]);
                        bx.produced.insert((tile.0, 0), Producer::Op(idx));
                        result = Some(idx);
                    }
                }
            }
            OpKind::LoadPixels => {
                let cols = bx.out_cols(tile, 0, "pixels cols")?;
                let idx = bx.push_op(LoweredOp::LoadPixels { in_features: cols }, vec![]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::LoadPosEmbeds => {
                // Same shape as LoadPixels: a synthesized source tile
                // whose width the FUF already carries.
                let cols = bx.out_cols(tile, 0, "pos_embeds cols")?;
                let idx = bx.push_op(LoweredOp::LoadPosEmbeds { width: cols }, vec![]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::VisionRope => {
                let q = bx.input_at(tile, 0)?;
                let k = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::VisionRope, vec![q, k]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                bx.produced.insert((tile.0, 1), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::QuickGelu => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(LoweredOp::QuickGelu, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::GeluErf => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(LoweredOp::GeluErf, vec![x]);
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
                let idx = bx.push_op(LoweredOp::EmbeddingGather { indices_kind }, vec![x]);
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
                let idx = bx.push_op(LoweredOp::VarlenAttention { cu_kind }, vec![q, k, v]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Mean => {
                let x = bx.input_at(tile, 0)?;
                let idx = bx.push_op(LoweredOp::Mean, vec![x]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Sub => {
                let a = bx.input_at(tile, 0)?;
                let b2 = bx.input_at(tile, 1)?;
                let idx = bx.push_op(LoweredOp::Sub, vec![a, b2]);
                bx.produced.insert((tile.0, 0), Producer::Op(idx));
                result = Some(idx);
            }
            OpKind::Add => {
                // Gemma's `(1 + w)` norm-gain convention rides the DSL
                // as `Add(weight, scalar)`. The offset is a KERNEL-time
                // constant (`NORM_WEIGHT_OFFSET` on metal; each target's
                // rmsnorm applies it), so the Add tile ALIASES its
                // weight input on the tape — nothing is emitted, and a
                // scalar that isn't the arch's declared offset refuses.
                let scalar = node.inputs.iter().find_map(|i| match i {
                    crate::fuf::FufInput::Scalar(v) => Some(*v),
                    _ => None,
                });
                if let Some(v) = scalar {
                    let declared = crate::codegen::norm_weight_runtime_offset(bx.model) as f64;
                    // Declared-offset arches: the kernel const applies it →
                    // 0 on the tape. Otherwise the offset is DATA the
                    // consuming rmsnorm carries (Gemma's per-instruction
                    // offset vocabulary).
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
                    let idx = bx.push_op(LoweredOp::Add, vec![a, b2]);
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
        },
        bindings: bx.bindings,
        op_tiles,
        norm_gain_add_tiles: bx.norm_gain_add_tiles,
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
        // can drift from it (`lowered_op_name` is derived from the
        // same rows the enum lock guard checks).
        let name = scratchy_subtile::ops::lowered_op_name(&od.op);
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
