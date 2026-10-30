// SPDX-License-Identifier: Apache-2.0
//! The canonical wavefront **SubtileIR** — region-granular SSA over
//! tensors, the fold of `region.rs` (v2) + `subtile.rs` (v1) into one
//! module.
//!
//! ## What this is
//!
//! Every value lives in a **tensor** (a logical row-major buffer): leaf
//! `sources` (weights / activations / prefix-KV / embed / cos-sin) and one
//! **op-output tensor** per op. A [`SubtileNode`] *writes a region* of
//! one output tensor and *reads regions* of input tensors. Dependencies
//! are derived from **region overlap**: a node that reads region `R` of
//! an op-output tensor `T` depends on every earlier node whose
//! write-region on `T` overlaps `R`. Reads of a leaf source have no
//! dependency (sources are bound at eval time).
//!
//! That is the SSA model that supports **true subtile granularity** on
//! the GPU: `q_proj` split into N-blocks where each block writes a slice
//! of Q, then rope → attention reads the *whole* Q assembled from those
//! slices. The v1 whole-output `Operand::Sub` model couldn't express
//! that — and is gone (it lives only in [`legacy`] for the few v1-only
//! carcasses that have not been deleted yet: [`crate::tape`],
//! [`crate::scheduler`], [`crate::lower::lower`] which are slated for
//! deletion in later staged commits).
//!
//! ## Two-tier validation
//!
//! - **Tier A′ (decomposition equivalence):** [`eval_dag`] equals
//!   `cpu_golden` whole-op output. Bit-exact at `nb >= n` and at every
//!   N-block width that doesn't reorder a per-output reduction; only
//!   token-exact once a reduction is reassociated (split-K).
//! - **Tier A (self-consistency):** scheduled tape replay equals
//!   [`eval_dag`] (lands with the wavefront scheduler).
//!
//! `cpu_golden` (in `scratchy-forward-compiler`) is the deterministic f32 substrate
//! the player computes with — not the correctness *oracle* (that is
//! scratchy-target-metal non-mega at temp=0).

#![allow(dead_code)]

use crate::model_geometry::{HeadDim, ModelAttnGeometry};
use std::marker::PhantomData;

// ── Identifiers & geometry ─────────────────────────────────────────

/// Dense index into a [`SubtileIR::nodes`] vector. The DAG is
/// topologically ordered: every overlap-predecessor of a node has a
/// smaller id (so a single pass over `nodes` is a valid evaluation
/// order).
/// Sealed per §2: the inner field is `pub(crate)`, so external code
/// cannot construct or read this id outside the crate. Internal
/// construction stays the cheap `SubtileId(N)` tuple form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SubtileId(pub(crate) u32);

impl SubtileId {
    /// The dense subtile index (read-only). Public for the same reason
    /// `TensorId::index` is: the SDSC lowering that reads it now lives
    /// in the spyre target crate, so it addresses nodes by index
    /// without the sealed field. Construction stays in-crate.
    pub fn index(&self) -> usize {
        self.0 as usize
    }

    /// Name the subtile at `index` from outside the crate — same
    /// contract as [`TensorId::from_index`]: it names a node, it does
    /// not create one.
    pub fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Half-open range `[start, start + len)` along one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub start: u32,
    pub len: u32,
}

impl Range {
    pub fn new(start: u32, len: u32) -> Self {
        Self { start, len }
    }
    pub fn end(&self) -> u32 {
        self.start + self.len
    }
}

/// A rectangular slice of a logically row-major `[rows, cols]` buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub rows: Range,
    pub cols: Range,
}

/// Logical shape of a leaf source buffer (used by the v1
/// `LoweringInput`-side bridge in [`crate::lower`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceShape {
    pub rows: u32,
    pub cols: u32,
}

/// Elementwise op kind. Numerics mirror `cpu_golden` exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EwKind {
    /// `x / (1 + e^-x)`.
    Silu,
    /// `x · Φ(x)`. A REAL device primitive on the targets that have one — the
    /// dxp DDL ships `"gelu"` (`OpFunc::Gelu`) with its own SFP polynomial
    /// table, so this lowers to one pointwise op rather than the tanh
    /// approximation expanded into arithmetic.
    Gelu,
    /// `a * b`.
    Mul,
    /// `a + b`.
    Add,
    /// `x · σ(1.702x)` — the vision tower's activation. Distinct from [`EwKind::Gelu`]: it is a
    /// different function, not a different approximation of one, and the 1.702 sigmoid form is
    /// what the checkpoint was trained against.
    QuickGelu,
    /// `x · Φ(x)` with the EXACT erf, not the tanh approximation — the vision patch-merger MLP.
    /// Kept apart from [`EwKind::Gelu`] because a target with only one of the two must say so
    /// rather than substitute the other and shift every merged patch slightly.
    GeluErf,
    /// `relu(x)²` — Nemotron-H's non-gated MLP activation (mlx `nn.relu2`).
    Relu2,
    /// `a - b`, with a `[m, 1]` broadcast on `b` (the mean-centering half of a LayerNorm).
    Sub,
    /// `a + bias`, with a `[1, n]` row broadcast on `bias` — a weight source (a projection's
    /// bias). Kept apart from [`EwKind::Add`] because the operand is a WEIGHT, not an
    /// activation: a target binds it through the projection's accessor, and the bias is the
    /// learned constant a KV codec must remove before quantizing a cached K/V.
    BiasAdd,
}

// ── Tensors & regions ──────────────────────────────────────────────

/// Dense index into [`SubtileIR::tensors`]. Tensors `[0, num_sources)`
/// are leaf sources bound at eval time; the rest are op outputs written
/// by subtile nodes.
/// Sealed per §2: inner field is `pub(crate)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub(crate) u32);

impl TensorId {
    /// The dense tensor index (read-only). Public so out-of-crate consumers — e.g. the staged SDSC
    /// lowering in `scratchy-sdsc`, which reads SubtileIR's node structure to lower it — can address
    /// `tensors[id.index()]` without the `pub(crate)` field. Construction stays sealed in-crate.
    pub fn index(&self) -> usize {
        self.0 as usize
    }

    /// Name the tensor at `index` from OUTSIDE the crate. Construction
    /// is otherwise sealed; a target emitter that appends a synthetic
    /// tensor to `SubtileIR::tensors` (the KTIR mask, for one) needs to
    /// name the slot it just created. The contract is exactly that:
    /// `index` must address a slot that exists, or is about to, in the
    /// same graph — this mints an ID, it does not allocate storage.
    pub fn from_index(index: usize) -> Self {
        Self(index as u32)
    }
}

/// Logical shape of a tensor (row-major `[rows, cols]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    pub rows: u32,
    pub cols: u32,
}

impl TensorShape {
    /// The region covering the whole tensor.
    pub fn whole(&self) -> Region {
        Region {
            rows: Range::new(0, self.rows),
            cols: Range::new(0, self.cols),
        }
    }
}

/// A rectangular slice of a tensor — used for both a node's input reads
/// and its single output write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorRegion {
    pub tensor: TensorId,
    pub region: Region,
}

// ── Typed witnesses on SubtileIR DAG nodes ─────────────────────────
//
// Each witness encodes one IR-level invariant. Producer / consumer
// ops carry the SAME witness *value* by construction (single source of
// truth), or — in the case of [`RopeForm`] — share the SAME
// `F: RopeForm` const-generic *type* on the entire IR (so mixing
// NeoX/Interleaved in one forward is a compile error, not a runtime
// surprise).

#[doc(hidden)]
pub mod sealed {
    /// Sealing token — inner `()` is `pub(super)` so external code
    /// cannot construct a `Seal` value. Carried by every type that
    /// must be constructable only inside `subtile_ir`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct Seal(pub(super) ());
}

/// **`RopeForm`** — sealed marker trait selecting the rotary pairing
/// form. Llama-3.2 uses [`NeoX`]; some other architectures use
/// [`Interleaved`]. Encoded as a const-generic phantom on
/// [`SubtileIR<F>`] / [`SubtileNode<F>`] / [`SubOp<F>`] so a single
/// forward's rope nodes ALL share the same form by construction —
/// mixing NeoX and Interleaved in one IR is a compile error.
///
/// ```compile_fail
/// // Mixing NeoX and Interleaved in one IR is rejected at the type
/// // level: SubtileIR<NeoX>::nodes is Vec<SubtileNode<NeoX>>, so a
/// // SubtileNode<Interleaved> won't fit in it. No need for a runtime
/// // check; the const generic enforces it.
/// use scratchy_subtile::subtile_ir::{
///     Interleaved, NeoX, SubOp, SubtileId, SubtileIR, SubtileNode, TensorId, TensorRegion,
///     Region, Range,
/// };
/// use std::marker::PhantomData;
/// let interleaved_node = SubtileNode::<Interleaved> {
///     id: SubtileId(0),
///     op: SubOp::<Interleaved>::RopeRotate {
///         head_dim: 4,
///         _form: PhantomData,
///     },
///     inputs: vec![],
///     output: TensorRegion {
///         tensor: TensorId(0),
///         region: Region { rows: Range::new(0, 1), cols: Range::new(0, 4) },
///     },
/// };
/// let _ir: SubtileIR<NeoX> = SubtileIR {
///     tensors: vec![],
///     num_sources: 0,
///     nodes: vec![interleaved_node], // type mismatch
///     result: TensorId(0),
/// };
/// ```
pub trait RopeForm:
    rope_form_seal::Sealed
    + Copy
    + std::fmt::Debug
    + PartialEq
    + Eq
    + std::hash::Hash
    + Send
    + Sync
    + 'static
{
    /// Erased tag for runtime introspection (printing, unit tests).
    const TAG: RopeFormTag;
}

/// NeoX rope: pairs `(d, d + half)` per head. Llama-3.2 invariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NeoX {}

/// Interleaved rope: pairs `(2k, 2k + 1)` per head.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Interleaved {}

#[doc(hidden)]
pub mod rope_form_seal {
    pub trait Sealed {}
    impl Sealed for super::NeoX {}
    impl Sealed for super::Interleaved {}
}

impl RopeForm for NeoX {
    const TAG: RopeFormTag = RopeFormTag::NeoX;
}
impl RopeForm for Interleaved {
    const TAG: RopeFormTag = RopeFormTag::Interleaved;
}

/// Erased rope-form tag for runtime use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeFormTag {
    NeoX,
    Interleaved,
}

// ── KvCacheLayout ────────────────────────────────────────────────────
//
// The per-model K/V cache dims (`num_kv_heads`, `head_dim`, …) are NOT
// duplicated here: they already live on the lowered IR ops
// (`SubOp::AttnDecode { num_kv_heads, head_dim, … }`,
// `SubOp::RopeAppend { head_dim, … }`), stamped from `model.bounds` in
// `to_wavefront`. The lowering consumes those op fields directly.
// `KvCacheLayout` is just the sealed pair of cache TensorIds the
// producer (`RopeAppend`) writes and the consumer (`AttnDecode`) reads.

/// **`KvCacheLayout`** — sealed witness naming the K-cache (or V-cache)
/// tensor a single forward reads from / writes to. The orchestrator
/// builds ONE per cache tensor; `RopeAppend`'s write and `AttnDecode`'s
/// read both reach for the SAME instance, making layout drift between
/// producer and consumer structurally impossible.
///
/// Constructable only via [`KvCacheLayout::for_cache_tensors`] — sealed.
///
/// ```compile_fail
/// // Sealed: external code cannot construct a KvCacheLayout via the
/// // struct literal because the `_seal: sealed::Seal` field is
/// // private. The only path is `KvCacheLayout::for_cache_tensors(...)`,
/// // which makes layout drift (a divergent producer / consumer
/// // fabrication) structurally impossible.
/// use scratchy_subtile::subtile_ir::{KvCacheLayout, TensorId};
/// let _ = KvCacheLayout {
///     cache_tensor: TensorId(0),
/// };
/// ```
///
/// The cache's numeric dimensions (`num_kv_heads`, `head_dim`) are NOT
/// carried here — they already live on the IR ops (`SubOp::AttnDecode`
/// / `SubOp::RopeAppend`), stamped from `model.bounds`. This witness is
/// purely the cache-tensor identity pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvCacheLayout {
    /// K-side cache tensor (rotated K is written here).
    cache_tensor: TensorId,
    /// V-side cache tensor (un-rotated V is written here). Often a
    /// separate tensor; some layouts use the same tensor as K with
    /// different layer-base offsets.
    v_cache_tensor: TensorId,
    _seal: sealed::Seal,
}

impl KvCacheLayout {
    /// Sealed constructor binding both K and V cache tensors into
    /// the witness; per-axis numerics come from the IR op fields.
    pub const fn for_cache_tensors(k_cache_tensor: TensorId, v_cache_tensor: TensorId) -> Self {
        Self {
            cache_tensor: k_cache_tensor,
            v_cache_tensor,
            _seal: sealed::Seal(()),
        }
    }

    /// K-side cache tensor.
    pub const fn cache_tensor(&self) -> TensorId {
        self.cache_tensor
    }
    /// V-side cache tensor.
    pub const fn v_cache_tensor(&self) -> TensorId {
        self.v_cache_tensor
    }
}

// The per-axis numeric accessors (num_kv_heads / head_dim /
// row_elements / max_position / row_bytes / layer_base_bytes) were
// deleted: those dims now live on the IR ops, not on the layout
// witness. See the offset functions in tk_tape.rs for how the dims
// flow from the ops into the byte-offset math.

/// **`KvCacheProducer`** — sealed enum naming HOW the K (or V) cache
/// that an [`SubOp::AttnDecode`] reads got populated. The variants are
/// sealed (constructable only via [`KvCacheProducer::from_rope_append`]
/// / [`KvCacheProducer::pre_populated_ext`]) and the enum is
/// `#[non_exhaustive]` so external `match`es must include a wildcard
/// arm — preventing the silent `_ =>` regression on a future variant.
///
/// ```compile_fail
/// // External match without a wildcard is rejected: the enum is
/// // #[non_exhaustive], so the compiler forces a `_ =>` arm. That
/// // makes adding a new variant a soft-fail (existing matchers route
/// // it to the wildcard) rather than a silent miscompile of the kind
/// // a non-exhaustive enum without `non_exhaustive` would suffer.
/// use scratchy_subtile::subtile_ir::KvCacheProducer;
/// fn name(p: KvCacheProducer) -> &'static str {
///     match p {
///         KvCacheProducer::SameForwardRopeAppend { .. } => "rope_append",
///         KvCacheProducer::PrePopulatedExt { .. } => "ext",
///     }
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KvCacheProducer {
    /// The cache was written by a KV writer (`SubOp::RopeAppend` or
    /// `SubOp::KvAppend`) earlier in this same forward (the producer is
    /// `nodes[producer_node_idx]`).
    SameForwardRopeAppend {
        producer_node_idx: u32,
        #[doc(hidden)]
        _seal: sealed::Seal,
    },
    /// The cache is pre-populated by an out-of-band per-op forward and
    /// is read-only inside this fused kernel.
    PrePopulatedExt {
        #[doc(hidden)]
        _seal: sealed::Seal,
    },
}

impl KvCacheProducer {
    pub const fn from_rope_append(producer_node_idx: u32) -> Self {
        Self::SameForwardRopeAppend {
            producer_node_idx,
            _seal: sealed::Seal(()),
        }
    }

    pub const fn pre_populated_ext() -> Self {
        Self::PrePopulatedExt {
            _seal: sealed::Seal(()),
        }
    }
}

/// **`SoftmaxStateId`** — opaque identifier for the per-AttnDecode
/// online-softmax accumulator (`m`, `l`, `o`). The lowering walker
/// allocates one per AttnDecode; the TkTape lowering binds it to the
/// concrete register set at emit time. Constructable only inside this
/// crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SoftmaxStateId {
    id: u32,
    _seal: sealed::Seal,
}

impl SoftmaxStateId {
    pub(crate) const fn new(id: u32) -> Self {
        Self {
            id,
            _seal: sealed::Seal(()),
        }
    }
    pub const fn index(&self) -> u32 {
        self.id
    }
}

// ── Op stages ──────────────────────────────────────────────────────

/// Which side of [`lower_region`] a [`SubOp`] value lives on.
///
/// ⭐ ONE VOCABULARY, TWO STAGES. The front end states each op of a forward as a `SubOp` of the
/// [`Arch`] stage — [`crate::lower::LoweringInput`]'s ops — and `lower_region` turns each into
/// the [`Tiled`] nodes of a [`SubtileIR`], binding what only the lowering knows: the KV-cache
/// tensors a rope writes and an attention reads, which node produced that cache, and each
/// attention's softmax accumulator. The op is the same value on both sides; the stage says which
/// of those witnesses it can carry.
pub trait OpStage:
    stage_seal::Sealed + Copy + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + 'static
{
    /// The cache tensors a `RopeAppend` writes / an `AttnDecode` reads.
    type KvLayout: Copy + std::fmt::Debug + PartialEq;
    /// How an `AttnDecode`'s cache got populated.
    type KvProducer: Copy + std::fmt::Debug + PartialEq;
    /// An `AttnDecode`'s online-softmax accumulator.
    type SoftmaxState: Copy + std::fmt::Debug + PartialEq;
    /// Carried by the ops only the lowering mints (the split-K combine, the two phases of a
    /// decomposed RmsNorm). Uninhabited on the arch stage, so no front end can state one.
    type Minted: Copy + std::fmt::Debug + PartialEq;
}

/// The op as the front end states it: nothing the lowering binds is known yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arch {}

/// The op on a node of a lowered [`SubtileIR`], every witness bound. The default stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tiled {}

/// A witness [`lower_region`] has not bound yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Unbound;

/// Marks an op the lowering minted. Constructible only in this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Minted(sealed::Seal);

impl Minted {
    const BY_LOWERING: Self = Self(sealed::Seal(()));
}

#[doc(hidden)]
pub mod stage_seal {
    pub trait Sealed {}
    impl Sealed for super::Arch {}
    impl Sealed for super::Tiled {}
}

impl OpStage for Arch {
    type KvLayout = Unbound;
    type KvProducer = Unbound;
    type SoftmaxState = Unbound;
    type Minted = std::convert::Infallible;
}

impl OpStage for Tiled {
    type KvLayout = KvCacheLayout;
    type KvProducer = KvCacheProducer;
    type SoftmaxState = SoftmaxStateId;
    type Minted = Minted;
}

/// How a view's row count relates to the op's token count `m`.
///
/// ⛔ NOT A `(mult, div)` PAIR. A pair admits a view that both multiplies and divides, and a zero
/// in either place; a view does exactly one of the two, by a positive factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowScale {
    /// `m · k` rows — a per-head view splits each token's row into `k` (`k = 1` keeps the rows).
    Times(std::num::NonZeroU32),
    /// `m / k` rows — the vision patch merger folds `k` tokens into one.
    Over(std::num::NonZeroU32),
}

impl RowScale {
    /// The view keeps one row per token.
    pub const fn preserves_rows(self) -> bool {
        matches!(self, Self::Times(k) if k.get() == 1)
    }
    /// The numerator of `m · mult / div` — 1 for `Over` (a view that divides
    /// cannot also multiply).
    pub const fn mult(self) -> u32 {
        match self {
            Self::Times(k) => k.get(),
            Self::Over(_) => 1,
        }
    }
    /// The denominator of `m · mult / div` — 1 for `Times`.
    pub const fn div(self) -> u32 {
        match self {
            Self::Times(_) => 1,
            Self::Over(k) => k.get(),
        }
    }
}

// ── Sub-operations ─────────────────────────────────────────────────

/// How an RmsNorm reads its gain vector.
///
/// ⛔ NOT AN `f32` OFFSET. The DSL folds the gain as `w + offset`, and a float admits values —
/// 0.5, NaN — that no checkpoint convention means and no kernel implements. There are exactly two
/// conventions in the world, so they get exactly two names, and the conversion is the place a
/// third one has to be rejected instead of silently scaling wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainConvention {
    /// `y = x̂ · w` — llama, qwen, granite.
    Scale,
    /// `y = x̂ · (1 + w)` — the gemma class, whose checkpoints store the gain centred on zero.
    OnePlusScale,
}

impl GainConvention {
    /// The multiplier a stored gain `w` contributes.
    pub fn apply(self, w: f32) -> f32 {
        match self {
            Self::Scale => w,
            Self::OnePlusScale => 1.0 + w,
        }
    }

    /// Read the convention off the DSL's folded `w + offset` gain.
    ///
    /// ⛔ REFUSES A THIRD VALUE. An offset that is neither 0 nor 1 is not a convention this
    /// pipeline has a kernel for, and rounding it to the nearest one would scale every
    /// normalized activation by the wrong constant — a defect that reads as "the model is
    /// slightly worse", not as a crash.
    pub fn from_offset(offset: f32) -> Option<Self> {
        match offset {
            0.0 => Some(Self::Scale),
            1.0 => Some(Self::OnePlusScale),
            _ => None,
        }
    }

    /// The constant added to the stored gain — the inverse of [`Self::from_offset`].
    pub const fn offset(self) -> f32 {
        match self {
            Self::Scale => 0.0,
            Self::OnePlusScale => 1.0,
        }
    }
}

/// Which positions a decode-time attention may attend to.
///
/// ⛔ NOT A `bool`. A bare `sliding` / `is_global` boolean at a struct literal is the shape that
/// lets one land in the other (they are negations of each other, and both are in scope at the
/// same call). Naming the two masks makes the swap a type error instead of a model that quietly
/// attends to the wrong span. An attention and the rope that appends its K/V carry the same
/// value: the class of the layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMask {
    /// Every position up to and including the query's own.
    Causal,
    /// The gemma class's local layers: causal, further restricted to the last `window`
    /// positions. The window size RIDES the tape — it is a model constant, but one the
    /// numeric oracle needs (`eval_dag`'s host reference attends to the last `window`
    /// positions, exactly what a sliding kernel does); without it on the tape a hybrid
    /// model had NO golden bisection, only a panic. Set from the bridge
    /// (`model.bounds["sliding_window"]`) at the one place every attention is minted.
    SlidingWindow {
        /// How many of the most recent positions (including the query's own) stay
        /// attendable. A model constant, identical for every sliding layer.
        window: std::num::NonZeroU32,
    },
}

impl AttnMask {
    /// The sliding window size, for every consumer that only needs the number.
    pub fn window(self) -> Option<std::num::NonZeroU32> {
        match self {
            AttnMask::Causal => None,
            AttnMask::SlidingWindow { window } => Some(window),
        }
    }
}

// ── The mixture-of-experts vocabulary ──────────────────────────────
//
// A MoE block is its math, stated as ops: router logits → top-k → score transforms → the chosen
// experts' projections over (token, expert) PAIR rows → their weighted sum. The pair rows and the
// routing tables are not activations; which values a target keeps off its arena is the target's
// fact (its colour rules), not the op's.

/// How many routed experts a MoE block holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NumExperts(std::num::NonZeroU32);

/// How many experts each token is routed to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TopK(std::num::NonZeroU32);

macro_rules! nonzero_count {
    ($($t:ident),*) => {$(
        impl $t {
            pub const fn new(n: std::num::NonZeroU32) -> Self {
                Self(n)
            }
            pub const fn get(self) -> u32 {
                self.0.get()
            }
        }
    )*};
}
nonzero_count!(NumExperts, TopK);

/// Which of an expert's three projections a matmul is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpertProj {
    Gate,
    Up,
    Down,
}

/// The activation an expert gates its up projection with: `act(gate) * up`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GatedAct {
    Silu,
    Gelu,
    /// gpt-oss's SwiGLU-OAI: `gate = min(g, 7)`, `up = clamp(u, -7, 7)`,
    /// `y = (up + 1) · gate · sigmoid(1.702 · gate)`. The ±7 clamps are
    /// the reference's hard limits (transformers `SwiGLUOAI`, mlx
    /// `SwigluOAI` at `limit` 7.0 — asserted at config parse), part of
    /// the activation itself, not a softcap.
    SwigluOai,
}

/// The activation an ungated expert applies to its one in-projection: `act(fc1 · x)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UngatedAct {
    /// `relu(x)²` — Nemotron-H's `SwitchMLP` experts (mlx `nn.ReLU2`).
    Relu2,
}

/// What an expert computes between its in- and out-projections — which also says which
/// projections it HAS: a gated expert reads a gate and an up projection, an ungated one only its
/// `fc1`, which the block states as its up projection (and `fc2` as its down).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpertBody {
    /// `act(gate · x) * (up · x)`.
    Gated(GatedAct),
    /// `act(up · x)`.
    Ungated(UngatedAct),
}

/// The weight bundle a router reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RouterBundle {
    /// Gemma-4's own router (pre-norm gain, projection, per-expert scale).
    Gemma,
    /// The router inside a Mixtral-style fused MoE bundle.
    Fused,
    /// The router inside a Qwen-style fused MoE bundle.
    SharedFused,
    /// gpt-oss's router: a biased linear (the bias orders the top-k picks
    /// AND is read back as the pre-softmax score) over the experts bundle.
    GptOss,
}

/// The weight bundle an expert projection reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExpertBundle {
    /// Gemma-4's SwitchGLU expert stack.
    SwitchGlu,
    Fused,
    SharedFused,
    /// gpt-oss's SwiGLU-OAI experts: SwitchGLU-shaped stacks plus
    /// per-expert linear biases on gate/up/down.
    GptOss,
}

impl RouterBundle {
    /// THE bundle table: the weight kind a router bundle binds as.
    pub const fn weight_kind(self) -> crate::handoff::WeightKind {
        use crate::handoff::WeightKind as K;
        match self {
            Self::Gemma => K::GemmaRouter,
            Self::Fused => K::FusedMoe,
            Self::SharedFused => K::SharedFusedMoe,
            Self::GptOss => K::GptOssMoe,
        }
    }
}

impl ExpertBundle {
    /// THE bundle table: the weight kind an expert bundle binds as.
    pub const fn weight_kind(self) -> crate::handoff::WeightKind {
        use crate::handoff::WeightKind as K;
        match self {
            Self::SwitchGlu => K::GemmaSwitchGlu,
            Self::Fused => K::FusedMoe,
            Self::SharedFused => K::SharedFusedMoe,
            Self::GptOss => K::GptOssMoe,
        }
    }
}

/// A fused MoE bundle's shared-expert width, as the config declares it (`None`: no shared
/// expert). A DSL that states its shared expert as ops owns it; a target decides what an
/// undeclared one means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SharedExpertBound(pub Option<std::num::NonZeroU32>);

// ── The KV codec vocabulary ────────────────────────────────────────
//
// A KV codec stores a layer's cache compressed: each writer's new K and V are encoded into a
// packed store, and each attention either decodes that store directly or reads it staged back
// into the codec's rotated domain. Which classes a target codes, and when each step runs, are its
// facts (`crate::kv_codec`).

/// Which half of a layer's KV cache a codec step touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvOperand {
    K,
    V,
}

/// Which rows a codec rotation turns: the query into the codebook's domain (`R·q`), or the
/// attention's output back out of it (`Rᵀ·o`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RotatedRows {
    Query,
    Output,
}

/// What makes two nodes the SAME PROGRAM for re-rolling purposes.
///
/// ⭐ THE CLASS, NOT THE INSTANCE. Hashes every attribute that decides what a node *does*, and
/// masks the ones that only say *which layer it is*: the KV cache tensors, the producer's node
/// index, the softmax-state id, and `RopeAppend::layer`. Two layers hash equal iff they are the
/// same program at different weight offsets.
///
/// ⛔ WITHOUT THIS THE RE-ROLL CANNOT SEE A LAYER CLASS. `reroll_fingerprint` used to hash only
/// the structural SHAPE of a `Compute`'s inputs and to discard `node` entirely, so a Gemm and an
/// RmsNorm hashed alike — and, fatally, so did a sliding-window layer and a global one. gemma-3
/// then re-rolled 26 layers onto a single body and emitted one rope flavour for all of them
/// (`RopeAppend(.., is_global=false)` where the un-rolled stream has `true`). With the class in
/// the hash the run comes out as the SIX-layer `SSSSSG` cell it actually is.
///
/// ⛔ ENUMERATED, NEVER A CATCH-ALL. A new `SubOp` is a compile error here, not a silent
/// mis-roll — the failure mode this exists to end is invisible at the tape and only shows up as
/// wrong tokens on the device.
impl<F: RopeForm, S: OpStage> SubOp<F, S> {
    /// The layer this node belongs to, when it carries one.
    ///
    /// Only the KV-cache writer names its layer on the tape; everything else in a layer is
    /// identified by its per-layer WEIGHTS, not by an index. That is enough: the re-roll only
    /// needs to know how far the layer advances between two copies of a body, and any one
    /// layer-carrying op in the body measures that.
    pub fn layer_index(&self) -> Option<u32> {
        match self {
            SubOp::RopeAppend { layer, .. } | SubOp::KvAppend { layer, .. } => Some(*layer),
            _ => None,
        }
    }

    /// [`Self::reroll_class_key`] with the weight STORAGE masked: two nodes hash equal iff they
    /// are the same program at any bit width / group size.
    ///
    /// ⭐ THIS IS WHERE A MIXED-PRECISION MODEL'S LAYERS ARE. OptiQ quantizes gemma-4's layers
    /// 0–4 at 8 bits and 5–21 at 4, so under the class key only two of its five six-layer cells
    /// match and the other eighteen layers emit straight-line. The storage-masked key sees the
    /// same five cells plain gemma-4 has; the class key then says which LAYERS inside them are
    /// the same program.
    pub fn reroll_shape_key(&self, h: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        match self {
            SubOp::MatmulTile { n, weight } => {
                std::mem::discriminant(self).hash(h);
                n.hash(h);
                std::mem::discriminant(weight).hash(h);
            }
            SubOp::ExpertMatmul {
                proj,
                n,
                k,
                quant: _,
                bundle,
            } => {
                std::mem::discriminant(self).hash(h);
                (proj, n, k, bundle).hash(h);
            }
            // Everything else carries no storage: its shape IS its class.
            _ => self.reroll_class_key(h),
        }
    }

    pub fn reroll_class_key(&self, h: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        std::mem::discriminant(self).hash(h);
        match self {
            // The weight's storage scheme IS part of the class: a 4-bit and an
            // 8-bit affine matmul are different programs. Without this, OptiQ's
            // mixed-precision layers all hash alike and re-roll onto one body.
            SubOp::MatmulTile { n, weight } => {
                n.hash(h);
                std::mem::discriminant(weight).hash(h);
                if let crate::lower::GemmWeight::Affine { affine } = weight {
                    affine.bits().get().hash(h);
                    affine.group().get().hash(h);
                }
            }
            // No attributes: the discriminant is the whole class.
            SubOp::SumReduce { .. }
            | SubOp::SiluMul
            | SubOp::ScalarWeightMul
            | SubOp::GateApply
            | SubOp::GateScale
            | SubOp::VisionRope
            | SubOp::GatedDeltaNet
            | SubOp::Mean
            | SubOp::AllReduce
            | SubOp::RouteSoftmax
            | SubOp::RouteArgsort
            | SubOp::RouteGatherScores
            | SubOp::RouteRenorm
            | SubOp::ExpertUnsort => {}
            // The whole width a gather restores is class: two widths are two exchanges.
            SubOp::AllGather { cols } => cols.hash(h),
            // The vocab and the row count are class: two vocab widths are
            // different programs (every staged table and the whole row's tile
            // shape key off it), and so are two batch widths (the grid and
            // every per-row operand's shape key off it).
            SubOp::Sample { vocab, rows } => (vocab, rows).hash(h),
            // The gathered row's width AND the row it replaced are class: two
            // hidden widths are different programs (the walk's stride and the
            // row's extent key off them), and two replaced rows are different
            // wirings (the stamp's resident excuse keys off the stated id).
            SubOp::EmbedRowGather { hidden, replaces } => (hidden, replaces.0).hash(h),
            // The table range and the entry split are class: two pad-vocabs
            // or two entry counts are different programs (the staged tables'
            // extents and lane patterns key off them).
            SubOp::ChainStage {
                pad_vocab,
                table_rows,
                k,
            } => (pad_vocab, table_rows, k).hash(h),
            SubOp::Mamba2 { inner } => inner.hash(h),
            SubOp::RouterNorm { eps, router } => (eps.to_bits(), router).hash(h),
            SubOp::RouterLogits { experts, router } => (experts, router).hash(h),
            SubOp::RouteSigmoidBias { router } => router.hash(h),
            SubOp::RouteBias { router } => router.hash(h),
            SubOp::RouteTopK { k } => k.hash(h),
            SubOp::RouteScale { scale } => scale.to_bits().hash(h),
            SubOp::RouteExpertScale { router } => router.hash(h),
            SubOp::ExpertSort { experts, k, bundle } => (experts, k, bundle).hash(h),
            // The per-layer expert width is class: an 8-bit and a 4-bit layer are different programs.
            SubOp::ExpertMatmul {
                proj,
                n,
                k,
                quant,
                bundle,
            } => (proj, n, k, quant.group().get(), quant.bits().get(), bundle).hash(h),
            SubOp::ExpertGatedAct { act } => act.hash(h),
            SubOp::ExpertAct { act } => act.hash(h),
            SubOp::ExpertCombine { hidden, shared } => (hidden, shared).hash(h),
            SubOp::KvEncode { operand } | SubOp::KvStage { operand } => operand.hash(h),
            SubOp::RotateRows { rows } => rows.hash(h),
            SubOp::AttnPackedKv
            | SubOp::SampleRowsGather
            | SubOp::SampleRowsScatter
            | SubOp::AllRowsMatmul => {}
            // The row scale is a fact of the VIEW's shape class (per-head vs
            // merger), not per-layer identity — a sliding layer's per-head
            // view and a global layer's re-roll onto different extents than
            // its cells claim.
            SubOp::Reshape { rows, cols } => (rows, cols).hash(h),
            SubOp::Elementwise(k) => std::mem::discriminant(k).hash(h),
            SubOp::ScalarMul { scale } => scale.to_bits().hash(h),
            SubOp::TanhSoftCap { cap } => cap.to_bits().hash(h),
            SubOp::RmsNorm { eps, gain } => {
                eps.to_bits().hash(h);
                std::mem::discriminant(gain).hash(h);
            }
            SubOp::RmsNormReduce { eps, .. } | SubOp::RmsNormUnit { eps } => eps.to_bits().hash(h),
            SubOp::RmsNormApply { gain, .. } => std::mem::discriminant(gain).hash(h),
            SubOp::RopeRotate { head_dim, .. } => head_dim.get().hash(h),
            // `layer` and `layout`'s cache tensors are per-layer IDENTITY, not class.
            SubOp::RopeAppend {
                head_dim,
                attn,
                pairing,
                ..
            } => {
                head_dim.get().hash(h);
                std::mem::discriminant(attn).hash(h);
                pairing.hash(h);
            }
            SubOp::KvAppend { head_dim, attn, .. } => {
                head_dim.get().hash(h);
                std::mem::discriminant(attn).hash(h);
            }
            // `layout`, `producer`'s node index and `softmax_state` are per-layer identity;
            // `mask` is the class (sliding-window vs full) and is exactly what gemma-3 needs —
            // and the WINDOW is a model constant, so it hashes with the class rather than
            // splitting one model's layers across hash buckets.
            SubOp::AttnDecode {
                geom,
                scale,
                valid_len,
                producer,
                mask,
                ..
            } => {
                geom.nqh().get().hash(h);
                geom.nkvh().get().hash(h);
                geom.hd().get().hash(h);
                scale.to_bits().hash(h);
                valid_len.hash(h);
                std::mem::discriminant(producer).hash(h);
                std::mem::discriminant(mask).hash(h);
                if let Some(w) = mask.window() {
                    w.get().hash(h);
                }
            }
            SubOp::GateSplit { half_cols } => half_cols.hash(h),
            SubOp::Concat { cols } => cols.hash(h),
            SubOp::LoadRows { source, width } => {
                source.hash(h);
                width.hash(h);
            }
            SubOp::EmbeddingGather { indices_kind } => indices_kind.hash(h),
            SubOp::VarlenAttention { cu_kind } => cu_kind.hash(h),
            SubOp::EncoderAttn { geom, scale } => {
                geom.nqh().get().hash(h);
                geom.nkvh().get().hash(h);
                geom.hd().get().hash(h);
                scale.to_bits().hash(h);
            }
        }
    }
}

/// The ops whose lowering binds a witness, as a front end states them.
impl<F: RopeForm> SubOp<F, Arch> {
    /// `RopeRotate` in the graph's pairing form `F`.
    pub const fn rope_rotate(head_dim: HeadDim) -> Self {
        SubOp::RopeRotate {
            head_dim,
            _form: PhantomData,
        }
    }

    /// `RopeAppend` for `layer`: the cache tensors it writes are bound by [`lower_region`].
    pub const fn rope_append(
        head_dim: HeadDim,
        layer: u32,
        attn: AttnMask,
        pairing: RopeFormTag,
    ) -> Self {
        SubOp::RopeAppend {
            head_dim,
            layer,
            attn,
            pairing,
            layout: Unbound,
            _form: PhantomData,
        }
    }

    /// `KvAppend` for `layer`: the cache tensors it writes are bound by [`lower_region`].
    pub const fn kv_append(head_dim: HeadDim, layer: u32, attn: AttnMask) -> Self {
        SubOp::KvAppend {
            head_dim,
            layer,
            attn,
            layout: Unbound,
        }
    }

    /// `AttnDecode`: its cache, that cache's producer and its softmax accumulator are bound by
    /// [`lower_region`].
    pub const fn attn_decode(
        geom: ModelAttnGeometry,
        scale: f32,
        valid_len: u32,
        mask: AttnMask,
    ) -> Self {
        SubOp::AttnDecode {
            geom,
            scale,
            valid_len,
            layout: Unbound,
            producer: Unbound,
            softmax_state: Unbound,
            mask,
        }
    }
}

/// THE op vocabulary: what the front end states a forward's ops as, and what every node of a
/// lowered [`SubtileIR`] performs. Generic over `F: RopeForm` (default [`NeoX`]) so a SubtileIR's
/// rope nodes share the same pairing form by construction, and over `S: OpStage` (default
/// [`Tiled`]) — the [`Arch`] stage is the front end's, before [`lower_region`] binds the KV-cache
/// witnesses. Every fact the registry derives (kind, name, arity, output width) is declared once,
/// in [`crate::ops`].
///
/// Every variant has a `cpu_golden`-backed host evaluation in
/// [`eval_node`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SubOp<F: RopeForm = NeoX, S: OpStage = Tiled> {
    /// Matmul output tile over one K-chunk:
    /// `out[i, j] = Σ_l A[i, l] · W[l, j]`.
    /// `inputs[0]` = A slice `[mr, kr]`; `inputs[1]` = W slice `[kr, nr]`
    /// (W is row-major `[K, N]` per the FUF convention `gemm(x:
    /// [..., K], w: [K, N])`). Output is the dense partial
    /// `[mr.len, nr.len]` contributed by this K-chunk.
    ///
    /// `weight` is the weight's storage SCHEME, carried on the node because it
    /// is part of what this matmul *is*: a 4-bit and an 8-bit affine matmul are
    /// different programs, and without it here they hash to the same re-roll
    /// class (see `reroll_class_key`) and a mixed-precision model re-rolls
    /// layers that do not match. It is not the operand arity — fp8 is arity-3
    /// and affine is arity-2, but `Dense` and `Affine` are both arity-2.
    ///
    /// `n` is the Linear's full output width N — the op's output columns as the front end
    /// states it. A tile of the matmul keeps its Linear's `n`.
    MatmulTile {
        n: u32,
        weight: crate::lower::GemmWeight,
    },
    /// Elementwise sum of equal-shaped inputs — the split-K combine. All
    /// inputs and the output are `[out_rows, out_cols]`.
    SumReduce { minted: S::Minted },
    /// Re-interpret `inputs[0]`'s elements under a new `[out_rows, out_cols]`.
    /// Element COUNT is preserved and row-major order is preserved; only the
    /// extents change. Host semantics are the identity.
    ///
    /// The target extents are a function of the TOKEN COUNT:
    /// `out_rows = m · rows_mult / rows_div`, `out_cols = cols` — the same
    /// fact [`crate::lower::LoweredOp::Reshape`] carries, threaded here so
    /// the node's regions can state the TRUE re-laid extents (a per-head
    /// view `[m, heads·hd] → [m·heads, hd]` MULTIPLIES rows; the vision
    /// patch merger DIVIDES). Without it the node understates the output by
    /// the mult factor and every downstream region (the per-head norms, the
    /// placement pool) inherits the lie.
    ///
    /// ⛔ IT IS NOT FREE ON A STICK-LAID-OUT DEVICE, and that is the whole
    /// reason it is an OP rather than a view. `sdsc_abstract::dev_off_stk`
    /// places element `(i, j)` at `(j/stk)*(a*stk) + i*stk + (j%stk)`, where
    /// `a` is the ROW COUNT — so changing the extents moves every element.
    /// A target must lower this to a RESTICKIFY (a real re-laying copy); it
    /// must NOT give the output the input's placement. Pinned by
    /// `sdsc_abstract::tests::reshape_is_not_a_placement_alias_under_stick_layout`.
    ///
    /// The target extents: `rows` relative to the token count (per-head views multiply, the
    /// vision patch merger divides) and `cols`. Per-head norms read their row multiplier here.
    Reshape { rows: RowScale, cols: u32 },
    /// Shape-preserving elementwise op over a tile. Unary (`Silu`) reads
    /// `inputs[0]`; binary (`Mul`, `Add`) read `inputs[0]` and
    /// `inputs[1]`, both matching the output shape. Col-tiling never
    /// reorders a computation, so always bit-exact vs the whole op.
    Elementwise(EwKind),
    /// Fused SwiGLU activation: `out[j] = silu(gate[j]) * up[j]`.
    /// `inputs[0]` = gate, `inputs[1]` = up, both `[out_rows, out_cols]`.
    /// Matches `cpu_golden::fused_gate_up_silu_mul`. The GPU has only a
    /// *fused* `silu_mul` arm (no standalone silu), so the MLP's separate
    /// `Silu` + `Mul` are fused into this one node *before scheduling*
    /// (so the pair is one node in the dataflow graph and downstream
    /// schedulers treat it atomically); see `crate::lower::fuse_silu_mul`.
    SiluMul,
    /// `out[j] = x[j] * scale` — shape-preserving multiply by a compile-time
    /// constant. `inputs[0]` = x `[rows, cols]`; the scalar rides in the field,
    /// not as an operand. Emitted for architectures with scalar activation multipliers (Granite's
    /// `embedding_multiplier` / `residual_multiplier` / `recip(logits_scaling)`).
    /// Col-tiling is bit-exact (per-element scale), so it joins the elementwise
    /// tiling set. Mirrors `ir::Instruction::ScalarMul(_, _, f32)`.
    ScalarMul { scale: f32 },
    /// RMS-norm over each row: `out[i] = x[i] / rms(x[i,:]) * weight`,
    /// `rms = sqrt(mean(x²) + eps)`. `inputs[0]` = x `[rows, cols]`,
    /// `inputs[1]` = weight `[1, cols]`.
    ///
    /// **Whole** form. Lower-stage decomposes into [`SubOp::RmsNormReduce`]
    /// (one reduce node) + N [`SubOp::RmsNormApply`] nodes (one per
    /// chunk of `cols`) when `nb < cols`; the whole form is the op the front end
    /// states, and the canonical IR-level op for the eval_node host reference
    /// that pre-decomposition callers (validate-only tests, partition lowering)
    /// consume.
    RmsNorm { eps: f32, gain: GainConvention },
    /// Phase 1 of chunked RmsNorm: reads x `[m, hidden]` (whole),
    /// computes `inv_rms[i] = 1 / sqrt(mean(x[i,:]^2) + eps)`. Output
    /// region is `[m, 1]` — a per-row scalar (the inv-RMS vector). The
    /// host eval is identical to `1 / sqrt(mean_sq + eps)` over the
    /// whole input row.
    ///
    /// `inputs[0]` = x `[m, hidden]`. The lower_compute side channels
    /// the inv_rms vec via `LoweringState` so that downstream
    /// [`SubOp::RmsNormApply`] nodes can broadcast it without a tile-
    /// shaped re-store.
    RmsNormReduce { eps: f32, minted: S::Minted },
    /// Phase 2 of chunked RmsNorm: per-chunk apply.
    /// `inputs[0]` = x chunk `[m, chunk_cols]`,
    /// `inputs[1]` = inv_rms `[m, 1]` (the producing
    /// [`SubOp::RmsNormReduce`] node's output region; lower_compute
    /// reads the side-channelled smem-vec slot, not this tile-region),
    /// `inputs[2]` = gamma chunk `[1, chunk_cols]`.
    /// Output: `y[m, chunk_cols] = x * inv_rms[per row] * gamma[per col]`.
    RmsNormApply {
        gain: GainConvention,
        minted: S::Minted,
    },
    /// Rotary embedding over `[rows, heads * head_dim]` in the
    /// `F: RopeForm` pairing. `inputs[0]` = x, `inputs[1]` = cos row,
    /// `inputs[2]` = sin row.
    RopeRotate {
        head_dim: HeadDim,
        #[doc(hidden)]
        _form: PhantomData<F>,
    },
    /// The K-side `rope_append`: rotate K in the `F: RopeForm` pairing
    /// **and** write the rotated K + un-rotated V into the paged KV
    /// cache (the cache identity is bound by `layout`). The host eval is
    /// **rotation only** (identical to [`SubOp::RopeRotate`]); V and the
    /// cache write are GPU-only. `inputs` = `[K, cos, sin, V, K_cache, V_cache]`.
    RopeAppend {
        head_dim: HeadDim,
        layer: u32,
        /// The class of the layer whose K/V this appends — the value its attention's `mask`
        /// carries. On a hybrid sliding/global arch it selects the rotary table and head
        /// geometry; `Causal` on a uniform model.
        attn: AttnMask,
        /// The pairing the checkpoint declares (commandr interleaves). [`lower_region`] builds
        /// every graph in the NeoX form `F`, so a target with an interleaved kernel reads the
        /// checkpoint's pairing here.
        pairing: RopeFormTag,
        layout: S::KvLayout,
        #[doc(hidden)]
        _form: PhantomData<F>,
    },
    /// The KV-cache writer of an attention with no positional rotation
    /// (Nemotron-H): write K and V, as projected, into the paged KV cache
    /// (the cache identity is bound by `layout`). The host eval is the
    /// identity on K; the cache write is GPU-only.
    /// `inputs` = `[K, V, K_cache, V_cache]`.
    KvAppend {
        head_dim: HeadDim,
        layer: u32,
        /// The class of the layer whose K/V this appends — the value its
        /// attention's `mask` carries.
        attn: AttnMask,
        layout: S::KvLayout,
    },
    /// Decode attention. `inputs[0]` = Q `[Mq, geom.q_width()]`;
    /// the remaining inputs are alternating `(K_seg, V_seg)` pairs.
    /// `layout` names the K-cache (single source of truth shared with
    /// the producing `RopeAppend`); `producer` names how that cache got
    /// populated; `softmax_state` is the per-AttnDecode online-softmax
    /// accumulator id.
    AttnDecode {
        /// The model's head geometry, minted once from its `config.json` and carried whole. Three
        /// `u32` fields here is how a query-head count and a kv-head count came to be swappable at
        /// a struct literal, and how the GQA group size came to be re-divided (with a `max(1)`) by
        /// every consumer that wanted it.
        geom: ModelAttnGeometry,
        scale: f32,
        /// Logical valid sequence length = `decode_position + 1`. The
        /// number of cache positions that hold real (prefix ++ new) K/V;
        /// the cache TENSOR may be larger (capacity > valid_len). For a
        /// `SameForwardRopeAppend` producer this slices the prefix segment
        /// to `valid_len - 1` rows so `eval_node` attends to exactly
        /// `valid_len` positions independent of the cache row count (the
        /// un-rigged host-verification pin). The GPU mask uses the runtime
        /// `DecodePosition` kernel arg (= `valid_len - 1`).
        valid_len: u32,
        layout: S::KvLayout,
        producer: S::KvProducer,
        softmax_state: S::SoftmaxState,
        mask: AttnMask,
    },

    // ── The rest of the arch vocabulary ────────────────────────────
    //
    // ⭐ THESE EXIST SO THE SHARED FRONT END CAN EXPRESS EVERY ARCH, NOT SO EVERY TARGET CAN RUN
    // THEM. Each used to be a `panic!("has no SubOp form on this pipeline yet")` in
    // `lower_region`, which meant an arch using one had no SubtileIR at all and therefore no
    // tape — so a target consuming the tape simply could not compile that arch, whatever its
    // kernels could do. Representing the op moves the "can this run here?" question to the ONE
    // place that can answer it: the target's opcode lowering.
    /// `cap · tanh(x / cap)` — Gemma's logit/attention soft cap.
    ///
    /// ⭐ THE CAP IS ON THE TAPE, following the `RmsNormUnit { eps }` precedent: a model constant
    /// the device-side decomposition needs (`x / cap`, `· cap`) is carried on the op, minted from
    /// `model.scalars["final_logit_softcapping"]` at the bridge, so `eval_node`'s host reference
    /// and every target's emitter read ONE value rather than the front end's own knowledge of the
    /// config key.
    TanhSoftCap { cap: f32 },
    /// Unit-gain RmsNorm — no learnable scale (Gemma4 `v_norm`). `inputs[0]` = x.
    RmsNormUnit { eps: f32 },
    /// Multiply by a loaded `[1]`-shaped weight (Gemma4 `layer_scalar[layer]`).
    /// `inputs` = `[x, weight]`.
    ScalarWeightMul,
    /// Qwen3.5 attention output-gate split: deinterleave the doubled q_proj output per head into
    /// `[query | gate]`. `inputs` = `[qg]`; TWO outputs (q at this op's width, gate second).
    GateSplit { half_cols: u32 },
    /// Qwen3.5 attention output gate: `out = attn · σ(gate)`. `inputs` = `[attn, gate]`.
    GateApply,
    /// Row concatenation of two same-width operands: `out[t] = a[t] ++ b[t]`, output `cols`
    /// wide (each operand `cols / 2`). `inputs` = `[a, b]`.
    Concat { cols: u32 },
    /// Qwen3.5-MoE shared-expert combine: `out = routed + shared · σ(g)`, `g` a `[m, 1]`
    /// row-broadcast. `inputs` = `[routed, shared, g]`.
    GateScale,
    /// Host-staged rows: the runtime buffer `source` names (patchified pixels, interpolated
    /// position embeddings, a target model's hidden states) lands in this op's slot. NO
    /// inputs; output `[T, width]`.
    LoadRows {
        source: scratchy_ir::RowsExtern,
        width: u32,
    },
    /// Row permutation by a host-staged index table — Qwen2.5-VL window attention gathers into
    /// window order on encoder entry (`indices_kind` 0) and back after the merger (1). The table
    /// is a runtime input, not an operand, so this takes ONE input.
    EmbeddingGather { indices_kind: u8 },
    /// Vision 2D rope pair-rotation over q/k, in place. `inputs` = `[q, k]`; TWO outputs.
    VisionRope,
    /// Varlen bidirectional vision attention. `inputs` = `[q, k, v]`; `cu_kind` selects the
    /// cu_seqlens/max_seqlen extern set (0 default / 1 full / 2 window).
    VarlenAttention { cu_kind: u8 },
    /// Bidirectional encoder attention, no KV cache: contiguous Q/K/V from upstream slots,
    /// non-causal. `inputs` = `[q, k, v]`.
    EncoderAttn { geom: ModelAttnGeometry, scale: f32 },
    /// Gated-DeltaNet linear attention (Qwen3.5 / Qwen3-Next).
    /// `inputs` = `[qkv, z, a, b, weight_bundle]`; the conv/ssm state is runtime-ambient per
    /// layer, which is why there is no host reference.
    GatedDeltaNet,
    /// Mamba-2 selective state-space mixer (Nemotron-H). `inputs` = `[zxbcdt, weight_bundle]`,
    /// `zxbcdt` the whole in-projection `z | xBC | dt`; `inner` = `d_inner`, the gated-normed
    /// output's width. The conv/ssm state is runtime-ambient per layer, as GDN's.
    Mamba2 { inner: u32 },
    /// Row mean: `out[m, 1] = mean(x, axis = -1)`. `inputs` = `[x]`.
    Mean,

    // ── Collectives: what tensor parallelism weaves in ──────────────
    //
    // ⛔ NEVER STATED BY A FRONT END. Only `crate::tensor_parallel::shard` inserts these, at
    // exactly the edges where a value's placement does not meet its consumer's demand, so the
    // per-rank forward that carries them is the one the analysis proved equal to the unsharded
    // one. Their operand is a value in rank-local shape; every rank runs the same op.
    /// The sum over every rank of a PARTIAL value: `out = Σ_r x_r`, summed in ABSOLUTE RANK
    /// ORDER (`((x_0 + x_1) + x_2) + …`) on every rank, so the replicated result is
    /// bit-identical across ranks. `inputs` = `[x]`; output the shape of `x`.
    AllReduce,
    /// The concatenation over every rank of a column-SPLIT value: rank `r`'s `[rows, cols / W]`
    /// block lands at columns `[r · cols / W, (r + 1) · cols / W)`. `inputs` = `[x]`; `cols` is the
    /// WHOLE width the gather restores.
    AllGather { cols: u32 },

    // ── A MoE block, op by op (see `NumExperts`) ──────────────────
    /// The router's own pre-norm: `rmsnorm(x, router.scale)`. `inputs` = `[x, router]`.
    RouterNorm { eps: f32, router: RouterBundle },
    /// Router logits `x · W_router`, `[m, experts]`. `inputs` = `[x, router]`.
    RouterLogits {
        experts: NumExperts,
        router: RouterBundle,
    },
    /// Row softmax. `inputs` = `[scores]`.
    RouteSoftmax,
    /// Row sigmoid of the logits plus each expert's correction bias, in place on the logits
    /// (GLM-4 / DeepSeek-V3 `noaux_tc` routing). The BIASED values order the top-k picks; the
    /// buffer keeps the UNBIASED sigmoids, so the standard score gather reads the values mlx
    /// renormalizes. `inputs` = `[logits, router]`.
    RouteSigmoidBias { router: RouterBundle },
    /// The router's linear bias added to its logits, in place (gpt-oss routing). The BIASED
    /// values order the top-k picks AND are read back as the gathered scores — a softmax over
    /// the top-k then normalizes them, so unlike [`Self::RouteSigmoidBias`] the buffer keeps the
    /// values the picks read. `inputs` = `[logits, router]`.
    RouteBias { router: RouterBundle },
    /// Each row's indices sorted by ascending score. `inputs` = `[scores]`.
    RouteArgsort,
    /// The last `k` sorted indices of each row — its top-k experts. `inputs` = `[sorted]`.
    RouteTopK { k: TopK },
    /// The scores at the chosen indices, `[m, k]`. `inputs` = `[scores, indices]`.
    RouteGatherScores,
    /// Scores times a constant (a softmax temperature). `inputs` = `[scores]`.
    RouteScale { scale: f32 },
    /// Scores divided by their row sum. `inputs` = `[scores]`.
    RouteRenorm,
    /// Each score times its expert's learned scale. `inputs` = `[scores, indices, router]`.
    RouteExpertScale { router: RouterBundle },
    /// The (token, expert) pair rows, ordered by expert. `inputs` = `[x, indices]`; the output
    /// is also the routing every expert projection of the block reads.
    ExpertSort {
        experts: NumExperts,
        k: TopK,
        bundle: ExpertBundle,
    },
    /// One projection of each pair's expert, `n` wide. `inputs` = `[rows, routing, experts]`.
    ExpertMatmul {
        proj: ExpertProj,
        n: u32,
        k: TopK,
        quant: crate::lower::ExpertQuant,
        bundle: ExpertBundle,
    },
    /// `act(gate) * up` over the pair rows. `inputs` = `[gate, up]`.
    ExpertGatedAct { act: GatedAct },
    /// `act(up)` over the pair rows of an ungated expert ([`ExpertBody::Ungated`]). `inputs` =
    /// `[up]`.
    ExpertAct { act: UngatedAct },
    /// The pair rows back in token order. `inputs` = `[rows, routing]`.
    ExpertUnsort,
    /// Each token's pair rows summed by its scores, `[m, hidden]`. `inputs` = `[rows, scores]`.
    ExpertCombine {
        hidden: u32,
        shared: SharedExpertBound,
    },

    // ── A KV codec, step by step (see `KvOperand`) ────────────────
    /// A writer's new `operand` rows, encoded into the layer's packed store, their additive
    /// offset removed first. `inputs` = `[cache, cos?, bias?]`: the cache half the writer filled,
    /// a rotating writer's rotary row, and the projection whose bias the operand carries.
    KvEncode { operand: KvOperand },
    /// The layer's `operand`, decoded out of its packed store into the codec's rotated domain,
    /// where the attention reads it. `inputs` = `[packed]`.
    KvStage { operand: KvOperand },
    /// `rows` turned into or out of the codebook's domain, in place. `inputs` = `[x]`.
    RotateRows { rows: RotatedRows },
    /// An attention read straight off the packed store: its query, the output it replaces, and
    /// the store's two halves. `inputs` = `[q, out, packed_k, packed_v]`, optionally with the
    /// attention's trailing `[num_heads]` sinks weight source (the dense form's 6th operand).
    AttnPackedKv,

    // ── The sampled rows of a result matmul (see `sample_rows`) ───
    /// The step's sampled rows of `x` moved to its front, in order, in place: the result matmul's
    /// activation, which the matmul then reads. `inputs` = `[x]`.
    SampleRowsGather,
    /// What runs after the sampled matmul, in place over its output `x` — one of two, by how the
    /// target ran the matmul: the gathered activation put back, where it ran over the sampled
    /// rows alone (their logits already at the front); or every row's logits compacted to the
    /// sampled rows', where it ran over every row. Either way the sampled rows' logits are at the
    /// front of `x` after it. `inputs` = `[x]`.
    SampleRowsScatter,
    /// The sampled matmul over EVERY row, taking over its scattered output: the alternative a
    /// step that reads every row runs instead of the gather, the matmul and the scatter — never
    /// on a step they run on, so its rows are the activation untouched.
    /// `inputs` = `[rows, over, weight…]`: the gathered activation, the output it replaces, and
    /// the matmul's weight operands.
    AllRowsMatmul,

    // ── The token draw (see `kernels/sample_fwd.py`) ─────────────────
    /// The FIRST-maximum Gumbel-max draw over a `[rows, PAD_V]` logits block,
    /// as the `[rows, BLOCK]` i32 idx stick the device's index ingestion
    /// reads — one program per row, at EVERY decode rung (the solo m=1 rung
    /// and every batched rung alike; a batched rung without the sampler does
    /// not build). `inputs` = `[logits, slo, shi, stp, sgate, invt,
    /// bramp, lramp, binc, linc, ones, sixtyfour, padoff, gramp, ginc, hc1,
    /// hc2, hc3, c256, lane, fm8, ftop, fzero, fhalf, fm22, cmin]`: the
    /// `[rows, PAD_V]` fp16 block (whose pad bytes are NOT this op's
    /// contract — the lm-head's zero-pad columns write 0.0 there every
    /// step), the FIVE PER-ROW, PER-STEP staged operands (the seed
    /// quadruple's four `[rows, 1]` i32 word tables — seed_lo, seed_hi,
    /// step, greedy_gate — from which the card computes the noise
    /// itself, zeros
    /// at the greedy temperature 0 — and the `[rows, 1]` f32 temperature
    /// reciprocals), then the twenty staged tables, functions of `vocab`
    /// alone, constant across decode steps — the two decreasing ramps that
    /// make the maxima first-max by monotonicity, the two increasing index
    /// tables that read the selection back out, the two widen tables that
    /// keep every broadcast a two-tensor multiply, the OFFSET row that
    /// carries the pad law (0.0 on the real lanes, f32::MIN (−FLT_MAX,
    /// 0xFF7FFFFF) on the pads — the fp32 reduce template's own "negInf",
    /// NEVER IEEE −inf, which the card reads as NaN; the kernel adds it
    /// LAST, after the score's two roundings, so a pad lane can never win
    /// and an all-−FLT_MAX row still answers lane 0), the two-level
    /// ramp's GROUP pair, and the noise section's own eleven (the four i32
    /// hash/fold constants, the lane table, the fold's five f32 constants,
    /// the fp16 island's clamp — ten of them ONE-WORD `[1, 1]` tables, the
    /// wall-5 measured law). An all-NaN row answers past the last real lane (the
    /// sentinel); the consumer refuses `idx > vocab` loudly rather than feed
    /// a NaN-born index to the detokenizer.
    Sample { vocab: u32, rows: u32 },
    // ── The chained head (constructed only by the chain mint, `mint_embed_chain`) ──
    /// ⭐ THE CHAINED INPUT ROWS — `[width, hidden]` embedding rows gathered out
    /// of the (tied) embedding matrix at the indices the DEVICE picked, one row
    /// per request: `inputs` = `[s_stick, l_stick, embed_matrix]`, out
    /// `[width, hidden]`.
    ///
    /// ⛔ THIS IS NOT `EmbeddingGather`, AND THE DIFFERENCE IS THE CLASS:
    /// `EmbeddingGather`'s index table is a HOST-STAGED extern (a runtime
    /// input, not an operand — the vision window permutation); this op's
    /// indices are OPERAND REGIONS over device-written placements — the
    /// persistent s/l sticks the chain mint inserts, written by the previous
    /// step's `ChainStage` through its staged tables. The gather cannot tell
    /// host-staged bytes from device-written bytes (a placement is a
    /// placement); what makes this the chain's op is that its indices NEVER
    /// left the card.
    ///
    /// The pick DECOMPOSES across the two sticks, because the pick `t`
    /// reaches `PAD_V + 4160` — past fp16 range, and the sticks' lanes must
    /// stay fp16-exact (a staged u32 word re-encoded as fp16 would rewrite
    /// the index). `t = s<<6 | l`: the s stick carries the SLAB index
    /// `t >> 6` (clamped to 0 past `pad_vocab`) as the LANE PATTERN
    /// [`chain_s_stick_pattern`] — lane `i` names entry `k·s + (i mod k)` of
    /// the slab gather's `k`-entry tile — and the l stick carries the LANE
    /// RAMP [`chain_l_stick_ramp`] — word `j` names source stick
    /// `64·(j mod entries) + l`. Word 0 of each row re-joins the pick
    /// exactly (`s = word0/k`, `l = word0`): the HOST reference's law, and
    /// the doors' own first-word reads.
    ///
    /// The matrix operand is the lm-head's own weight source (embeddings
    /// TIED — the admission's fact, checked where the model config lives):
    /// `[hidden, PAD_V]` row-major with its stated cols WIDENED to the pick's
    /// pad ([`widen_tied_embed_weight`]'s law), so token `t`'s embedding is
    /// column `t` — `out[h] = W[h·pad + t]`, a stride-`pad` walk. `hidden` is
    /// the op's own field, never a shape read: the emitted width is the law.
    /// `replaces` is STATED METADATA, never an input: the host-bound embed
    /// rows this gather replaces. A fourth input would fake a read edge on a
    /// tensor nothing reads anymore (the re-point's own law) and falsify the
    /// zero-consumers accounting; the stated id is what the resident stamp's
    /// excuse keys off.
    ///
    /// The sentinel law: a broken logits row can answer an index PAST the
    /// vocab (the pick op's sentinel contract); the l stick then carries the
    /// SENTINEL RAMP ([`chain_l_sentinel_ramp`] — word 0 the mark `hidden`,
    /// every word naming a stick in the transposed placement's never-written
    /// zero tail), and the gather stages a ZERO row rather than read past
    /// the matrix — the refusal the host readback's `idx > vocab` check
    /// performed, moved into the op, so an already-broken chat cannot also
    /// fault the device bus.
    EmbedRowGather { hidden: u32, replaces: TensorId },
    /// ⭐ THE PICK STAGED INTO THE CHAIN — the persistent placements' writer:
    /// `inputs` = `[idx_stick, s_stick, l_stick, s_tab, l_tab]`, where the
    /// sticks are the persistent SOURCES the mint inserted (the same
    /// producer-writes/consumer-reads class as `RopeAppend`'s cache inputs —
    /// placements the device owns after the host seeds them once) and the
    /// tables are ORDINARY TENSOR OPERANDS staged once at load (the mint
    /// states them, this node consumes them, the worker stages their bytes
    /// through the same load-const list every staged table rides). The pick
    /// `t` rides `idx_stick`'s word 0 per row; the decompose is two u32
    /// table gathers — `s_tab[t] = t >> 6` clamped past `pad_vocab`,
    /// `l_tab[t]` the lane ramp at `t & 63` (the sentinel rows past `pad`
    /// carry the sentinel ramp) — written into the persistent sticks,
    /// GPU-only (`KvAppend`'s law). The output is a fresh s-value stick
    /// `[width, CHAIN_IDX_BLOCK]` (every op's output its own): the pattern
    /// `k·s + (i mod k)` the device's own s-gather writes, so the two ends
    /// agree on what the node produced.
    ///
    /// `pad_vocab` and `table_rows` state the DECOMPOSE'S OWN EXTENT —
    /// `table_rows = pad_vocab + SAMPLE_SENTINEL_OFFSET + 1`, every
    /// answerable pick up to the sentinel INCLUSIVE, the sentinel the LAST
    /// row — the pick's own mint's derivation, never re-derived from
    /// `pad_vocab` a second time. `k` is the SLAB-GATHER ENTRY COUNT
    /// ([`chain_slab_entries`]'s answer at this graph's hidden), stated so
    /// the fill, the doors, and both reference arms read one fact.
    ChainStage {
        pad_vocab: u32,
        table_rows: u32,
        k: u32,
    },
}

/// The token draw's padded row: the vocab rounded up to a power-of-two whole
/// row, STRICTLY ABOVE the vocab — the descriptor block's own law (the block
/// must be a power of two) AND the sentinel law's (a power-of-two vocab still
/// needs pad lanes to land its all-NaN one-hot on, so a vocab that is already
/// a power of two rounds to the NEXT one up: gemma-4's 262144 pads to
/// 524288). ONE derivation, read by the host reference, the splice's
/// constexpr flow, and the load-time table staging, so no two of them can
/// disagree.
pub fn sample_pad_vocab(vocab: u32) -> u32 {
    let pad = vocab.max(1).next_power_of_two();
    if pad == vocab { pad * 2 } else { pad }
}

/// The token draw's stick width — the idx stick's own lane count and the
/// per-lane tables' (`ones`, `sixtyfour`) width. ONE derivation for every
/// reader: the registry's `[stick]` out-cols class, the splice's `NB`/`BLOCK`
/// constexprs, the host reference's sentinel arithmetic, and the mint's
/// output tensor — four sites that must all move together if the stick ever
/// widens.
pub const SAMPLE_BLOCK: u32 = 64;

/// The token draw's GROUP width — the number of 64-lane blocks per group in
/// the two-level ramp. The staged block ramp is PERIODIC with this period
/// (strictly decreasing within each group, restarting per group), so its lane
/// values top out at 64 — exact at ANY vocab — while the group ramp above it
/// tops out at the group count, which the admission gate keeps ≤ 2048 (exact
/// in the f32 selection). ONE derivation shared by the staged tables, the
/// admission gate, and the sentinel arithmetic.
pub const SAMPLE_GROUP_BLOCKS: u32 = 64;

/// The token draw's group count — `pad_vocab` split into
/// [`SAMPLE_GROUP_BLOCKS`]-block groups. The two-level ramp's outer level and
/// the group index table's ceiling.
pub fn sample_groups(pad_vocab: u32) -> u32 {
    pad_vocab / (SAMPLE_BLOCK * SAMPLE_GROUP_BLOCKS)
}

/// The token draw's SEED WORD COUNT — the per-row quadruple
/// `(seed_lo, seed_hi, step, greedy_gate)`, minted as FOUR separate
/// `[rows, 1]` i32 word tables (each `invt`'s own class: one stick per row,
/// the word at the stick head, loaded corner-free at `[pid, 0]` — the
/// int-bcast card leg's measured Col form, and the only spelling the Col
/// operand mode can address: it reads the stick's word 0 and has no
/// within-stick offset). ONE derivation for the mint's four word sources,
/// the kernel's own four `[B, 1]` descriptors (`sample_fwd.py`'s
/// signature), and the host reference's quadruple read — sites that must
/// all move together. (The fold's first spelling — one `[rows, 5]` table
/// read through five `[1, 1]` windows — carried column corners the integer
/// door refuses by name; the auditor's wall-4 ruling (a) re-spelled it as
/// corner-free tables. ⭐ THE ROW WORD IS GONE FOR GOOD — auditor ruling
/// 2026-10-27, the P3 localization: the first spelling keyed the hash on
/// the quintuple's fifth word, the LAUNCH row index, so a concurrent
/// request's noise depended on the slot it happened to land in and could
/// never equal its own solo text. The key is request-owned alone — the
/// seed and the request's own step count.)
pub const SAMPLE_SEED_WORDS: u32 = 4;

// ── Nodes & graph ──────────────────────────────────────────────────

/// One unit of work: reads `inputs` (regions of tensors), computes its
/// `op`, and writes the result to `output` (a region of one op-output
/// tensor). Produces a dense
/// `[output.region.rows.len, output.region.cols.len]` buffer that is
/// scattered into the output tensor.
#[derive(Clone, Debug)]
pub struct SubtileNode<F: RopeForm = NeoX> {
    pub id: SubtileId,
    pub op: SubOp<F>,
    pub inputs: Vec<TensorRegion>,
    pub output: TensorRegion,
}

/// The canonical wavefront SubtileIR — a tensor-region SSA dataflow
/// graph in the `F: RopeForm` pairing. Nodes are topologically ordered:
/// every node that reads an op-output region is preceded by the nodes
/// that write the overlapping region (so a single pass over `nodes` is
/// a valid evaluation order).
#[derive(Clone, Debug)]
pub struct SubtileIR<F: RopeForm = NeoX> {
    pub tensors: Vec<TensorShape>,
    /// `tensors[0..num_sources]` are leaf sources.
    pub num_sources: u32,
    pub nodes: Vec<SubtileNode<F>>,
    /// The tensor whose buffer is the forward result (logits).
    pub result: TensorId,
    /// PROVENANCE: `op_output[j]` is the tensor written by source op `j`
    /// of the `LoweringInput` this IR was lowered from.
    ///
    /// ⭐ A LOWERING IS A MAP, AND THIS IS THE MAP. A target that plays
    /// the tape reads its schedule and its slot coloring from the tape,
    /// but the WEIGHTS a step binds live on the source op's operands — its
    /// accessor path, its quant preset, its scale. Without this, a tape
    /// player can say what to compute and where to put it but not what
    /// to multiply by, and the only way back is to re-derive the map by
    /// re-running the lowering, which is a second implementation of it.
    ///
    /// The rewrites (`decompose_rmsnorm`, `head_tile_rope`) add nodes
    /// but never retire an op's output tensor, so they carry it through
    /// unchanged.
    pub op_output: Vec<TensorId>,
}

impl<F: RopeForm> SubtileIR<F> {
    pub fn shape(&self, t: TensorId) -> TensorShape {
        self.tensors[t.0 as usize]
    }
    pub fn is_source(&self, t: TensorId) -> bool {
        t.0 < self.num_sources
    }
    /// The rope form of this IR. All rope nodes use this pairing by
    /// construction (the const generic guarantees it).
    pub const fn rope_form(&self) -> RopeFormTag {
        F::TAG
    }
}

// ── The token draw's mint (see `kernels/sample_fwd.py`) ───────────

/// What [`mint_sample`] inserted, for the caller's bake log and the wiring.
#[derive(Clone, Copy, Debug)]
pub struct SampleMint {
    /// The EMITTED vocab — the result row's own width, the sentinel threshold
    /// the consumer refuses past. Never the model config's word for it.
    pub vocab: u32,
    /// `sample_pad_vocab(vocab)` — the width the result row was widened to
    /// and the whole-row tables share.
    pub pad_vocab: u32,
    /// The row count the rung serves — the node's grid width and every
    /// per-row operand's row count. The rung's OWN rows, read off its result
    /// tensor: the solo m=1 rung mints at 1, every batched rung at its own
    /// width.
    pub rows: u32,
    /// The minted SEED WORD sources' tids — the FOUR `[rows, 1]` i32 per-row
    /// word tables of the quadruple `(seed_lo, seed_hi, step, greedy_gate)`,
    /// in that order, each `invt`'s own class (one stick per row, the
    /// word at the stick head) and staged EVERY STEP by the worker (the
    /// noise itself is computed on the card from them). The wiring names them
    /// to the worker through the mint's facts; they are never load-time
    /// consts.
    pub seed_words: [TensorId; SAMPLE_SEED_WORDS as usize],
    /// The minted INVT source's tid — the `[rows, 1]` f32 temperature
    /// reciprocals, staged every step with the seed words.
    pub invt: TensorId,
}

/// The rows the suffix's lm-head computes — the FOLD LAW the splice already
/// states (`is_prefill_lm_head_tail`), handed to the mint by the caller and
/// never re-derived here. A DECODE graph computes every result row (the solo
/// rung's one, a batch rung's one per request); a PREFILL graph's lm-head tail
/// is FOLDED to one row (the emitter re-lowers the matmul at m=1 over the last
/// prompt row, `lower_prefill_lm_head_fold`), so the draw serves that one row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleRows {
    /// Every result row is computed and drawn — the decode rungs.
    All,
    /// The folded lm-head row alone — the prefill rungs. The Sample node reads
    /// the region the fold writes (the tail matmul's own one-row output region,
    /// the same `one_row` law the emitter's re-lowering keeps), and the seat,
    /// the idx stick and every per-row operand sit at rows = 1 — the proven
    /// solo-decode shape.
    Folded,
}

/// Append the token draw to a graph — every rung's own row count, the solo m=1
/// decode rung, every batched rung, and every prefill rung alike: five per-row
/// per-step staged sources (the seed quadruple's four `[rows, 1]` i32 word
/// tables, invt), twenty staged-table sources, the result tail chain's
/// tensors widened to `[rows, PAD_V]`, the `Sample` node
/// reading the widened block (every row on a decode rung; the folded row alone
/// on a prefill rung — [`SampleRows`]), and the graph's RESULT re-pointed to
/// the node's idx stick. Pure graph surgery — the GATE (which rungs mint, and
/// which rows their suffix computes) is the caller's fact; this fn states only
/// its own invariants and panics on their violation, which at expansion time
/// is a build error, never a runtime one.
///
/// ⭐ THE ANSWER RE-POINT: `result` names the idx stick, so every seat keyed
/// on the result — the wiring's `result` tid, the placement pass's
/// `SegRole::Logits` classification, the worker's readback — addresses the
/// picked tokens, and the dtype the kernel's own signature states for that
/// output (`i32`) rides the wiring as the fact that tells the consumer to
/// read it RAW instead of through the f16 convert. The ROW is NOT deleted:
/// it stays exactly where the mint put it, a placed op output readable by
/// tid — the re-point is the draw's redirect of the ANSWER, never a removal
/// of the row it was drawn from (the test oracle's logits readback is that
/// row's own reader, and it is why the row must survive the mint).
///
/// ⭐ THE TWENTY-FIVE SOURCES ARE INSERTED AT `[num_sources,
/// num_sources+25)`, NOT APPENDED: sources must hold
/// `tensors[0..num_sources]`, so the insert renumbers every op-output
/// `TensorId` up by twenty-five — nodes' input/output regions, `op_output`,
/// and `result` (validate demands the result is an op output, so it always
/// moves). Source ids below the old `num_sources` are UNTOUCHED, which is
/// what keeps the caller's `weight_ids` and the wiring's binding scan valid
/// against the minted graph. The twenty tables carry NO `SourceBinding` and
/// stage at LOAD through the wiring's `load_consts` (the rope tables' law);
/// the four seed word tables and invt carry none either but stage EVERY STEP
/// through the worker's sampler seat (the mint's facts name their tids) —
/// they are minted below the `LoweredDecode` layer like the tables (the
/// `embed_src` law: the answer, not the data to compute it from).
///
/// ⭐ THE WHOLE TAIL CHAIN IS WIDENED, NOT THE RESULT ALONE: the tail's
/// writers are column-CHUNKED at production `nb` (granite's vocab 49155 is
/// seven chunks), and a chunked pointwise node's window law demands every
/// operand's TENSOR hold the same width — widening only the result made the
/// epilogue's chunks disagree (output a PAD_V-wide storage, input a
/// vocab-wide one) and the splice refused the node. The walk is the
/// admission's own tail walk (single-input `ScalarMul`/`TanhSoftCap`
/// epilogues back to the lm-head `MatmulTile`; anything else panics — the
/// admission gate already refused it, so the mint was called outside its
/// gate). Every writer keeps its own region — `[0, vocab)` exactly as before
/// — so the pad tail `[vocab, PAD_V)` is written by no node below, and no
/// staging covers it either: the pad law rides the OFFSET row (a staged
/// table), which the kernel adds to the score LAST, after its two roundings,
/// so the tail's raw bytes (the lm-head's zero-pad 0.0 columns) never reach a
/// max.
///
/// ⭐ THE WIDENING IS THIS RUNG'S OWN: every decode rung mints at its own row
/// count (the solo rung at 1, a batched rung at its baked width), so each
/// graph (and its bundle's seg4) carries its own padded block — the mint
/// reads the row count off the result tensor and never assumes one. A prefill
/// rung mints at the FOLDED row's 1 whatever its `mq` (the suffix computes one
/// logits row; a draw at `mq` would read rows nobody wrote and break the
/// kernel's grid law).
pub fn mint_sample<F: RopeForm>(graph: &mut SubtileIR<F>, suffix: SampleRows) -> SampleMint {
    let result_rows = graph.tensors[graph.result.index()].rows;
    let vocab = graph.tensors[graph.result.index()].cols;
    let pad_vocab = sample_pad_vocab(vocab);
    let rows = match suffix {
        SampleRows::All => result_rows,
        SampleRows::Folded => 1,
    };
    assert!(
        rows >= 1,
        "the sampler serves at least one row; this graph's result tensor has {result_rows} rows"
    );
    // The twenty-five staged sources, in the kernel's operand order after the
    // row: the per-row per-step word tables first (the seed quadruple's FOUR
    // `[rows, 1]` i32 tables and the temperature reciprocals), then the
    // twenty load-time tables — the two decreasing ramps and the two
    // increasing index tables (whole-row `[1, PAD_V]`, shared by every row),
    // the two per-lane widen tables (`[1, BLOCK]`), the pad law's offset row,
    // the two-level ramp's group pair, then the noise section's own eleven:
    // the four i32 hash/fold constants, the fold's five f32 constants and
    // the fp16 island's clamp — ⭐ TEN `[rows, 1]` TABLES, the SEED WORDS'
    // OWN CLASS (the auditor's 2026-10-27 ruling): the same word at every
    // row's stick head, baked by the bake at the rung's own rows, so the
    // worker writes nothing for them and the consuming ops read them through
    // the broadcast operand modes at every band height — never whole-row
    // splats (the wall-5 measured law: ten `[1, PAD_V]` splats are PAD_V·4 B
    // each of local store for one repeated word, 2.5 MiB against the 2 MiB
    // LX) — and the lane table, the one GENUINELY per-lane fact of the
    // eleven, whole-row. The array's LENGTH is the single derivation of
    // the renumber shift below.
    let tables = [
        // The seed quadruple's four word tables — each `[rows, 1]` i32,
        // invt's own class, staged EVERY STEP by the worker (one stick per
        // row, the word at the stick head; the auditor's wall-4 ruling (a)).
        // ⭐ NO ROW WORD (auditor ruling 2026-10-27, the P3 localization):
        // the hash key is request-owned alone — the seed and the request's
        // own step count — never the baked row the request lands in.
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        // invt — [rows, 1] f32, staged every step with the seed words.
        TensorShape { rows, cols: 1 },
        // bramp, lramp — the periodic decreasing ramps.
        TensorShape {
            rows: 1,
            cols: pad_vocab,
        },
        TensorShape {
            rows: 1,
            cols: pad_vocab,
        },
        // binc, linc — the periodic increasing index tables.
        TensorShape {
            rows: 1,
            cols: pad_vocab,
        },
        TensorShape {
            rows: 1,
            cols: pad_vocab,
        },
        // ones, sixtyfour — the [1, BLOCK] widen tables.
        TensorShape {
            rows: 1,
            cols: SAMPLE_BLOCK,
        },
        TensorShape {
            rows: 1,
            cols: SAMPLE_BLOCK,
        },
        // padoff — the pad law's offset row.
        TensorShape {
            rows: 1,
            cols: pad_vocab,
        },
        // gramp, ginc — the two-level ramp's group pair.
        TensorShape {
            rows: 1,
            cols: pad_vocab,
        },
        TensorShape {
            rows: 1,
            cols: pad_vocab,
        },
        // hc1, hc2, hc3, c256 — the i32 hash and wrap-fold constants,
        // `[rows, 1]`: the same word at every row's stick head, the seed
        // words' own class (baked by the bake at the rung's rows; the worker
        // writes nothing for them), broadcast by their consuming ops.
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        // lane — the per-lane index row, staged f32 (the fptosi exit reads
        // it as an integer-valued f32).
        TensorShape {
            rows: 1,
            cols: pad_vocab,
        },
        // fm8, ftop, fzero, fhalf, fm22 — the fold's f32 constants,
        // `[rows, 1]` (the seeds' own class, broadcast operands).
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        TensorShape { rows, cols: 1 },
        // cmin — the fp16 island's clamp (2^-14, the smallest normal),
        // `[rows, 1]`: the `maximum`'s broadcast operand, the seeds' own
        // class.
        TensorShape { rows, cols: 1 },
    ];
    let shift = tables.len() as u32;
    let n_src = graph.num_sources;
    // 1) Renumber every op-output tid up by `shift`, so the insert below
    //    leaves sources at [0, num_sources+shift) and op outputs above them.
    let bump = |tr: &mut TensorRegion| {
        if tr.tensor.0 >= n_src {
            tr.tensor.0 += shift;
        }
    };
    for node in &mut graph.nodes {
        for inp in &mut node.inputs {
            bump(inp);
        }
        bump(&mut node.output);
    }
    for t in &mut graph.op_output {
        if t.0 >= n_src {
            t.0 += shift;
        }
    }
    if graph.result.0 >= n_src {
        graph.result.0 += shift;
    }
    // 2) Insert the twenty-five sources and widen the RESULT TAIL CHAIN to
    //    the padded width — the whole chain, per the mint's own law above. The
    //    ROWS are untouched: the rung keeps its own row count.
    graph.tensors.splice(n_src as usize..n_src as usize, tables);
    graph.num_sources += shift;
    let seed_words: [TensorId; SAMPLE_SEED_WORDS as usize] =
        core::array::from_fn(|k| TensorId(n_src + k as u32));
    let invt = TensorId(n_src + SAMPLE_SEED_WORDS);
    let mut t = graph.result;
    // Widen the RESULT TAIL CHAIN to the padded width — the whole chain, per
    // the mint's own law above. The ROWS are untouched: the rung keeps its own
    // row count, and the fold (the emitter's own `lower_prefill_lm_head_fold`)
    // is the one that states the one-row tail on a prefill rung.
    loop {
        graph.tensors[t.index()].cols = pad_vocab;
        let node = graph
            .nodes
            .iter()
            .find(|n| n.output.tensor == t)
            .expect("the result tail's writer");
        match &node.op {
            SubOp::ScalarMul { .. } | SubOp::TanhSoftCap { .. } if node.inputs.len() == 1 => {
                t = node.inputs[0].tensor;
            }
            SubOp::MatmulTile { .. } => break,
            other => panic!(
                "the device sampling mint walked a {other:?} result tail — the admission \
                 gate (an arity-2 Dense lm-head under single-input ScalarMul/TanhSoftCap) \
                 refuses this graph, so the mint was called outside it"
            ),
        }
    }
    // 3) The idx stick's op-output tensor, then the node reading the whole
    //    widened block, the per-row word tables, and the twenty tables.
    let idx = TensorId(graph.tensors.len() as u32);
    graph.tensors.push(TensorShape {
        rows,
        cols: SAMPLE_BLOCK,
    });
    let whole = |t: TensorId| TensorRegion {
        tensor: t,
        region: Region {
            rows: Range::new(0, graph.tensors[t.index()].rows),
            cols: Range::new(0, graph.tensors[t.index()].cols),
        },
    };
    let mut inputs = Vec::with_capacity(26);
    // The logits the draw reads: the whole result block on an All graph, and
    // the ONE row the folded tail writes on a Folded graph — row 0, the row
    // `lower_prefill_lm_head_fold`'s m=1 matmul writes (the result tensor's
    // DECLARED shape stays [mq, vocab]; only this region states the fold).
    inputs.push(match suffix {
        SampleRows::All => whole(graph.result),
        SampleRows::Folded => TensorRegion {
            tensor: graph.result,
            region: Region {
                rows: Range::new(0, 1),
                cols: Range::new(0, pad_vocab),
            },
        },
    });
    for s in n_src..graph.num_sources {
        inputs.push(whole(TensorId(s)));
    }
    graph.nodes.push(SubtileNode {
        id: SubtileId(graph.nodes.len() as u32),
        op: SubOp::Sample { vocab, rows },
        inputs,
        output: whole(idx),
    });
    // 4) The answer re-point: the rung's result IS the stick. validate
    //    demands the result is an op output — the Sample node's output
    //    qualifies, and the row keeps its own placement and its own readers
    //    either way.
    graph.result = idx;
    SampleMint {
        vocab,
        pad_vocab,
        rows,
        seed_words,
        invt,
    }
}

// ── The chained head's laws (the constants every reader shares) ─────

/// The chained head's index-stick width — the SENUINT32 stick (32 four-byte
/// lanes), the format a gather's INDEX tensor carries on the spyre device.
/// The pick's own stick is [`SAMPLE_BLOCK`] (64 i32 lanes, the pick
/// producer's output); the s/l sticks the chain reads are this NARROWER
/// class because a staged table's gathered destination must be exactly one
/// stick of its format (the one-stick law) and a u32 stick is 32 lanes.
/// ONE derivation for every reader: the registry's `[ustick]` out-cols
/// class, the mint's persistent sources and fresh output, and the host
/// reference's stick width.
pub const CHAIN_IDX_BLOCK: u32 = 32;

// ⭐ THE PICK'S OWN LAWS ARE THE SAMPLER'S, NEVER A SECOND COPY: the pad
// ([`sample_pad_vocab`]), the stick ([`SAMPLE_BLOCK`]), the group split
// ([`sample_groups`]) and the sentinel arithmetic all live with the token
// draw (Front A's card-proven op), and every chain reader points there. The
// #603 tip carried `argmax_*` twins of these facts; they are deleted — two
// copies of one fact WILL drift, and the sentinel arithmetic was checked
// identical before the swap (the offset from the pad is
// `SAMPLE_GROUP_BLOCKS * SAMPLE_BLOCK + SAMPLE_BLOCK` in both laws).

/// The sampler's sentinel OFFSET from the padded row — the distance from
/// `PAD_V` to the all-NaN row's answer `PAD_V + G·4096 + 4160`: the staged
/// tables' `ginc`/`binc`/`linc` combine, `SAMPLE_GROUP_BLOCKS · SAMPLE_BLOCK +
/// SAMPLE_BLOCK` = 4160. The sampler's own sentinel law (the host
/// reference's `None` arm) stated as the offset the chain's table range
/// reads — ONE derivation, no argmax twin.
pub const SAMPLE_SENTINEL_OFFSET: u32 = SAMPLE_GROUP_BLOCKS * SAMPLE_BLOCK + SAMPLE_BLOCK;

/// The chained slab's fp16 stick width — the `[hidden, 64]` slab's one-stick
/// sub-rows. COINCIDENT with [`SAMPLE_BLOCK`] (both 64) but a DIFFERENT fact:
/// this one is the fp16 elems-per-stick of the chained row's value operand on
/// the spyre device, that one is the pick's idx-stick lane count. Stated
/// separately so neither reader silently borrows the other's law.
pub const CHAIN_SLAB_STICK_ELEMS: u32 = 64;

/// The vendor dxp scheduler's per-core LX budget for one op —
/// `floor((2 MiB − 64 KiB) × 0.2)`: the fraction `reserveFrontendLx` leaves
/// for the scheduler to allocate ops into (the fence itself is the 0.8 carve
/// from address 0; ProgramLayout.cpp:81-100). SPELLED HERE because this crate
/// cannot see ktir-superdsc (the leaf-crate law: a third-party KTIR producer
/// consumes the lowering without pulling scratchy in) while the mint and both
/// reference arms — unconditional code in THIS crate — need the law. The same
/// number lives beside its receipts as `VENDOR_LX_SCHED_BYTES` in
/// ktir-superdsc's `superdsc_opspec.rs`; scratchy-target-spyre cross-pins the
/// pair so a drift in either crate is a cargo-build failure.
pub const CHAIN_SLAB_LX_BUDGET: u64 = 406_323;

/// ⭐⭐⭐⭐⭐ THE CHAINED HEAD'S SLAB-GATHER ENTRY COUNT (#603) — how many
/// entries leg 1 of the embed-row-gather door (ktir-superdsc's
/// `embed_row_gather`, cited not linked — the leaf crate cannot see back
/// here) splits the `[hidden, CHAIN_SLAB_STICK_ELEMS]` slab into. ONE
/// SPELLING, called at three sites — the mint (which states it into the
/// node), the EmbedRowGather reference arm (which re-joins `s = word0/k`),
/// and the spyre emitter — and one spelling called on the same `hidden`
/// always answers the same k.
///
/// THE LAW: the vendor scheduler charges a gathered op `4 × one entry's
/// bytes` per core (allocAllMem: 2 buffers × value + 2 × output,
/// L3DlOpsScheduler.cpp), and one entry is `hidden/K` positions × one fp16
/// stick ([`CHAIN_SLAB_STICK_ELEMS`] elems × 2 B), so the per-core charge is
/// `512 · (hidden/K)` B. K is the SMALLEST divisor of `hidden` with that
/// charge ≤ [`CHAIN_SLAB_LX_BUDGET`] and `K ≤` [`CHAIN_IDX_BLOCK`] — the u32
/// index stick's width, which is BOTH the lane pattern's home stick and the
/// IBR entry ceiling (`CopyDims::ENTRIES_PER_OP`, the same 32 one layer up in
/// ktir-superdsc; cross-pinned from scratchy-target-spyre). Smallest K ⇒
/// largest page ⇒ the least core spread that fits. `K = 1` whenever
/// `512 · hidden` itself fits (every hidden ≤ 793): the whole slab is ONE
/// entry — today's geometry, and the fill's lane pattern degenerates to
/// today's splat byte-for-byte (`i mod 1 = 0`). granite-3.1-2b (hidden 2048):
/// K=1 charges 1,048,576 B and K=2 524,288 B — both over — K=4 charges
/// 262,144 B ⇒ **K=4, page 512**: the pod-measured refusal this law answers
/// (`Expect valid chunk size that fits in LX`, the class-0 body bundle's slab
/// gather pinned at one plane-sized entry, 2026-10-25).
///
/// ⛔ A MISS IS LOUD, NEVER A CLAMP: an `hidden` whose divisor window is
/// empty — no K ≤ [`CHAIN_IDX_BLOCK`] with `512·(hidden/K) ≤ CHAIN_SLAB_LX_BUDGET`
/// — panics here, naming both ceilings and the divisor window; the offending
/// `hidden`'s value is named by the CALLER'S context (the mint's own assert
/// names the graph's hidden), because a const panic cannot format (E0015).
/// The door's raw entries refusal stays as the backstop for a hand-minted
/// program that bypasses this law.
///
/// ⛔ NO 64-MULTIPLE PAGE LAW: the door's `PageExtent::of_positions` takes
/// any count, the row-gather leg already declares page 32 card-proven, and an
/// entry is `page` WHOLE sticks for every `page` — the only constraints are
/// the two ceilings above.
pub const fn chain_slab_entries(hidden: u64) -> u64 {
    assert!(
        hidden > 0,
        "chain_slab_entries: hidden is 0 — no slab to split"
    );
    // The per-core LX charge per entry POSITION: 4 buffers (allocAllMem:
    // 2 × value + 2 × output) × one fp16 stick's bytes.
    const CHARGE_PER_POSITION: u64 = 4 * 2 * CHAIN_SLAB_STICK_ELEMS as u64;
    let mut k = 1u64;
    while k <= CHAIN_IDX_BLOCK as u64 {
        if hidden.is_multiple_of(k) && CHARGE_PER_POSITION * (hidden / k) <= CHAIN_SLAB_LX_BUDGET {
            return k;
        }
        k += 1;
    }
    panic!(
        "chain_slab_entries: this hidden has no divisor K <= 32 (CHAIN_IDX_BLOCK, the \
         index stick's width) with 512*(hidden/K) <= 406,323 B (the vendor LX \
         scheduler budget) — the per-core LX charge of the chained head's slab \
         gather cannot fit at any admissible entry count; a build-time refusal, \
         never a clamp"
    );
}

// THE LAW'S PINS — compile-time first (a regression fails this crate's build
// before any test runs; the types-not-comments law), the named tests below
// riding the `chain` filter the cell already runs.
const _: () = assert!(
    chain_slab_entries(2048) == 4,
    "granite-3.1-2b's hidden 2048: K=4, page 512 — the pod-measured LX refusal's answer"
);
const _: () = assert!(
    chain_slab_entries(2) == 1,
    "hidden 2 — the mint test's own toy — is one entry"
);
const _: () = assert!(
    chain_slab_entries(128) == 1,
    "the door tests' toy hidden 128 is one entry"
);
const _: () = assert!(
    chain_slab_entries(793) == 1 && chain_slab_entries(794) == 2,
    "the budget's boundary: 512*793 = 406,016 fits, 512*794 = 406,528 does not"
);

/// ⭐ THE S-STICK SEED'S LANE PATTERN — the one spelling of the chain's s-seed
/// law, shared by BOTH host halves that push it (the card path's `ChainSeed`
/// steps in scratchy-target-spyre's forward_tape and the worker's dynamic
/// s-source push in scratchy-serving-worker): lane `i` of the stick carries
/// `k·s + (i mod k)` — ENTRY `k·s + (i mod k)` of the slab gather's `k`-entry
/// tile, the flat-address law's own addressing (the extra lanes repeat the
/// first `k`, so whatever word discipline the IBR takes stays inside the
/// picked slab). The device's own ChainStage writes the same pattern for the
/// same pick, which is what keeps a host seed idempotent with the device's
/// writes at every `k`.
///
/// At `k = 1` the pattern IS the splat (`i mod 1 = 0`): the degenerate's seed
/// is today's one-word splat byte-for-byte — the k = 1 seed test pins that
/// leg, the k > 1 patterned-seed test the other, both through this one
/// spelling.
///
/// `k ≥ 1` is the mint's stated law (asserted at the mint, guarded at the
/// `chain_staged_tables` fill in scratchy-target-spyre); a `k` of `0` panics
/// the remainder here — loudly, never silently.
pub fn chain_s_stick_pattern(k: u32, s: u32, lanes: usize) -> Vec<u32> {
    (0..lanes).map(|i| k * s + (i as u32 % k)).collect()
}

/// ⭐⭐⭐⭐⭐ THE L INDEX'S OWN STICK COUNT — how many `[CHAIN_IDX_BLOCK]`-lane
/// sticks the chained head's lane index carries. The row gather runs
/// `entries = hidden/STICK_ELEMS` one-stick entries, and the IBR is loaded
/// ONE [`CHAIN_IDX_BLOCK`]-word stick at a time with each core's read offset
/// taken modulo that stick — so the pass is ONE OP PER INDEX STICK, the KV
/// fold's own cut (`GatherScratch::copies`, the rung-8 remedy), and the l
/// index must carry every entry: `ceil(entries / CHAIN_IDX_BLOCK)` sticks.
///
/// ⭐ A GEOMETRY COUNT, IN `div_ceil` AT BOTH LEVELS — a partial stick still
/// occupies a stick of placement (the toy graphs the mint's surgery tests
/// ride at sub-stick hiddens), so the count is `ceil(hidden/STICK_ELEMS)`
/// entries over `ceil(·/CHAIN_IDX_BLOCK)` sticks. The ALIGNMENT law — hidden
/// a whole number of fp16 sticks — is the RAMP'S and the SPYRE DOOR'S own
/// ([`chain_l_stick_ramp`] refuses a sub-stick hidden; the door's transpose
/// leg requires both extents stick-aligned), never this count's: at every
/// width those laws admit (every real model's hidden), the two spellings of
/// the entry count agree.
///
/// This is the count the MINT states as the l stick's row width and the l
/// table's row width in sticks, the STAGED FILL derives for the l-tab's
/// shape, both HOST SEEDS inherit through the self-sized ramp, and the DOORS
/// read off the operands they are handed — one spelling, cross-pinned by the
/// ramp's length. At every hidden ≤ 2048 it is 1: the whole law degenerates
/// to today's single-stick index byte-for-byte.
pub const fn chain_l_sticks(hidden: u32) -> u32 {
    assert!(
        hidden > 0,
        "chain_l_sticks: hidden is zero — no row to name"
    );
    hidden
        .div_ceil(CHAIN_SLAB_STICK_ELEMS)
        .div_ceil(CHAIN_IDX_BLOCK)
}

// THE LAW'S PINS — compile-time first (a regression fails this crate's build
// before any test runs; the types-not-comments law), the named tests below
// riding the `chain` filter the cell already runs.
const _: () = assert!(
    chain_l_sticks(128) == 1 && chain_l_sticks(2048) == 1,
    "every admitted hidden ≤ 2048 is ONE l stick — the degenerate the cut \
     must keep byte-identical"
);
const _: () = assert!(
    chain_l_sticks(4096) == 2,
    "granite-3.1-8b's hidden 4096: 64 entries, TWO l index sticks — the \
     per-stick cut the 8b build wall's fix rides"
);
const _: () = assert!(
    chain_l_sticks(4160) == 3,
    "a partial last stick is lawful: hidden 4160 is 65 entries over 3 sticks, \
     the last op running one entry"
);

/// ⭐ THE L-STICK SEED'S LANE RAMP — the one spelling of the chain's l-seed
/// law, shared by BOTH host halves that push it (the card path's `ChainSeed`
/// steps in scratchy-target-spyre's forward_tape and the worker's dynamic
/// l-source push in scratchy-serving-worker) AND the l-tab's fill: word `j`
/// of the index carries `STICK_ELEMS·(j mod entries) + l` — the SOURCE STICK
/// NUMBER of the transposed slab's row `l`, entry `j mod entries` of the row
/// gather's `entries`-entry tile. The restickify transpose writes the
/// transposed row's elements at sticks `{STICK_ELEMS·j + l}` — scattered at a
/// 64-stick stride, the door's own slab-major output law — so the row
/// gather's entry `j` must read exactly stick `STICK_ELEMS·j + l`; a flat
/// `j` (the old splat's read) lands on a patch of raw weight bytes instead,
/// the defect this law closes. Word 0 is `l` itself, so the reference's
/// re-join (`l = word0`) is unchanged. The modulus is the gather's ENTRY
/// COUNT `entries = hidden / STICK_ELEMS` — the s-pattern's own idiom: the
/// extra lanes past the entry count repeat the row's first entries, so
/// whatever word discipline the IBR takes stays inside the picked row.
///
/// ⭐ SELF-SIZED at [`chain_l_sticks`]'s own count — the ramp fills the WHOLE
/// l index (every stick the per-stick cut walks), so its length IS the
/// placement's word count and the two cannot drift. At `entries` past one
/// stick (hidden > 2048) the words past the first stick continue the same
/// law — the wrap engages only on the final stick's pad lanes, never between
/// the ops of the cut.
///
/// ⛔ A MISS IS LOUD, NEVER A CLAMP: `hidden` must be a whole number of
/// [`CHAIN_SLAB_STICK_ELEMS`]-elem sticks — a partial stick has no stick
/// indices to name, and the ramp refuses it here rather than floor-divide
/// its way to a silent wrong row.
pub fn chain_l_stick_ramp(l: u32, hidden: u32) -> Vec<u32> {
    assert!(
        hidden > 0 && hidden.is_multiple_of(CHAIN_SLAB_STICK_ELEMS),
        "chain_l_stick_ramp: hidden {hidden} is not a whole number of \
         {}-elem sticks — the transposed row's stick indices are undefined",
        CHAIN_SLAB_STICK_ELEMS
    );
    let entries = hidden / CHAIN_SLAB_STICK_ELEMS;
    let words = chain_l_sticks(hidden) as usize * CHAIN_IDX_BLOCK as usize;
    (0..words)
        .map(|j| CHAIN_SLAB_STICK_ELEMS * (j as u32 % entries) + l)
        .collect()
}

/// ⭐ THE L-STICK'S SENTINEL ROW — the zero-tail ramp the l-tab's sentinel
/// rows (every `t ≥ PAD_V`) carry and NOTHING else: word `j` names tail
/// stick `hidden + (j mod entries)` — the transposed placement's UNWRITTEN
/// sticks past the real transpose. The real transpose tops out at stick
/// `hidden − 1` (element `(out, y)` at stick `64·⌊y/64⌋ + out` with
/// `out < 64`, `y < hidden`), and the `[STICK_ELEMS + 1, hidden]` placement
/// reserves `hidden + entries` sticks — so the tail `[hidden, hidden +
/// entries)` is exactly the sentinel's lanes, all zeros, the same bytes the
/// old flat read reached. The `mod entries` wrap is the whole safety law at
/// multi-stick widths: the l index carries `l_sticks·32` words but only
/// `entries` sticks of tail exist, so the words past `entries` RE-BIND to
/// the tail's own sticks — every read any op of the cut takes lands in the
/// natural zero tail, in bounds at every width, never in the next tensor's
/// bytes (the #717 class the shared-stick alternative could not avoid).
/// Word 0 is `hidden`, the sentinel MARK the reference's re-join refuses on
/// (a real ramp's word 0 is a lane, and `l ≤ 63 < 64 ≤ hidden` at every
/// lawful hidden, so the mark is unambiguous). The seeds never push this
/// stick — the host seeds real picks; the sentinel arises only when the
/// device's own pick answers past the vocab and the stage gathers the
/// table's sentinel row.
///
/// ⭐ SELF-SIZED at [`chain_l_sticks`]'s own count, same as the real ramp —
/// the two fills share one word count and cannot drift.
pub fn chain_l_sentinel_ramp(hidden: u32) -> Vec<u32> {
    assert!(
        hidden > 0 && hidden.is_multiple_of(CHAIN_SLAB_STICK_ELEMS),
        "chain_l_sentinel_ramp: hidden {hidden} is not a whole number of \
         {}-elem sticks — the zero tail has no stick indices to name",
        CHAIN_SLAB_STICK_ELEMS
    );
    let entries = hidden / CHAIN_SLAB_STICK_ELEMS;
    let words = chain_l_sticks(hidden) as usize * CHAIN_IDX_BLOCK as usize;
    (0..words).map(|j| hidden + j as u32 % entries).collect()
}

// ── The seeds at the launch's own width (the ONE per-row spelling) ──

/// ⭐⭐⭐ THE S SEED AT THE LAUNCH'S OWN WIDTH — `[rows, CHAIN_IDX_BLOCK]` u32 words,
/// row `r` carrying ITS OWN token's pattern (`chain_s_stick_pattern` of
/// `toks[r.min(n-1)] >> 6`): the seed a batched decode's every row gathers with,
/// and the same bytes the device's own [`ChainStage`]-table gathers write for the
/// same picks — which is what keeps a host seed idempotent with the device's
/// writes at every width.
///
/// ⛔ THE CLAMP IS THE PAD-ROW LAW, not a convenience: the seed's rows are the
/// LAUNCH's rows (`rows`, the baked rung width) while `toks` names the REAL rows
/// (`n ≤ rows`), and a pad row repeats the last real token's decompose — the same
/// law the embedding staging and `ChunkRows::row_logical_pos` pin, so the sticks
/// and the rows they feed cannot disagree at a pad row.
///
/// At `rows == 1` this is the solo spelling byte-for-byte (one stick, the one
/// token's pattern) — the degenerate every W=1 rung rode before the batched rung
/// existed, kept exact by construction rather than by a width branch.
pub fn chain_s_seed(k: u32, toks: &[u32], rows: usize) -> Vec<u32> {
    assert!(
        !toks.is_empty(),
        "chain_s_seed: no token to seed from — a chain with no token names no slab"
    );
    let lanes = CHAIN_IDX_BLOCK as usize;
    let last = toks.len() - 1;
    (0..rows)
        .flat_map(|r| chain_s_stick_pattern(k, toks[r.min(last)] >> 6, lanes))
        .collect()
}

/// ⭐⭐⭐ THE L SEED AT THE LAUNCH'S OWN WIDTH — `[rows·l_sticks, CHAIN_IDX_BLOCK]`
/// u32 words, row `r`'s `l_sticks` ramp sticks (`chain_l_stick_ramp` of
/// `toks[r.min(n-1)] & 63`) then row `r+1`'s — the placement shape the mint
/// states and the row gather's per-stick cut walks.
///
/// The clamp is the pad-row law [`chain_s_seed`] states; at `rows == 1` this is
/// the solo spelling byte-for-byte.
pub fn chain_l_seed(toks: &[u32], hidden: u32, rows: usize) -> Vec<u32> {
    assert!(
        !toks.is_empty(),
        "chain_l_seed: no token to seed from — a chain with no token names no lane"
    );
    let last = toks.len() - 1;
    (0..rows)
        .flat_map(|r| chain_l_stick_ramp(toks[r.min(last)] & 63, hidden))
        .collect()
}

// ── The tied embed weight's pad widening (the chained head's extent law) ──

/// What [`widen_tied_embed_weight`] stated, for the caller's bake log.
#[derive(Clone, Copy, Debug)]
pub struct TiedEmbedWiden {
    /// The widened weight's SOURCE id — the lm-head matmul's `inputs[1]`,
    /// the tied embedding matrix.
    pub weight: TensorId,
    /// The weight's REAL column count — the vocab the walk read off the
    /// tensor before widening. The number the bake log reports; never the
    /// widened cols.
    pub vocab: u32,
    /// The widened column count — `sample_pad_vocab(vocab)`, the one width
    /// the whole chain states.
    pub pad_vocab: u32,
}

/// Widen the tied embed weight's STATED columns to the pick's pad — the
/// extent law the chained head's gather addresses. ⭐ ONE WIDTH: the
/// decompose tables, the sentinel's reserved row, the `s`-clamp's boundary,
/// and (after this pass) the weight's stated columns are ALL `pad_vocab` —
/// the clamp boundary IS the weight boundary by construction, so every
/// answerable pick's slab gather is in extent because the unsafe window is
/// UNWRITABLE, not because a derivation spanning three crates says it is
/// unreachable. The door's stick law
/// (`embed_row_gather`: vocab a whole 64-element fp16 multiple) checks the
/// same fact at expansion — `pad_vocab` is a power of two ≥ 64, a whole
/// stick multiple by construction, where the raw vocab (granite 49155 =
/// 768·64 + 3) is not and the door refuses it as a partial-slab address.
///
/// ⭐ THE PAD DERIVES FROM THE WEIGHT'S OWN STATED COLS — the one fact every
/// rung's graph carries — never from a pick node (only a device-sampling
/// rung has one). That is what makes the pass UNIFORM BY CONSTRUCTION: it
/// runs on every rung's graph of a tied model, and every bundle's placement
/// reserves the same widened bytes (the weight-segment parity ledger refuses
/// a model whose bundles disagree — a non-uniform widening is a build error,
/// never a card failure).
///
/// ⭐ ONLY THE STATED SHAPE MOVES. The matmul's own weight REGION stays
/// `0..vocab` — a partial view over the widened tensor, the exact situation
/// an output-side widening created for the matmul's output — so the
/// matmul's compute, its kernel manifest, and the worker's staging (all
/// region-driven) are untouched: the reservation grows, the FLOPs do not.
/// (On the spyre target that staging fact is stated in one place: the
/// lowering's `gemm_weight_host_pairs`, the union of each gemm weight's
/// matmul consumers' view regions, which the wiring's manifest shapes and
/// the three-width-laws check both read as the host pair.) The pad columns
/// past the worker's staged device width keep the segment's prepare-time
/// zeros (weights stage once at load and are never reused), which makes
/// `idx ≥ vocab` gather a ZERO row in value as well as in extent.
///
/// ⭐ NON-ADMITTING TAILS RETURN `None` AND TOUCH NOTHING: a rung whose
/// result tail is not the admitting shape (an arity-2 Dense lm-head under
/// single-input `ScalarMul`/`TanhSoftCap` epilogues, its `inputs[1]` a
/// source) keeps the host-gather path — a tied model whose NO rung admits
/// is byte-identical under this pass. The caller gates on the model's own
/// tie fact; the pass itself states only the tail's structure.
///
/// ⛔ NOT IDEMPOTENT, BY DESIGN: `sample_pad_vocab` always exceeds its input
/// (a power of two, doubled when the vocab already is one), so a second call
/// on a widened graph would widen again. The pass runs ONCE per graph, and
/// [`mint_embed_chain`]'s assert — the widened cols against the pad stated
/// off the SAME vocab — is the typed proof it ran exactly right.
pub fn widen_tied_embed_weight<F: RopeForm>(graph: &mut SubtileIR<F>) -> Option<TiedEmbedWiden> {
    // The mints' own tail walk, None instead of panic at every non-admitting
    // turn: this pass runs on every rung's graph, where the mints' admission
    // gate has not run.
    let mut t = graph.result;
    let weight = loop {
        let node = graph.nodes.iter().find(|n| n.output.tensor == t)?;
        match &node.op {
            SubOp::ScalarMul { .. } | SubOp::TanhSoftCap { .. } if node.inputs.len() == 1 => {
                t = node.inputs[0].tensor;
            }
            SubOp::MatmulTile { .. } => break node.inputs[1].tensor,
            _ => return None,
        }
    };
    if !graph.is_source(weight) {
        return None;
    }
    let vocab = graph.tensors[weight.index()].cols;
    let pad_vocab = sample_pad_vocab(vocab);
    graph.tensors[weight.index()].cols = pad_vocab;
    Some(TiedEmbedWiden {
        weight,
        vocab,
        pad_vocab,
    })
}

// ── The chained head's mint (the pipelined decode's graph cut) ──────

/// What [`mint_embed_chain`] inserted, for the caller's bake log and the
/// wiring's seed fact.
#[derive(Clone, Copy, Debug)]
pub struct EmbedChainMint {
    /// The gathered row's width — the op's own `hidden`.
    pub hidden: u32,
    /// The matrix's REAL column count — the vocab the caller stated, the
    /// number the bake log reports. The weight's OWN stated width is the
    /// PAD ([`widen_tied_embed_weight`]'s law), never this number.
    pub vocab: u32,
    /// The rung's row count — the batch width this graph serves; one chain
    /// stick row and one gathered row per request row.
    pub width: u32,
    /// The persistent s-stick's SOURCE id — the placement holding the pick's
    /// SLAB index (`t >> 6`, clamped to 0 past `pad_vocab`): the host's seed
    /// lands here (the first play's seed is the prompt's final token, a fact
    /// the host owns; the per-step path then re-states the seed each step,
    /// idempotent with the device's own writes, and the block path stages it
    /// once per block) and every step's `ChainStage` write keeps it current.
    /// One row per request: the `[width, CHAIN_IDX_BLOCK]` placement, row
    /// `r` request `r`'s pattern.
    pub s_stick: TensorId,
    /// The persistent l-stick's SOURCE id — the placement holding the pick's
    /// LANE RAMP ([`chain_l_stick_ramp`] at `t & 63`; the sentinel ramp, the
    /// transposed placement's zero tail, past the pad): the same
    /// seed-then-device-owns law as `s_stick`. One row per request, each row
    /// `chain_l_sticks(hidden)` sticks wide (`CHAIN_IDX_BLOCK` words per
    /// stick, the per-stick cut's own addressing — stick `r·l_sticks + b`
    /// of the placement).
    pub l_stick: TensorId,
}

/// Append the CHAINED HEAD to a decode graph: the persistent s/l stick
/// sources and the decompose's two staged tables, an `EmbedRowGather` at the
/// entry (every whole-tensor consumer of `embed_row` re-pointed onto its
/// output), and a `ChainStage` at the tail (the pick decomposed into the
/// persistent placements through the tables, one row per request). Pure
/// graph surgery — the GATE (which rungs mint, and the TIE the matrix
/// operand assumes) is the caller's fact; this fn states only its
/// own invariants and panics on their violation, which at expansion time is
/// a build error, never a runtime one.
///
/// ⭐ THE PICK IS A NAMED OPERAND, AND THE CALLER NAMES IT — VOCAB INCLUDED:
/// `pick_stick` is the device pick producer's output — the sampler op the
/// emission finds and passes here — and `vocab` is that producer's own
/// column count, the REAL vocab the tables' range decomposes over. This
/// mint asserts the STRUCTURE (an op output, one row per request, the tail
/// walk to the lm-head admits, the weight pre-widened to this vocab's pad);
/// it cannot and does not check the producer's KIND — the caller that names
/// a pick owns that it is one, and a caller with no producer to name does
/// not call this fn: it REFUSES the bake (a host-staged pick is not a pick,
/// and there is no fallback arm to hand it to).
///
/// ⭐ THE CHAIN IS CROSS-STEP, THE GRAPH IS PER-STEP: the gather cannot
/// read this step's own pick (a cycle, and the wrong step's pick besides)
/// — it reads the PERSISTENT source the PREVIOUS step's `ChainStage`
/// wrote. The `RopeAppend`-writes/`AttnDecode`-reads law (the sealed
/// `KvCacheLayout` pair) is the precedent: a placement the device owns
/// across forwards. Step j-1's stick write → step j's gather read is
/// therefore the SAME ordering class as step j-1's KV append → step j's
/// `AttnDecode` — cross-submission ordering the ring already provides
/// (per-CB fences; FIFO = program order). One law, two consumers.
///
/// ⭐ THE TIE IS THE CALLER'S FACT: the gather's matrix operand is the
/// lm-head's own weight source, which is the embedding matrix IFF the
/// model ties embeddings. This mint asserts the structure (the weight is
/// a source of shape `[hidden, vocab]`, the embed rows a `[width, hidden]`
/// source); the tie itself is checked where the model config lives, and
/// an untied model does not mint (a named follow-up admission, not a
/// silent scope).
///
/// ⭐ THE FOUR-SOURCE INSERT renumbers every op-output tid up by four
/// (shift = 4); source ids below the old `num_sources` — the caller's
/// `embed_row` and the lm-head's weight among them — are UNTOUCHED, which
/// keeps the caller's `weight_ids` and the wiring's binding scan valid
/// against the minted graph. The sticks are SOURCES, not node outputs,
/// because they are CROSS-STEP CARRIERS: a node output with no in-graph
/// consumer is a free slot the coloring reuses, and the body's tensors
/// would clobber the chain's state between steps (the KV pool's own law —
/// persistent state lives as a placement, never as a graph edge). The
/// tables are sources because they are ORDINARY TENSOR OPERANDS: the mint
/// states them, the node consumes them, and the worker stages their bytes
/// once at load through the SAME load-const list every staged table rides
/// (a second minted node of the kind is a build error, never a
/// last-write-wins).
///
/// ⭐ THE GATHER NODE IS INSERTED AT NODE POSITION 0, not appended: the
/// graph's nodes are topologically ordered and the gather's consumers are
/// the body's entry ops, early in the list. The insert shifts every node
/// id up by one; every `KvCacheProducer::SameForwardRopeAppend` producer
/// index is remapped through the shift (the rewrites' own
/// `new_id_for_old` discipline).
pub fn mint_embed_chain<F: RopeForm>(
    graph: &mut SubtileIR<F>,
    embed_row: TensorId,
    pick_stick: TensorId,
    vocab: u32,
) -> EmbedChainMint {
    // 1) The pick: an OP OUTPUT (device-produced), one row per request. The
    //    caller named it and its vocab; these asserts catch a caller that
    //    named something else.
    assert!(
        !graph.is_source(pick_stick),
        "mint_embed_chain: the pick stick must be an OP OUTPUT — a device-produced pick; \
         a host-staged pick is not a pick"
    );
    let pick_node = graph
        .nodes
        .iter()
        .position(|n| n.output.tensor == pick_stick)
        .expect("mint_embed_chain: the pick stick's producer — the caller names a pick it owns");
    let pad_vocab = sample_pad_vocab(vocab);

    // 2) The lm-head matmul: the tail walk back from the pick's logits row —
    //    the same walk the pick's own admission performs (single-input
    //    ScalarMul/TanhSoftCap epilogues back to the MatmulTile; anything
    //    else is a graph the admission already refused, so the mint was
    //    called outside it). The loop breaks with the WEIGHT TID (a Copy), so
    //    no borrow of the node list outlives the walk the surgery below
    //    mutates.
    let mut t = graph.nodes[pick_node].inputs[0].tensor;
    let weight = loop {
        let node = graph
            .nodes
            .iter()
            .find(|n| n.output.tensor == t)
            .expect("the result tail's writer");
        match &node.op {
            SubOp::ScalarMul { .. } | SubOp::TanhSoftCap { .. } if node.inputs.len() == 1 => {
                t = node.inputs[0].tensor;
            }
            SubOp::MatmulTile { .. } => break node.inputs[1].tensor,
            other => panic!(
                "mint_embed_chain walked a {other:?} result tail — the admission gate \
                 (an arity-2 Dense lm-head under single-input ScalarMul/TanhSoftCap) \
                 refuses this graph, so the mint was called outside it"
            ),
        }
    };
    assert!(
        graph.is_source(weight),
        "mint_embed_chain: the lm-head's weight operand must be a source (the tied \
         embedding matrix the gather reads)"
    );
    let (w_rows, w_cols) = {
        let s = graph.shape(weight);
        (s.rows, s.cols)
    };
    // ⭐ THE WIDENED-LAW ASSERT — the typed proof the per-graph widening pass
    // ([`widen_tied_embed_weight`]) ran on THIS rung, once and exactly right:
    // the weight's stated cols are the pad this mint derived off the caller's
    // SAME vocab. The mint does NOT widen — the pass is per-graph and uniform
    // (the weight-segment parity ledger's law), and this rung is only one of
    // them. A raw-vocab weight here is a partial slab (the gather door's
    // stick law would refuse it at expansion as an address); a wider one is
    // a pass that ran twice; both fail HERE, at the mint, naming the pass.
    assert_eq!(
        w_cols,
        pad_vocab,
        "mint_embed_chain: the lm-head weight t{} states {} columns, not the pick's pad \
         {pad_vocab} — the per-graph widening pass (widen_tied_embed_weight) did not run \
         before this mint; the chain's tables, sentinel, and s-clamp are all keyed to the \
         pad, and the gather addresses the weight at that width or not at all",
        weight.index(),
        w_cols
    );
    assert!(
        graph.is_source(embed_row),
        "mint_embed_chain: the embed rows must be a source (the host-bound rows being \
         replaced by the chain)"
    );
    let embed_shape = graph.shape(embed_row);
    let width = embed_shape.rows;
    assert_eq!(
        embed_shape.cols, w_rows,
        "mint_embed_chain: the embed rows must be [width, hidden] with hidden = the \
         lm-head weight's row count"
    );
    assert_eq!(
        graph.shape(pick_stick).rows,
        width,
        "mint_embed_chain: the pick stick must carry one row per request row (the \
         batch width this graph serves)"
    );
    let hidden = w_rows;

    // 3) THE FOUR-SOURCE INSERT — the persistent s/l sticks AND the
    //    decompose's two staged tables at `[num_sources, +4)`; every
    //    op-output tid bumps up by four.
    let n_src = graph.num_sources;
    let bump = |tr: &mut TensorRegion| {
        if tr.tensor.0 >= n_src {
            tr.tensor.0 += 4;
        }
    };
    for node in &mut graph.nodes {
        for inp in &mut node.inputs {
            bump(inp);
        }
        bump(&mut node.output);
    }
    for ot in &mut graph.op_output {
        if ot.0 >= n_src {
            ot.0 += 4;
        }
    }
    if graph.result.0 >= n_src {
        graph.result.0 += 4;
    }
    // The tables' row count: every answerable pick `t` up to the sentinel
    // `pad_vocab + SAMPLE_SENTINEL_OFFSET` INCLUSIVE — the pick contract's
    // whole range, one row per pick (the sentinel is the LAST row). THE
    // GROUP LAW LIVES HERE, beside the one statement of the extent: the
    // sentinel pick is `G·4096 + SAMPLE_SENTINEL_OFFSET` with
    // `G = sample_groups(pad_vocab)`, which equals
    // `pad_vocab + SAMPLE_SENTINEL_OFFSET` — the last table row — only
    // when `pad_vocab` is a whole multiple of the pick group
    // (`SAMPLE_BLOCK · SAMPLE_GROUP_BLOCKS`). A pad that is not a group
    // multiple drops the sentinel pick SHORT of the last row — `G` is the
    // rounded-DOWN group count, so `G·4096 ≤ pad_vocab` and the pick lands
    // inside the past-pad marks, never past the staged bytes; what breaks
    // is the ONE-TO-ONE law this extent states (the sentinel is the LAST
    // row, the answerable range's own upper end) — so the mint refuses it
    // here, at the single site that computes the extent, rather than
    // letting a downstream door re-derive the arithmetic.
    let group = SAMPLE_BLOCK * SAMPLE_GROUP_BLOCKS;
    assert!(
        pad_vocab != 0 && pad_vocab.is_multiple_of(group),
        "mint_embed_chain: the padded vocab {pad_vocab} is not a whole multiple of the \
         pick group {group} — the sentinel pick would land short of the last table row \
         and the sentinel-is-the-last-row law would be false; the chain mints only \
         group-multiple pads"
    );
    let table_rows = pad_vocab + SAMPLE_SENTINEL_OFFSET + 1;
    // ⭐ THE L INDEX'S OWN STICK COUNT — chain_l_sticks' one STATEMENT call:
    // the mint derives it and states the two l shapes from it, so the fill
    // (wiring's staged tables), the door (both gather doors read the
    // operands they are handed), and both host seeds (the self-sized ramp)
    // all read one stated fact — never a second derivation from `hidden`.
    // The count is `ceil((hidden/STICK_ELEMS)/CHAIN_IDX_BLOCK)`: 1 at every
    // hidden ≤ 2048 (the degenerate the cut must keep byte-identical), 2 at
    // granite-3.1-8b's 4096 (64 one-stick entries — the row gather runs one
    // op per index stick, the KV fold's own cut).
    let l_sticks = chain_l_sticks(hidden);
    graph.tensors.splice(
        n_src as usize..n_src as usize,
        [
            TensorShape {
                rows: width,
                cols: CHAIN_IDX_BLOCK,
            },
            // ⭐ THE L STICK IS ONE STICK WIDE — [width·l_sticks, 32], row r's
            // l_sticks sticks then row r+1's (op (r, b) of the door's per-stick
            // cut reads its index at stick r·l_sticks + b). A [width,
            // 32·l_sticks] spelling would be stick-scattered on the device
            // (the stick-scatter law: element (r, c) lands at stick
            // (c/64)·width + r), so row r's stick b would sit at flat stick
            // b·width + r — a different buffer than every reader states. At
            // one stick wide, row-major and stick-major coincide, so the
            // address cannot be ambiguous. At l_sticks = 1 (every hidden
            // ≤ 2048) both forms are identical.
            TensorShape {
                rows: width * l_sticks,
                cols: CHAIN_IDX_BLOCK,
            },
            TensorShape {
                rows: table_rows,
                cols: CHAIN_IDX_BLOCK,
            },
            TensorShape {
                rows: table_rows,
                cols: CHAIN_IDX_BLOCK * l_sticks,
            },
        ],
    );
    let s_stick = TensorId(n_src);
    let l_stick = TensorId(n_src + 1);
    let s_tab = TensorId(n_src + 2);
    let l_tab = TensorId(n_src + 3);
    graph.num_sources += 4;
    // The pick stick is an op output (asserted above), so the bump moved it.
    let pick_stick = TensorId(pick_stick.0 + 4);

    // 4) THE GATHER — a fresh `[width, hidden]` output at the tensor table's
    //    end; the node is INSERTED at position 0 (topologically safe: it
    //    reads sources only), every node id and producer index shifts up.
    let whole = |g: &SubtileIR<F>, t: TensorId| TensorRegion {
        tensor: t,
        region: Region {
            rows: Range::new(0, g.tensors[t.index()].rows),
            cols: Range::new(0, g.tensors[t.index()].cols),
        },
    };
    let gather_out = TensorId(graph.tensors.len() as u32);
    graph.tensors.push(TensorShape {
        rows: width,
        cols: hidden,
    });
    graph.nodes.insert(
        0,
        SubtileNode {
            id: SubtileId(0),
            op: SubOp::EmbedRowGather {
                hidden,
                replaces: embed_row,
            },
            inputs: vec![
                whole(graph, s_stick),
                whole(graph, l_stick),
                whole(graph, weight),
            ],
            output: whole(graph, gather_out),
        },
    );
    for (i, node) in graph.nodes.iter_mut().enumerate().skip(1) {
        node.id = SubtileId(i as u32);
        if let SubOp::AttnDecode {
            producer:
                KvCacheProducer::SameForwardRopeAppend {
                    producer_node_idx, ..
                },
            ..
        } = &mut node.op
        {
            *producer_node_idx += 1;
        }
    }

    // 5) THE RE-POINT — every WHOLE-TENSOR consumer of the embed rows reads
    //    the gather's output instead. The rows' source survives, placed and
    //    readable by tid; the host simply stops seeding it.
    for node in graph.nodes.iter_mut().skip(1) {
        for inp in &mut node.inputs {
            if inp.tensor == embed_row {
                assert_eq!(
                    (
                        inp.region.rows.start,
                        inp.region.rows.len,
                        inp.region.cols.start,
                        inp.region.cols.len
                    ),
                    (0, width, 0, hidden),
                    "mint_embed_chain: a consumer reads a SLICE of the embed rows — the \
                     chain replaces the WHOLE tensor only"
                );
                inp.tensor = gather_out;
            }
        }
    }

    // 6) THE STAGE — the pick decomposed into the persistent placements
    //    through the staged tables; the output is a fresh s-value stick
    //    (every op's output its own), one row per request. `pad_vocab` and
    //    `table_rows` are stated from the caller's own vocab — the
    //    decompose's one derivation and the tables' own extent — so producer
    //    and consumer can never disagree on the range the decompose is exact
    //    over, and no door re-derives the extent from `pad_vocab` a second
    //    time. The tables ride the node as ORDINARY OPERANDS (inputs 3 and
    //    4): the wiring's load-const list derives their `(tid, kind)` entries
    //    from THIS node's inputs, so the worker's staging and the bake's
    //    numeric gate read the same pair the door's gathers name.
    // ⭐ THE SLAB-GATHER ENTRY COUNT — chain_slab_entries' one STATEMENT call:
    // the mint computes k and carries it in the payload, so the fill, the
    // door, both host seeds, and both reference arms read one stated fact.
    // The law's miss panic cannot format the value (E0015); THIS is the
    // caller context it names — the hidden that could miss is THIS graph's
    // hidden, the lm-head weight's row count, and the assert below re-states
    // the contract with both numbers should it ever break.
    let k = chain_slab_entries(u64::from(hidden)) as u32;
    assert!(
        (1..=CHAIN_IDX_BLOCK).contains(&k) && hidden.is_multiple_of(k),
        "mint_embed_chain: chain_slab_entries answered k = {k} for hidden \
         {hidden} — the entry count must divide the hidden and sit inside the \
         index stick (CHAIN_IDX_BLOCK)"
    );
    let stage_out = TensorId(graph.tensors.len() as u32);
    graph.tensors.push(TensorShape {
        rows: width,
        cols: CHAIN_IDX_BLOCK,
    });
    graph.nodes.push(SubtileNode {
        id: SubtileId(graph.nodes.len() as u32),
        op: SubOp::ChainStage {
            pad_vocab,
            table_rows,
            k,
        },
        inputs: vec![
            whole(graph, pick_stick),
            whole(graph, s_stick),
            whole(graph, l_stick),
            whole(graph, s_tab),
            whole(graph, l_tab),
        ],
        output: whole(graph, stage_out),
    });

    EmbedChainMint {
        hidden,
        // The REAL vocab — the caller's own statement, the number the bake
        // log reports; the weight's stated cols are the PAD by the widening
        // pass's law, asserted above.
        vocab,
        width,
        s_stick,
        l_stick,
    }
}

// ── Host evaluation ────────────────────────────────────────────────

/// Gather a tensor region into a dense row-major `(buf, rows, cols)`.
pub(crate) fn gather<F: RopeForm>(
    tr: &TensorRegion,
    graph: &SubtileIR<F>,
    bufs: &[Vec<f32>],
) -> (Vec<f32>, u32, u32) {
    let shape = graph.shape(tr.tensor);
    let src = &bufs[tr.tensor.0 as usize];
    let (r, c) = (tr.region.rows.len, tr.region.cols.len);
    let mut out = Vec::with_capacity((r * c) as usize);
    for i in 0..r {
        let row = tr.region.rows.start + i;
        let base = ((row * shape.cols) + tr.region.cols.start) as usize;
        out.extend_from_slice(&src[base..base + c as usize]);
    }
    (out, r, c)
}

/// Scatter a dense `[rows, cols]` buffer into `bufs[tensor]` at `region`.
pub fn scatter(
    bufs: &mut [Vec<f32>],
    tensor: TensorId,
    region: Region,
    data: &[f32],
    shape: TensorShape,
) {
    let (r, c) = (region.rows.len as usize, region.cols.len as usize);
    // Host-evaluator helper; r * c == data.len() is structurally
    // entailed by callers (eval_node always builds out from the
    // region geometry it then scatters with). Defensive in debug only.
    debug_assert_eq!(data.len(), r * c, "scatter data size vs region");
    let dst = &mut bufs[tensor.0 as usize];
    for i in 0..r {
        let row = region.rows.start as usize + i;
        let base = row * shape.cols as usize + region.cols.start as usize;
        dst[base..base + c].copy_from_slice(&data[i * c..(i + 1) * c]);
    }
}

/// Evaluate the SubtileIR on the host. `sources[s]` is the row-major
/// buffer for source tensor `s` (`s < num_sources`), matching
/// `graph.tensors[s]`. Returns the backing buffer of every tensor
/// (indexed by [`TensorId`]); the logits are `bufs[graph.result]`.
pub fn eval_dag<F: RopeForm>(graph: &SubtileIR<F>, sources: &[&[f32]]) -> Vec<Vec<f32>> {
    // Host-evaluator source-count check; debug-only since the
    // codegen-side validate() catches structural mismatches.
    debug_assert_eq!(
        sources.len(),
        graph.num_sources as usize,
        "source count mismatch"
    );
    let mut bufs: Vec<Vec<f32>> = graph
        .tensors
        .iter()
        .map(|t| vec![0f32; (t.rows * t.cols) as usize])
        .collect();
    for (s, src) in sources.iter().enumerate() {
        debug_assert_eq!(src.len(), bufs[s].len(), "source {s} buffer size mismatch");
        bufs[s].copy_from_slice(src);
    }
    for node in &graph.nodes {
        let out = eval_node(node, graph, &bufs);
        let shape = graph.shape(node.output.tensor);
        scatter(
            &mut bufs,
            node.output.tensor,
            node.output.region,
            &out,
            shape,
        );
    }
    bufs
}

/// Compute one node's dense `[out_rows, out_cols]` output. Per-op
/// arithmetic mirrors `cpu_golden`.
pub fn eval_node<F: RopeForm>(
    node: &SubtileNode<F>,
    graph: &SubtileIR<F>,
    bufs: &[Vec<f32>],
) -> Vec<f32> {
    let out_rows = node.output.region.rows.len;
    let out_cols = node.output.region.cols.len;
    match node.op {
        SubOp::MatmulTile { .. } => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            let (w, wr, wc) = gather(&node.inputs[1], graph, bufs);
            // Host-evaluator shape checks — debug-only. The codegen
            // pipeline is the source of truth (`validate()` + the
            // typed witnesses on SubOp); these are defensive on the
            // f32 reference path only. W is `[K, N]` per the FUF
            // convention (see `SubOp::MatmulTile` doc).
            debug_assert_eq!(ac, wr, "matmul K mismatch");
            debug_assert_eq!(ar, out_rows, "matmul A rows vs out_rows");
            debug_assert_eq!(wc, out_cols, "matmul W cols vs out_cols");
            let (m, n, k) = (ar as usize, wc as usize, ac as usize);
            // ⭐ ROW-STREAMING ORDER, SAME SUMS. W is `[K, N]` row-major (`w[l * N + j]`), so
            // the inner loop walks one W ROW contiguously and accumulates `a[i, l] · w[l, :]`
            // into `out[i, :]`. Every output element still adds its `K` products in `l`
            // order from 0.0, so the result is bit-identical to the dot-per-element form —
            // whose inner loop strode a whole W row (N floats) per multiply-add, a cache
            // miss per FMA: MEASURED as the bake gate's dominant cost (gemma-4-12b lead6
            // slice, ~110 s of a 146 s build in `eval_node`).
            let mut out = vec![0f32; m * n];
            for i in 0..m {
                let row = &mut out[i * n..(i + 1) * n];
                for l in 0..k {
                    let s = a[i * k + l];
                    let wrow = &w[l * n..(l + 1) * n];
                    for (o, &wv) in row.iter_mut().zip(wrow) {
                        *o += s * wv;
                    }
                }
            }
            out
        }
        SubOp::SumReduce { .. } => {
            let len = (out_rows * out_cols) as usize;
            let mut out = vec![0f32; len];
            for inp in &node.inputs {
                let (b, _, _) = gather(inp, graph, bufs);
                debug_assert_eq!(b.len(), len, "reduce operand size mismatch");
                for (o, v) in out.iter_mut().zip(&b) {
                    *o += *v;
                }
            }
            out
        }
        SubOp::Sample { vocab, rows: _ } => {
            // The HOST reference states the KERNEL'S OWN LAW, never a
            // backend's argmax tie promise: per row, FIRST-maximum over
            // `score_i = (x_i·inv_t + g_i·gate) + padoff_i` in f32, with `g` the card's
            // own noise — the wrapping-i32 counter hash over the row's seed
            // quadruple and the lane (request-owned alone — no row word,
            // the auditor's 2026-10-27 ruling), the multiply-by-2^8 wrap-fold, the
            // strict-interior folds, and the fp16-island Gumbel, every
            // constant at its staged value (the kernel's exact sequence).
            // The pad law (`padoff` 0.0 on the real lanes, f32::MIN
            // (−FLT_MAX, 0xFF7FFFFF) on the pads — the fp32 reduce
            // template's own "negInf", NEVER IEEE −inf, which the card
            // reads as NaN — added LAST, after the score's two roundings)
            // keeps the pad lanes lost, so the scan over `[0, vocab)` is the
            // whole contest; a NaN score never wins the ordered max and
            // never satisfies the equality, so an all-NaN row answers the
            // sentinel past the last real lane — the consumer's
            // `idx > vocab` refusal, #666's recorded divergence from
            // host-argmax-over-NaN). The output is the `[rows, 64]` idx
            // stick with every lane equal (the store is a whole stick per
            // row); the values are whole numbers below 2^24, exact in f32.
            // The greedy row (gate 0, inv_t 1) is the plain first-max —
            // `g·0 = 0` exact, never a separate law.
            //
            // ⚠ COVERAGE, STATED: the host evaluator's buffers are f32, so
            // the seed words ride as f32 IMAGES and this reference reads
            // them back with `as i32` — exact while every word is below 2^24
            // in magnitude, which the gate's own staging (zeros, the step
            // counter, the row index) always satisfies. A noise-decided draw
            // over full-width seed words is NOT this reference's to check:
            // that fidelity is the card probe's verdict (FB-NOISE-DONE),
            // never the gate's.
            let (block, _, cols) = gather(&node.inputs[0], graph, bufs);
            let (slo_t, _, wcols) = gather(&node.inputs[1], graph, bufs);
            let (shi_t, _, _) = gather(&node.inputs[2], graph, bufs);
            let (stp_t, _, _) = gather(&node.inputs[3], graph, bufs);
            let (sgate_t, _, _) = gather(&node.inputs[4], graph, bufs);
            let (invt, _, _) = gather(&node.inputs[5], graph, bufs);
            assert!(
                wcols == 1,
                "the seed word tables are {wcols} wide, not the one word of their own \
                 `[rows, 1]` class (the mint mints four of them, the quadruple's own)"
            );
            let vocab = vocab as usize;
            assert!(
                cols as usize >= vocab,
                "the sample row is {cols} wide, narrower than the vocab {vocab}"
            );
            let cols = cols as usize;
            // The noise section's constants at their staged values — the
            // hash multipliers and the wrap-fold's 2^8 in i32, the fold's
            // five f32 constants, the fp16 island's clamp (fp16's smallest
            // normal). One citation for the whole chain: sample_fwd.py's
            // noise section, whose sequence this reference states.
            const HC1: i32 = -1640531535; // 0x9E3779B1
            const HC2: i32 = -2048144777; // 0x85EBCA77
            const HC3: i32 = -1028477379; // 0xC2B2AE3D
            const FM8: f32 = 0.00390625; // 2^-8
            const FTOP: f32 = 8388607.0; // 2^23 - 1
            const FZERO: f32 = 0.0;
            const FHALF: f32 = 0.5;
            // 2^-22 and 2^-14, spelled as the division they are (exact in f32 —
            // powers of two, no rounding anywhere in the fold).
            const FM22: f32 = 1.0 / 4194304.0; // 2^-22
            const CMIN: f32 = 1.0 / 16384.0; // 2^-14
            let mut out = Vec::with_capacity((out_rows * out_cols) as usize);
            for r in 0..out_rows as usize {
                let inv_t = invt[r];
                let slo = slo_t[r] as i32;
                let shi = shi_t[r] as i32;
                let stp = stp_t[r] as i32;
                let gate = sgate_t[r] as i32 as f32;
                let base = r * cols;
                // `+ 0.0` is the pad law's own offset, applied LAST — the
                // kernel adds the staged `padoff` after the score's two
                // roundings, and the eval states the kernel's sequence
                // exactly (adding 0.0 to a finite f32 changes nothing). The
                // noise: the wrapping-i32 hash (the card's bit-exact integer
                // class), then `r = h·2^8 mod 2^32` — a multiple of 2^8 with
                // at most 24 significant bits, so `as f32` is exact — then
                // the abs fold, the strict-interior folds, and the fp16
                // island: each f16 boundary rounds through IEEE binary16
                // (`half::f16::from_f32`), each log is the f32 `ln` of an
                // f16-representable operand — exactly the emulator's
                // semantics, the arrangement-blind elementwise law (the
                // card's dl16tofp32 stagger is the arrangement table's fact,
                // never a consumer's assumption).
                let score = |i: usize| {
                    let lane = i as i32;
                    let h = lane.wrapping_mul(HC1).wrapping_add(slo);
                    let h = h.wrapping_mul(HC2).wrapping_add(shi).wrapping_add(stp);
                    let h = h.wrapping_mul(h);
                    let h = h.wrapping_mul(HC1).wrapping_add(slo);
                    let h = h.wrapping_mul(h);
                    let h = h.wrapping_mul(HC3).wrapping_add(shi).wrapping_add(stp);
                    let h = h.wrapping_mul(h);
                    let h = h
                        .wrapping_mul(HC1)
                        .wrapping_add(lane.wrapping_mul(HC2))
                        .wrapping_add(slo);
                    let rf = h.wrapping_mul(256) as f32;
                    let n = rf.abs() * FM8;
                    let nm = n.min(FTOP - n);
                    let np = nm.max(FZERO);
                    let u = (np + FHALF) * FM22;
                    let u16 = half::f16::from_f32(u);
                    let c1 = u16.to_f32().max(CMIN);
                    let lu = half::f16::from_f32(c1.ln());
                    let c2 = (-lu.to_f32()).max(CMIN);
                    let lt = half::f16::from_f32(c2.ln());
                    let noise = (-lt).to_f32();
                    (block[base + i] * inv_t + noise * gate) + 0.0
                };
                let mut m = f32::NEG_INFINITY;
                for i in 0..vocab {
                    let s = score(i);
                    if s > m {
                        m = s;
                    }
                }
                let idx = match (0..vocab).position(|i| score(i) == m) {
                    // A NaN score never wins the `>` scan and never
                    // satisfies `==`, so the first lane equal to the running
                    // max is the first-max; an all-−inf row ties every lane
                    // at m = −inf and answers lane 0.
                    Some(first) => first as u32,
                    // The all-NaN sentinel: the pad lanes stage ginc = G,
                    // binc = GROUP_BLOCKS and linc = BLOCK, so the combine
                    // answers G·4096 + 64·64 + 64 — at every admitted
                    // production shape (PAD_V a whole number of 4096-lane
                    // groups) that is PAD_V + 4160, far past the last real
                    // lane. The kernel's own arithmetic, not the flat
                    // ramp's PAD_V + BLOCK.
                    None => {
                        sample_groups(sample_pad_vocab(vocab as u32))
                            * (SAMPLE_BLOCK * SAMPLE_GROUP_BLOCKS)
                            + SAMPLE_GROUP_BLOCKS * SAMPLE_BLOCK
                            + SAMPLE_BLOCK
                    }
                };
                out.extend(std::iter::repeat_n(idx as f32, out_cols as usize));
            }
            out
        }
        // `replaces` is metadata here: the reference walks the sticks and the
        // matrix, never the replaced row (the re-point's own law — no reader).
        SubOp::EmbedRowGather { hidden, .. } => {
            // The HOST reference states the KERNEL'S OWN LAW: the two index
            // sticks' first words PER ROW (whole numbers below 2^24, exact in
            // f32 — the sticks' eval encoding) re-join into that row's pick —
            // word 0 of the s stick's row is the LANE PATTERN's first lane
            // `k·s + 0`, so `s = word0/k` EXACT for every k (`k` from
            // [`chain_slab_entries`] at this graph's own hidden — the one
            // spelling), word 0 of the l stick's row is the LANE RAMP's
            // first lane, `l` itself ([`chain_l_stick_ramp`]'s word-0 law),
            // then `t = s·64 + l` — and one stride-pad walk down the matrix's
            // column. The two sentinel legs stay: the sentinel ramp's word 0
            // is `hidden` (a real pick's ramp answers a lane
            // `t & 63 ≤ 63 < 64 ≤ hidden`, so the mark can never name a real
            // column), and every pick past the matrix's WIDENED width (the
            // region's own extent — the pad the widening pass stated) clamps
            // to a zero row, while a pick in `[vocab, pad)` reads the pad
            // columns' own zeros — the device must never read past the matrix
            // on an already-broken chat, and neither may the reference.
            let (s_stick, _, s_cols) = gather(&node.inputs[0], graph, bufs);
            let (l_stick, _, l_cols) = gather(&node.inputs[1], graph, bufs);
            let k = chain_slab_entries(u64::from(hidden)) as usize;
            let (w, _, wcols) = gather(&node.inputs[2], graph, bufs);
            let pad = wcols as usize;
            let hidden = hidden as usize;
            let width = out_rows as usize;
            assert_eq!(
                out_cols as usize, hidden,
                "the chained rows are [width, hidden], one row per stick row"
            );
            let mut out = vec![0f32; width * hidden];
            for r in 0..width {
                let s = s_stick[r * s_cols as usize] as usize / k;
                let l = l_stick[r * l_cols as usize] as usize;
                assert!(
                    (l as u32) < CHAIN_SLAB_STICK_ELEMS || l as u32 == hidden as u32,
                    "the lane stick answers {l}, neither a lane below the stick width \
                     nor the sentinel mark hidden = {hidden} — a corrupted chain, not a pick"
                );
                let t = s * SAMPLE_BLOCK as usize + l;
                if l != hidden && t < pad {
                    for (h, o) in out[r * hidden..(r + 1) * hidden].iter_mut().enumerate() {
                        *o = w[h * pad + t];
                    }
                }
            }
            out
        }
        SubOp::ChainStage { pad_vocab, k, .. } => {
            // The persistent s/l writes are GPU-only (KvAppend's law); the
            // reference is the S-VALUE STICK the device's own s-gather
            // writes into this node's fresh output, so the two ends agree on
            // what the node produced: the ruled table's law, `s = t >> 6`
            // clamped to 0 past `pad_vocab`, stated as the LANE PATTERN
            // `k·s + (i mod k)` over the stick's lanes — lane `i` names
            // entry `i mod k` of slab `s` (the sentinel leg — a degenerate
            // row answers a pick past `pad_vocab`, and the clamp keeps even
            // that read on slab 0's pattern `i mod k`, the zero-slab form
            // the k=1 splat's own zero already was). The pick rides word 0
            // of each idx-stick row (the whole-stick store law).
            // `table_rows` is unread here by the same law that leaves the
            // table OPERANDS unread: the reference states the decompose's
            // RESULT, never its walk, and the extent is the door's check.
            let (pick, _, pick_cols) = gather(&node.inputs[0], graph, bufs);
            let k = k as usize;
            let width = out_rows as usize;
            assert_eq!(
                out_cols as usize, CHAIN_IDX_BLOCK as usize,
                "the stage's fresh output is the s-value stick, [width, CHAIN_IDX_BLOCK]"
            );
            let mut out = Vec::with_capacity(width * CHAIN_IDX_BLOCK as usize);
            for r in 0..width {
                let t = pick[r * pick_cols as usize] as u32;
                let s = if t < pad_vocab { t >> 6 } else { 0 };
                out.extend((0..CHAIN_IDX_BLOCK as usize).map(|i| (k * s as usize + i % k) as f32));
            }
            out
        }
        SubOp::Reshape { .. } => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            // A reshape moves no data in the HOST/logical view — the element
            // sequence is identical and only the extents change. It refuses on
            // a count mismatch rather than truncating or zero-filling, because
            // a reshape that does not preserve the count is not a reshape.
            assert_eq!(
                (ar * ac) as usize,
                (out_rows * out_cols) as usize,
                "reshape must preserve the element count: [{ar}, {ac}] -> [{out_rows}, {out_cols}]"
            );
            a
        }
        SubOp::Elementwise(kind) => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            debug_assert_eq!((ar, ac), (out_rows, out_cols), "elementwise shape");
            match kind {
                EwKind::Silu => a.iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
                // The TANH approximation, not the erf form — this is the
                // reference the device's `gelu` SFP polynomial approximates, so
                // matching it here keeps `eval_dag` a usable oracle for the
                // on-card result. (`EwKind::GeluErf` is a DISTINCT op and
                // must not fold into this one.)
                EwKind::Gelu => a
                    .iter()
                    .map(|&x| {
                        let inner =
                            (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x * x * x);
                        0.5 * x * (1.0 + inner.tanh())
                    })
                    .collect(),
                EwKind::QuickGelu => a.iter().map(|&x| x / (1.0 + (-1.702 * x).exp())).collect(),
                // The EXACT erf form. `erf` is not in std, so this is the
                // Abramowitz & Stegun 7.1.26 rational approximation — max |err|
                // 1.5e-7, three orders below f32's own rounding, so the oracle
                // is limited by the format and not by the approximation.
                EwKind::GeluErf => a
                    .iter()
                    .map(|&x| {
                        let z = x / std::f32::consts::SQRT_2;
                        let sign = if z < 0.0 { -1.0f32 } else { 1.0 };
                        let t = 1.0 / (1.0 + 0.327_591_1 * z.abs());
                        let poly = t
                            * (0.254_829_6
                                + t * (-0.284_496_74
                                    + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
                        let erf = sign * (1.0 - poly * (-z * z).exp());
                        0.5 * x * (1.0 + erf)
                    })
                    .collect(),
                EwKind::Relu2 => a.iter().map(|&x| x.max(0.0) * x.max(0.0)).collect(),
                EwKind::Mul | EwKind::Add | EwKind::Sub => {
                    // `Sub` broadcasts a `[m, 1]` right operand across the row (the
                    // mean-centering); `Mul`/`Add` are same-shape. One gather, one
                    // index rule, so a broadcast operand cannot be read as dense.
                    let (b, br, bc) = gather(&node.inputs[1], graph, bufs);
                    debug_assert_eq!(br, ar, "elementwise binary rows");
                    debug_assert!(
                        bc == ac || bc == 1,
                        "elementwise binary cols must match or broadcast: {bc} vs {ac}"
                    );
                    let d = ac as usize;
                    (0..a.len())
                        .map(|i| {
                            let x = a[i];
                            let y = if bc == 1 { b[i / d] } else { b[i] };
                            match kind {
                                EwKind::Mul => x * y,
                                EwKind::Add => x + y,
                                _ => x - y,
                            }
                        })
                        .collect()
                }
                EwKind::BiasAdd => {
                    // The bias is ONE `[1, n]` row, added to every row of `a`.
                    let (b, br, bc) = gather(&node.inputs[1], graph, bufs);
                    debug_assert_eq!((br, bc), (1, ac), "bias_add bias must be one [1, n] row");
                    let d = ac as usize;
                    (0..a.len()).map(|i| a[i] + b[i % d]).collect()
                }
            }
        }
        SubOp::ScalarMul { scale } => {
            // Shape-preserving unary scale; the [rows,cols] identity is a
            // structural property of the op (not a checked invariant), so
            // no shape assertion — see `never_use_runtime_assertions`.
            let (a, _, _) = gather(&node.inputs[0], graph, bufs);
            a.iter().map(|&x| x * scale).collect()
        }
        SubOp::SiluMul => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            let (b, br, bc) = gather(&node.inputs[1], graph, bufs);
            debug_assert_eq!((ar, ac), (out_rows, out_cols), "silu_mul gate shape");
            debug_assert_eq!((br, bc), (ar, ac), "silu_mul up shape");
            a.iter()
                .zip(&b)
                .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
                .collect()
        }
        SubOp::RmsNorm { eps, gain } => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            let (wt, _wr, wc) = gather(&node.inputs[1], graph, bufs);
            debug_assert_eq!((xr, xc), (out_rows, out_cols), "rmsnorm shape");
            debug_assert_eq!(wc, out_cols, "rmsnorm weight width");
            let (m, d) = (xr as usize, xc as usize);
            let mut out = vec![0f32; m * d];
            for i in 0..m {
                let row = &x[i * d..(i + 1) * d];
                let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
                let inv_rms = 1.0 / (sum_sq / d as f32 + eps).sqrt();
                for j in 0..d {
                    out[i * d + j] = row[j] * inv_rms * gain.apply(wt[j]);
                }
            }
            out
        }
        SubOp::RmsNormReduce { eps, .. } => {
            // Phase 1: compute inv_rms[i] = 1/sqrt(mean(x[i,:]^2) + eps).
            // Output is `[m, 1]` — one scalar per row.
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            debug_assert_eq!(out_rows, xr, "rms_reduce out_rows vs x rows");
            debug_assert_eq!(out_cols, 1, "rms_reduce output must be [m, 1]");
            let (m, d) = (xr as usize, xc as usize);
            let mut out = vec![0f32; m];
            for i in 0..m {
                let row = &x[i * d..(i + 1) * d];
                let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
                out[i] = 1.0 / (sum_sq / d as f32 + eps).sqrt();
            }
            out
        }
        SubOp::RmsNormApply { gain, .. } => {
            // Phase 2: y[m, chunk_cols] = x * inv_rms[per row] * gamma[per col].
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            let (inv_rms, ir, ic) = gather(&node.inputs[1], graph, bufs);
            let (gamma, _gr, gc) = gather(&node.inputs[2], graph, bufs);
            debug_assert_eq!((xr, xc), (out_rows, out_cols), "rms_apply x shape");
            debug_assert_eq!(ir, out_rows, "rms_apply inv_rms rows");
            debug_assert_eq!(ic, 1, "rms_apply inv_rms cols must be 1");
            debug_assert_eq!(gc, out_cols, "rms_apply gamma cols");
            let (m, d) = (xr as usize, xc as usize);
            let mut out = vec![0f32; m * d];
            for i in 0..m {
                let inv = inv_rms[i];
                for j in 0..d {
                    out[i * d + j] = x[i * d + j] * inv * gain.apply(gamma[j]);
                }
            }
            out
        }
        // KvAppend writes K as projected: the identity on K; its V input +
        // the paged-cache write are GPU-only.
        SubOp::KvAppend { .. } => gather(&node.inputs[0], graph, bufs).0,
        // RopeAppend's host eval is rotation only (identical to RopeRotate);
        // its V input + the paged-cache write are GPU-only.
        SubOp::RopeRotate { head_dim, _form: _ } | SubOp::RopeAppend { head_dim, .. } => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            let (cos, cr, cc) = gather(&node.inputs[1], graph, bufs);
            let (sin, sr, sc) = gather(&node.inputs[2], graph, bufs);
            debug_assert_eq!((xr, xc), (out_rows, out_cols), "rope shape");
            let hd = head_dim.get() as usize;
            let half = hd / 2;
            let (rows, cols) = (xr as usize, xc as usize);
            debug_assert_eq!(cols % hd, 0, "rope cols not a multiple of head_dim");
            debug_assert!(cc as usize >= hd && sc as usize >= hd, "rope cos/sin width");
            let heads = cols / hd;
            // ⭐ ROW- AND HEAD-ALIGNED TABLE READS, BY THE SOURCE'S OWN SHAPE. A real
            // tape's cos/sin source is `[m, cols]` — one row per query row, each the
            // row's own position's table tiled across heads to the rope's full width
            // (`to_wavefront`'s `cos_sin` pushes exactly that shape, and both runtimes
            // stage it: `rope_cos_sin_kind(position[row])` then `tile_rotary_row`) — so
            // row `r`'s rotation reads row `r`'s table at the head's own slice. The old
            // `cos[d]`-for-every-row form is right only at m == 1 (one row, and the
            // period-half probe synth makes even a wrong read agree); at m > 1 it
            // rotates every row by ROW 0's angles. A `[1, hd]` source (the fixtures'
            // single head table) stays broadcast across rows and heads. Any other
            // width has no staging law — panic rather than silently misread.
            let table_row = |tr: usize, r: usize| if tr == 1 { 0 } else { r };
            let table_base = |tc: usize, h: usize| {
                if tc == cols {
                    h * hd
                } else if tc == hd {
                    0
                } else {
                    panic!(
                        "rope cos/sin source width {tc} is neither the rope's own width \
                         ({cols}) nor one head_dim ({hd}) — no staging law to read it by"
                    )
                }
            };
            let mut out = x.clone();
            for r in 0..rows {
                let (crow, srow) = (table_row(cr as usize, r), table_row(sr as usize, r));
                for h in 0..heads {
                    let base = r * cols + h * hd;
                    let (coff, soff) = (
                        crow * cc as usize + table_base(cc as usize, h),
                        srow * sc as usize + table_base(sc as usize, h),
                    );
                    for d in 0..half {
                        let x0 = x[base + d];
                        let x1 = x[base + half + d];
                        // Apply the cos/sin table ELEMENT-WISE (cos[base + d] for the first
                        // half, cos[base + half + d] for the second), matching the SuperDSC
                        // on-card RoPE (`out = x*cos + rotate_half(x)*sin`, where the pointwise
                        // mul reads the full [hd] table per element). For REAL rope_cos_sin
                        // tables the two halves are duplicated (cos[d+half]==cos[d]), so this
                        // is a NO-OP for every real run; it only matters when cos/sin are NOT
                        // duplicated (the selftest's synthetic sources), where the old
                        // `cos[d]`-for-both form made the eval_dag golden diverge from the
                        // faithful on-card lowering (the spurious t341 RopeRotate cos≈0.45
                        // selftest artifact).
                        out[base + d] = x0 * cos[coff + d] - x1 * sin[soff + d];
                        out[base + half + d] =
                            x0 * sin[soff + half + d] + x1 * cos[coff + half + d];
                    }
                }
            }
            out
        }
        // ⛔ A COLLECTIVE HAS NO ONE-RANK VALUE: it reads every rank's operand. The sharded
        // forward that carries one is evaluated by `tensor_parallel::eval_ranks`, which runs
        // every other node through here and the collectives across ranks itself.
        SubOp::AllReduce | SubOp::AllGather { .. } => panic!(
            "{} read on one rank: a sharded forward is evaluated by \
             tensor_parallel::eval_ranks",
            node.op.name()
        ),
        SubOp::Mean => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            debug_assert_eq!(out_cols, 1, "mean output must be [m, 1]");
            debug_assert_eq!(out_rows, xr, "mean row count");
            let d = xc as usize;
            (0..xr as usize)
                .map(|i| x[i * d..(i + 1) * d].iter().sum::<f32>() / d as f32)
                .collect()
        }
        SubOp::RmsNormUnit { eps } => {
            // Unit gain: no gamma operand at all, so there is nothing to apply a
            // GainConvention to. Same reduce as RmsNorm, no scale.
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            debug_assert_eq!((xr, xc), (out_rows, out_cols), "rmsnorm_unit shape");
            let (m, d) = (xr as usize, xc as usize);
            let mut out = vec![0f32; m * d];
            for i in 0..m {
                let row = &x[i * d..(i + 1) * d];
                let inv_rms =
                    1.0 / (row.iter().map(|&v| v * v).sum::<f32>() / d as f32 + eps).sqrt();
                for j in 0..d {
                    out[i * d + j] = row[j] * inv_rms;
                }
            }
            out
        }
        SubOp::ScalarWeightMul => {
            // The weight is a `[1]`-shaped SOURCE (gemma4 `layer_scalar[layer]`),
            // not a compile-time constant like ScalarMul's.
            let (x, _, _) = gather(&node.inputs[0], graph, bufs);
            let (w, _, _) = gather(&node.inputs[1], graph, bufs);
            debug_assert_eq!(w.len(), 1, "scalar weight is [1]-shaped");
            x.iter().map(|&v| v * w[0]).collect()
        }
        SubOp::GateApply => {
            let (attn, ar, ac) = gather(&node.inputs[0], graph, bufs);
            let (gate, gr, gc) = gather(&node.inputs[1], graph, bufs);
            debug_assert_eq!((ar, ac), (out_rows, out_cols), "gate_apply shape");
            debug_assert_eq!((gr, gc), (ar, ac), "gate_apply gate shape");
            attn.iter()
                .zip(&gate)
                .map(|(&a, &g)| a / (1.0 + (-g).exp()))
                .collect()
        }
        SubOp::Concat { cols } => {
            let (a, ar, ac) = gather(&node.inputs[0], graph, bufs);
            let (b, br, bc) = gather(&node.inputs[1], graph, bufs);
            debug_assert_eq!((ar, ac + bc), (out_rows, out_cols), "concat shape");
            debug_assert_eq!((br, bc), (ar, ac), "concat operands share a shape");
            debug_assert_eq!(ac + bc, cols, "concat width");
            let w = ac as usize;
            a.chunks(w)
                .zip(b.chunks(w))
                .flat_map(|(ra, rb)| ra.iter().chain(rb).copied())
                .collect()
        }
        SubOp::GateScale => {
            // `g` is `[m, 1]`, broadcast across the row.
            let (routed, rr, rc) = gather(&node.inputs[0], graph, bufs);
            let (shared, _sr, _sc) = gather(&node.inputs[1], graph, bufs);
            let (g, _gr, gc) = gather(&node.inputs[2], graph, bufs);
            debug_assert_eq!((rr, rc), (out_rows, out_cols), "gate_scale shape");
            debug_assert_eq!(gc, 1, "gate_scale g must be [m, 1]");
            let d = rc as usize;
            (0..routed.len())
                .map(|i| {
                    let gate = 1.0 / (1.0 + (-g[i / d]).exp());
                    routed[i] + shared[i] * gate
                })
                .collect()
        }
        SubOp::EncoderAttn { geom, scale } => {
            // Bidirectional: no causal bound and no cache. Q/K/V are whole
            // contiguous slots from upstream, so this is AttnDecode's inner loop
            // with the mask removed.
            let hd = geom.hd().get() as usize;
            let gqa = geom.gqa().get() as usize;
            let (q, qr, qc) = gather(&node.inputs[0], graph, bufs);
            let (k, kr, kc) = gather(&node.inputs[1], graph, bufs);
            let (v, _vr, _vc) = gather(&node.inputs[2], graph, bufs);
            let (t, qh) = (qr as usize, qc as usize / hd);
            let (s, kvh) = (kr as usize, kc as usize / hd);
            let mut out = vec![0f32; t * qh * hd];
            for qi in 0..t {
                for h in 0..qh {
                    let kvh_i = (h / gqa).min(kvh - 1);
                    let q_off = qi * qh * hd + h * hd;
                    let mut scores = vec![0f32; s];
                    for (si, sc) in scores.iter_mut().enumerate() {
                        let k_off = si * kvh * hd + kvh_i * hd;
                        *sc = (0..hd).map(|d| q[q_off + d] * k[k_off + d]).sum::<f32>() * scale;
                    }
                    let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut denom = 0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - mx).exp();
                        denom += *sc;
                    }
                    for (si, &score) in scores.iter().enumerate() {
                        let w = score / denom;
                        let v_off = si * kvh * hd + kvh_i * hd;
                        for d in 0..hd {
                            out[q_off + d] += w * v[v_off + d];
                        }
                    }
                }
            }
            out
        }

        // ⭐ THE CAP IS ON THE TAPE now, so the host reference exists — the same
        // `cap · tanh(x / cap)` every target's emitter states.
        SubOp::TanhSoftCap { cap } => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            debug_assert_eq!((xr, xc), (out_rows, out_cols), "tanh_softcap shape");
            x.iter().map(|&v| cap * (v / cap).tanh()).collect()
        }

        // ── Ops with NO host reference ──────────────────────────────
        //
        // ⛔ THESE PANIC, AND THE MESSAGE NAMES WHAT IS MISSING. `eval_node` is the
        // numeric ORACLE — the thing an on-card result is compared against. An op
        // whose value depends on state the host does not have (a runtime index
        // table, a staged pixel buffer, an ssm carry, an opaque weight bundle that
        // is not a gatherable tensor) has no oracle, and returning zeros or the
        // input unchanged would make the comparison PASS while proving nothing.
        // That is strictly worse than no oracle, so it is not on offer.
        SubOp::GateSplit { .. } => panic!(
            "SubOp::GateSplit has no host reference: it has TWO outputs (q and gate) \
             and eval_node returns the buffer for ONE region, so the gate half would \
             be silently dropped"
        ),
        SubOp::LoadRows { .. } => panic!(
            "SubOp::LoadRows has no host reference: the buffer is STAGED BY THE HOST at \
             runtime (patchified pixels, interpolated position embeddings, a target's \
             hidden states) and is not computed from operands"
        ),
        SubOp::EmbeddingGather { .. } => panic!(
            "SubOp::EmbeddingGather has no host reference: the permutation is a \
             runtime index table (window_index / reverse_indices), not an operand"
        ),
        SubOp::VisionRope => panic!(
            "SubOp::VisionRope has no host reference: the 2D grid positions come from \
             the runtime image grid, and it has TWO outputs (q and k, in place)"
        ),
        SubOp::VarlenAttention { .. } => panic!(
            "SubOp::VarlenAttention has no host reference: the sequence boundaries are \
             the runtime cu_seqlens/max_seqlen extern set, not operands"
        ),
        SubOp::GatedDeltaNet | SubOp::Mamba2 { .. } => panic!(
            "{} has no host reference: the conv/ssm state is runtime-ambient per layer, so a \
             single node's output is not a function of its operands alone",
            node.op.name()
        ),
        expansion_ops!() => panic!(
            "{} has no host reference: the router and expert weights arrive as opaque \
             bundle sources, a KV codec's packed stores as runtime buffers, and the sampled \
             rows at the runtime's sequence boundaries, none of which `gather` can read as \
             a tensor region",
            node.op.name()
        ),

        SubOp::AttnDecode {
            geom, scale, mask, ..
        } => {
            // ⭐ THE WINDOW IS ON THE TAPE, so a sliding-window layer HAS a host
            // reference: the window bound is `window` positions back from the
            // query's own, the same left-edge every sliding kernel reads. Without
            // it (a bare causal bound) the reference would silently attend across
            // the whole prefix and report agreement with a kernel that does not
            // do that — which is why this used to refuse.
            let window = mask.window().map(|w| w.get() as usize);
            // Head-block aware: this node computes a contiguous q-head range,
            // derived from the OUTPUT region (its column slice), and reads the
            // matching kv-head range — derived from the K input region. The
            // SubOp keeps the GLOBAL head geometry so the GQA ratio is exact; the
            // block's own counts come from the slice widths. Each q-head's
            // attention is independent ⇒ bit-exact vs the whole op.
            let hd = geom.hd().get() as usize;
            let gqa = geom.gqa().get() as usize;
            let (q, qr, qc) = gather(&node.inputs[0], graph, bufs);
            let mq = qr as usize;
            let qh_count = qc as usize / hd; // q-heads in this block
            let qh_start = node.output.region.cols.start as usize / hd; // global first q-head
            debug_assert_eq!(
                qc as usize,
                qh_count * hd,
                "attn Q width is a head multiple"
            );
            debug_assert!(node.inputs.len() >= 3, "attn needs Q + >=1 (K,V) segment");
            // The gpt-oss trailing sinks weight: the registry's arity
            // closure admits `Q + (K,V) pairs` (odd count) or exactly 6
            // = the 5-input decode form plus ONE trailing weight-source
            // input. An even count therefore marks the LAST input as
            // the per-head `[num_heads]` sinks — read it here and stop
            // the (K,V) walk before it (the pair loop below would
            // otherwise gather past the end).
            let sinks: Option<Vec<f32>> = if node.inputs.len().is_multiple_of(2) {
                Some(gather(&node.inputs[node.inputs.len() - 1], graph, bufs).0)
            } else {
                None
            };
            let attn_len = node.inputs.len() - sinks.is_some() as usize;
            debug_assert_eq!(
                attn_len % 2,
                1,
                "attn inputs = Q + (K,V) pairs (+ optional trailing sinks)"
            );
            // kv-head offset of this block (from the first K segment's column slice).
            let kvh_start = node.inputs[1].region.cols.start as usize / hd;
            // Concatenate K/V segments along sequence; each segment spans this
            // block's kv-heads (kv_count * hd wide).
            let mut k_all: Vec<f32> = Vec::new();
            let mut v_all: Vec<f32> = Vec::new();
            let mut kv_count = 0usize;
            let mut i = 1;
            while i < attn_len {
                let (k, kr, kc) = gather(&node.inputs[i], graph, bufs);
                let (v, vr, vc) = gather(&node.inputs[i + 1], graph, bufs);
                debug_assert_eq!((vr, vc), (kr, kc), "attn V seg shape");
                kv_count = kc as usize / hd;
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(&v);
                i += 2;
            }
            debug_assert!(kv_count >= 1, "attn K seg has at least one kv-head");
            let seg_w = kv_count * hd;
            let seq_len = k_all.len() / seg_w;
            let mut out = vec![0f32; mq * qh_count * hd];
            for qi in 0..mq {
                for hl in 0..qh_count {
                    // local q-head hl → global q-head → global kv-head → local kv.
                    let global_kv = (qh_start + hl) / gqa;
                    let local_kv = global_kv - kvh_start;
                    let q_off = qi * qh_count * hd + hl * hd;
                    // Causal boundary: the mq query rows are the LAST mq positions
                    // of the concatenated sequence, so query row qi sits at absolute
                    // position `seq_len - mq + qi` and may attend to keys 0..=that.
                    // For mq==1 this is `seq_len-1` (all keys) ⇒ decode is unchanged.
                    // A sliding layer additionally drops everything more than
                    // `window` positions behind its own — the left edge
                    // `own_pos + 1 - window`, the same bound every sliding kernel
                    // reads (`kv_len-1 - k >= window` masked out).
                    let own_pos = seq_len - mq + qi;
                    let left = match window {
                        Some(w) => own_pos + 1 - w.min(own_pos + 1),
                        None => 0,
                    };
                    let mut scores = vec![0f32; seq_len];
                    for (s, score) in scores.iter_mut().enumerate() {
                        if s > own_pos || s < left {
                            *score = f32::NEG_INFINITY;
                            continue;
                        }
                        let k_off = s * seg_w + local_kv * hd;
                        let mut dot = 0f32;
                        for d in 0..hd {
                            dot += q[q_off + d] * k_all[k_off + d];
                        }
                        *score = dot * scale;
                    }
                    // The sink column: appended after qk·scale, UNSCALED
                    // (sink logits are never multiplied by sm_scale),
                    // participates in the softmax max/denominator, and
                    // is dropped before the ·V accumulation below — it
                    // contributes to normalization only.
                    let sink_col = sinks.as_ref().map(|s| s[qh_start + hl]);
                    if let Some(sink) = sink_col {
                        scores.push(sink);
                    }
                    let maxs = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - maxs).exp();
                        sum += *sc;
                    }
                    for sc in scores.iter_mut() {
                        *sc /= sum;
                    }
                    if sink_col.is_some() {
                        scores.pop();
                    }
                    for d in 0..hd {
                        let mut val = 0f32;
                        for (s, &wgt) in scores.iter().enumerate() {
                            val += wgt * v_all[s * seg_w + local_kv * hd + d];
                        }
                        out[q_off + d] = val;
                    }
                }
            }
            out
        }
    }
}

/// The forward result buffer (logits) — `bufs[graph.result]`.
pub fn result_buffer<'a, F: RopeForm>(graph: &SubtileIR<F>, bufs: &'a [Vec<f32>]) -> &'a [f32] {
    &bufs[graph.result.0 as usize]
}

// ── Dependencies (region overlap) ──────────────────────────────────

fn ranges_overlap(a: Range, b: Range) -> bool {
    a.start < b.end() && b.start < a.end()
}

fn regions_overlap(a: Region, b: Region) -> bool {
    ranges_overlap(a.rows, b.rows) && ranges_overlap(a.cols, b.cols)
}

/// Predecessor node ids for each node: the earlier nodes whose write
/// overlaps one of this node's input reads on the same op-output tensor.
/// Reads of leaf sources contribute no dependency. This is the edge set
/// the per-target lowering turns into cross-execution-unit
/// synchronization (whatever primitive the target prefers).
pub fn predecessors<F: RopeForm>(graph: &SubtileIR<F>) -> Vec<Vec<SubtileId>> {
    // writers[t] = (node_id, out_region) for each op-output tensor, in id order.
    let mut writers: Vec<Vec<(u32, Region)>> = vec![Vec::new(); graph.tensors.len()];
    let mut preds: Vec<Vec<SubtileId>> = Vec::with_capacity(graph.nodes.len());
    for node in &graph.nodes {
        let mut p: Vec<SubtileId> = Vec::new();
        for inp in &node.inputs {
            if graph.is_source(inp.tensor) {
                continue;
            }
            for (wid, wreg) in &writers[inp.tensor.0 as usize] {
                if regions_overlap(*wreg, inp.region) && !p.contains(&SubtileId(*wid)) {
                    p.push(SubtileId(*wid));
                }
            }
        }
        p.sort();
        preds.push(p);
        writers[node.output.tensor.0 as usize].push((node.id.0, node.output.region));
    }
    preds
}

// ── Structural validation + ValidatedGraph<F> witness ──────────────

/// Sealed proof that a [`SubtileIR<F>`] passed [`validate`]. The only
/// way to obtain one is [`ValidatedGraph::new`] — internally calls
/// [`validate`] and on success wraps the borrow with a sealed marker.
/// Downstream lowerings (`lower_dag_to_tape`) take
/// `&ValidatedGraph<F>` and can elide their own runtime
/// validate-and-expect, so structural-precondition violations become
/// "no value to consume" type errors rather than runtime panics
/// rather than runtime panics.
pub struct ValidatedGraph<'g, F: RopeForm> {
    inner: &'g SubtileIR<F>,
    _seal: sealed::Seal,
}

impl<'g, F: RopeForm> ValidatedGraph<'g, F> {
    /// Validate `graph` and produce the sealed witness. Returns the
    /// validation error string verbatim on failure.
    pub fn new(graph: &'g SubtileIR<F>) -> Result<Self, String> {
        validate(graph)?;
        Ok(Self {
            inner: graph,
            _seal: sealed::Seal(()),
        })
    }

    /// Borrow the underlying graph. Consumers cannot fabricate a
    /// `ValidatedGraph` without going through [`Self::new`], so this
    /// borrow is proof-carrying.
    pub fn graph(&self) -> &'g SubtileIR<F> {
        self.inner
    }
}

/// Check the graph's invariants without evaluating: dense ids, in-range
/// tensors, in-bounds regions, op-output (not source) write targets,
/// op arity, and that every op-output read is covered by writers with a
/// strictly smaller id (acyclic + assembled-before-read). Returns the
/// node count on success. Prefer [`ValidatedGraph::new`] in the
/// wavefront lowerings (the typed witness elides downstream runtime
/// gates).
pub fn validate<F: RopeForm>(graph: &SubtileIR<F>) -> Result<usize, String> {
    let n_tensors = graph.tensors.len() as u32;
    if graph.num_sources > n_tensors {
        return Err(format!(
            "num_sources {} exceeds tensor count {n_tensors}",
            graph.num_sources
        ));
    }
    // Track written sub-regions per op-output tensor to check coverage.
    let mut writes: Vec<Vec<(u32, Region)>> = vec![Vec::new(); graph.tensors.len()];
    for (i, node) in graph.nodes.iter().enumerate() {
        if node.id.0 as usize != i {
            return Err(format!("node {i} has non-dense id {}", node.id.0));
        }
        let arity = node.inputs.len();
        // Arity comes from THE op registry (`crate::ops`) — one row per
        // op, no hand-maintained parallel match.
        if !node.op.arity_ok(arity) {
            return Err(format!("node {i} op {:?} bad arity {arity}", node.op));
        }
        // Output must target an op-output tensor, in bounds.
        let ot = node.output.tensor;
        if ot.0 >= n_tensors {
            return Err(format!("node {i} writes tensor {} out of range", ot.0));
        }
        if graph.is_source(ot) {
            return Err(format!("node {i} writes leaf source tensor {}", ot.0));
        }
        let oshape = graph.shape(ot);
        if node.output.region.rows.end() > oshape.rows
            || node.output.region.cols.end() > oshape.cols
        {
            return Err(format!("node {i} output region exceeds tensor {}", ot.0));
        }
        // Inputs in bounds; op-output reads must be covered by prior writers.
        for (a, inp) in node.inputs.iter().enumerate() {
            if inp.tensor.0 >= n_tensors {
                return Err(format!(
                    "node {i} input {a} tensor {} out of range",
                    inp.tensor.0
                ));
            }
            let ishape = graph.shape(inp.tensor);
            if inp.region.rows.end() > ishape.rows || inp.region.cols.end() > ishape.cols {
                return Err(format!(
                    "node {i} op {:?} input {a} region rows..{} cols..{} exceeds \
                     tensor {} shape [rows {}, cols {}]",
                    node.op,
                    inp.region.rows.end(),
                    inp.region.cols.end(),
                    inp.tensor.0,
                    ishape.rows,
                    ishape.cols,
                ));
            }
            if !graph.is_source(inp.tensor) {
                // Some earlier write must overlap (cheap acyclicity / use-before-def check).
                let has = writes[inp.tensor.0 as usize]
                    .iter()
                    .any(|(wid, wreg)| *wid < node.id.0 && regions_overlap(*wreg, inp.region));
                if !has {
                    return Err(format!(
                        "node {i} input {a} reads tensor {} region with no prior writer",
                        inp.tensor.0
                    ));
                }
            }
        }
        writes[ot.0 as usize].push((node.id.0, node.output.region));
    }
    if graph.result.0 >= n_tensors {
        return Err(format!("result tensor {} out of range", graph.result.0));
    }
    if graph.is_source(graph.result) {
        return Err("result is a leaf source, not an op output".into());
    }
    Ok(graph.nodes.len())
}

// ── LoweringInput → SubtileIR ──────────────────────────────────────

/// Tile `[0, total)` into contiguous blocks of width `block` (last block
/// may be shorter). `block.get() >= total` yields a single whole block.
///
/// `block: NonZeroU32` discharges the load-bearing termination invariant
/// at the type level — `block == 0` would loop forever, and that must
/// be a compile error, not a runtime assert.
pub fn n_blocks(total: u32, block: std::num::NonZeroU32) -> Vec<Range> {
    let block = block.get();
    let mut out = Vec::new();
    let mut start = 0;
    while start < total {
        let len = block.min(total - start);
        out.push(Range::new(start, len));
        start += len;
    }
    if out.is_empty() {
        out.push(Range::new(0, 0)); // total==0 (degenerate); keep one empty block
    }
    out
}

/// Tile `[0, total)` into **head-aligned** blocks of width ~`nb`, snapped
/// down to a whole number of `head_dim`-wide heads (at least one head). Used
/// for rope/attention so every block is a clean set of heads — the q→rope→
/// attn chain partitions on head boundaries (and o_proj split-Ks on them).
///
/// `nb: NonZeroU32` propagates the same termination witness as
/// [`n_blocks`]; `head_dim: NonZeroU32` makes the K5 termination
/// invariant structural — `block: NonZeroU32` is constructed without
/// any `.expect()`, since `NonZeroU32::saturating_mul` preserves
/// nonzero by type.
pub fn head_blocks(
    total: u32,
    nb: std::num::NonZeroU32,
    head_dim: std::num::NonZeroU32,
) -> Vec<Range> {
    use std::num::NonZeroU32;
    // `nb.get() / hd` may be zero (when nb < hd); fall back to one
    // head — `NonZeroU32::new(...).unwrap_or(NonZeroU32::MIN)` is the
    // canonical "floor at 1" idiom on a NonZeroU32-output path.
    let heads_per_block = NonZeroU32::new(nb.get() / head_dim.get()).unwrap_or(NonZeroU32::MIN);
    let block = heads_per_block.saturating_mul(head_dim);
    n_blocks(total, block)
}

/// Lower a flat [`crate::lower::LoweringInput`] to a SubtileIR,
/// **N-block tiling every GEMM** by `nb` (output columns split into
/// `ceil(n/nb)` MatmulTile subtiles, each writing a disjoint column
/// slice of the op's output tensor — no reduce, bit-exact). All other
/// ops stay whole (one subtile writing the whole output tensor).
/// `nb >= n` ⇒ a single block (coarse, equivalent to v1). Source tensors
/// mirror `input.sources`; op-output tensor `i` is
/// `TensorId(num_sources + i)`.
/// Lower a flat [`crate::lower::LoweringInput`] to a `SubtileIR<NeoX>`.
/// Llama-3.2 uses NeoX rotary; Interleaved-form lowerings (other
/// architectures) construct `SubtileIR<Interleaved>` directly. Mixing
/// forms in one IR is impossible by construction (the IR's rope nodes
/// carry `PhantomData<F>`, so a SubtileIR<NeoX> cannot hold an
/// Interleaved-form rope node).
///
/// Each node performs its source op — the SAME [`SubOp`], moved from the [`Arch`] stage to the
/// [`Tiled`] one: the KV-cache witnesses are bound here and nothing else about the op changes.
pub fn lower_region(
    input: &crate::lower::LoweringInput,
    nb: std::num::NonZeroU32,
) -> SubtileIR<NeoX> {
    use crate::lower::InputRef;
    let num_sources = input.sources.len() as u32;
    let mut tensors: Vec<TensorShape> = input
        .sources
        .iter()
        .map(|s| TensorShape {
            rows: s.rows,
            cols: s.cols,
        })
        .collect();
    let mut op_tensor: Vec<TensorId> = Vec::with_capacity(input.ops.len());
    let mut op_cols: Vec<u32> = Vec::with_capacity(input.ops.len());
    let mut nodes: Vec<SubtileNode<NeoX>> = Vec::new();
    // RopeAppend node id keyed by the K-cache TensorId it writes — used
    // to compute `KvCacheProducer` for any AttnDecode that reads the
    // same cache later in the forward.
    let mut k_cache_producer_node: std::collections::HashMap<TensorId, u32> =
        std::collections::HashMap::new();
    let mut next_softmax_state: u32 = 0;

    // Resolve an InputRef to (tensor id, shape).
    let resolve = |r: InputRef,
                   op_tensor: &[TensorId],
                   op_cols: &[u32],
                   tensors: &[TensorShape]|
     -> (TensorId, u32, u32) {
        match r {
            InputRef::Op(j) => {
                let t = op_tensor[j];
                (t, tensors[t.0 as usize].rows, op_cols[j])
            }
            InputRef::Ext(e) => {
                let s = tensors[e];
                (TensorId(e as u32), s.rows, s.cols)
            }
        }
    };
    let whole = |t: TensorId, tensors: &[TensorShape]| -> TensorRegion {
        TensorRegion {
            tensor: t,
            region: tensors[t.0 as usize].whole(),
        }
    };

    for desc in &input.ops {
        let m = desc.m;
        // ⛔ OPERAND 0 IS OPTIONAL. The host-staged loads (`LoadRows`) take NO operands — their buffer is delivered by the
        // runtime — so resolving `inputs[0]` eagerly indexed an empty vec and
        // panicked before reaching the arm that handles them. The arms that read
        // these bindings all have an operand by their arity row, which is why
        // this is an `expect` naming the op and not a substituted zero.
        let in0 = desc
            .inputs
            .first()
            .map(|r| resolve(*r, &op_tensor, &op_cols, &tensors));
        let in0_named = |what: &str| -> (TensorId, u32, u32) {
            in0.unwrap_or_else(|| {
                panic!(
                    "{what} reads operand 0, but {:?} was lowered with no operands",
                    desc.op
                )
            })
        };
        // The registry decides which operand sets the output width, so
        // hand it a lookup rather than in0 alone.
        let out_cols = desc
            .op
            .out_cols(|k| resolve(desc.inputs[k], &op_tensor, &op_cols, &tensors).2);
        // ⭐ TRUE OUTPUT ROW COUNT — `desc.m` is the TOKEN count, and the two diverge
        // exactly once a Reshape view enters the chain. The bridge stamps every op's
        // `m` with the token count (it never rescales through a per-head view), so the
        // row scale lives in the TENSORS: a shape-preserving op INHERITS operand 0's
        // registered rows through the registry's `rows = [in0]` class, and Reshape —
        // the one op the front end states a scale for — rescales the token count by its
        // own payload. So the graph's tensor shapes and node regions state the RE-LAID
        // extents: a per-head chain `[m, heads·hd] → [m·heads, hd] → norm → back`
        // multiplies rows through the norm and divides them back at the flatten.
        // Under-stating them (registering everything at `m`) made the Reshape count law
        // fire on every per-head view AND reserved the re-laid buffer at a fraction of
        // its size.
        let out_rows: u32 = desc.op.out_rows(m, |k| {
            resolve(desc.inputs[k], &op_tensor, &op_cols, &tensors).1
        });
        let out_t = TensorId(tensors.len() as u32);
        tensors.push(TensorShape {
            rows: out_rows,
            cols: out_cols,
        });

        // The node's op: the source op with the witnesses only this walk knows bound.
        // RopeAppend's KvCacheLayout is keyed on its `K_cache` input (desc.inputs[4] per the
        // arity contract); AttnDecode's KvCacheLayout/KvCacheProducer are keyed on its
        // prefix-K input (desc.inputs[1] in the fused decode shape).
        let subop: SubOp<NeoX> = match desc.op {
            SubOp::MatmulTile { n, weight } => SubOp::MatmulTile { n, weight },
            SubOp::Sample { vocab, rows } => SubOp::Sample { vocab, rows },
            SubOp::EmbedRowGather { hidden, replaces } => {
                SubOp::EmbedRowGather { hidden, replaces }
            }
            SubOp::ChainStage {
                pad_vocab,
                table_rows,
                k,
            } => SubOp::ChainStage {
                pad_vocab,
                table_rows,
                k,
            },
            SubOp::SumReduce { minted }
            | SubOp::RmsNormReduce { minted, .. }
            | SubOp::RmsNormApply { minted, .. } => match minted {},
            // ⛔ A RESHAPE IS NOT AN ALIAS, and the obvious implementation of one is wrong: a
            // device tensor is STICK-laid-out and `sdsc_abstract::dev_off_stk` places element
            // (i, j) at
            //     (j / stk) * (a * stk) + i * stk + (j % stk)
            // where `a` is the ROW COUNT. Two tensors over the same bytes with different
            // (rows, cols) therefore disagree on where every element lives. gemma-4's reshapes
            // are per-head views ([M, heads*hd] -> [M*heads, hd]) — rows change, so an alias
            // moves everything.
            //
            // Nor does the arrangement authority catch it: `declare_arrangement` keys on tensor
            // NAME, and an alias gives the two views two names, so no conflict is seen.
            //
            // A reshape needs a RESTICKIFY — a real copy re-laying out from the old stick
            // layout to the new. Spyre already has that machinery; this pipeline just has no op
            // for it yet.
            SubOp::Reshape { rows, cols } => SubOp::Reshape { rows, cols },
            // A row-broadcast bias (`EwKind::BiasAdd`) is exact at decode (m = 1, both rows);
            // m > 1 broadcasts its `[1, n]` source across the rows.
            SubOp::Elementwise(kind) => SubOp::Elementwise(kind),
            SubOp::SiluMul => SubOp::SiluMul,
            SubOp::ScalarMul { scale } => SubOp::ScalarMul { scale },
            SubOp::RmsNorm { eps, gain } => SubOp::RmsNorm { eps, gain },
            SubOp::RopeRotate { head_dim, _form } => SubOp::RopeRotate { head_dim, _form },
            SubOp::RopeAppend {
                head_dim,
                layer,
                attn,
                pairing,
                _form,
                ..
            } => {
                // K-cache TensorId comes from desc.inputs[4]
                // (the validate arity-6 contract); for legacy
                // 4-input fixtures (K, cos, sin, V only) fall
                // back to the K input's own tensor — the layout
                // witness is consulted only when the producer/
                // consumer pair is end-to-end (RopeAppend +
                // AttnDecode), so the sentinel never escapes.
                let k_cache_t = if desc.inputs.len() > 4 {
                    resolve(desc.inputs[4], &op_tensor, &op_cols, &tensors).0
                } else {
                    in0_named("RopeAppend").0
                };
                // V-cache TensorId from desc.inputs[5] (arity-6
                // contract). Without it the layout aliases V=K
                // (`for_cache_tensor`) and the V-cache StoreAsync lands
                // in the K cache — clobbering the rotated K, leaving the
                // real V cache unwritten (caught by the full-tape interp
                // diff). Legacy 4-input fixtures fall back to the K
                // cache (sentinel; never escapes end-to-end).
                let v_cache_t = if desc.inputs.len() > 5 {
                    resolve(desc.inputs[5], &op_tensor, &op_cols, &tensors).0
                } else {
                    k_cache_t
                };
                // The per-model dims (num_kv_heads / head_dim) live
                // on the op itself — stamped from `model.bounds`
                // upstream — and flow into the byte-offset math via the
                // SubOp fields. No separate shape-table to check against.
                let layout = KvCacheLayout::for_cache_tensors(k_cache_t, v_cache_t);
                k_cache_producer_node.insert(k_cache_t, nodes.len() as u32);
                SubOp::RopeAppend {
                    head_dim,
                    layer,
                    attn,
                    pairing,
                    layout,
                    _form,
                }
            }
            SubOp::KvAppend {
                head_dim,
                layer,
                attn,
                ..
            } => {
                // `[K, V, K_cache, V_cache]` — the arity-4 contract.
                let k_cache_t = resolve(desc.inputs[2], &op_tensor, &op_cols, &tensors).0;
                let v_cache_t = resolve(desc.inputs[3], &op_tensor, &op_cols, &tensors).0;
                let layout = KvCacheLayout::for_cache_tensors(k_cache_t, v_cache_t);
                k_cache_producer_node.insert(k_cache_t, nodes.len() as u32);
                SubOp::KvAppend {
                    head_dim,
                    layer,
                    attn,
                    layout,
                }
            }
            SubOp::AttnDecode {
                geom,
                scale,
                valid_len,
                mask,
                ..
            } => {
                let (prefix_k_t, _, _) = resolve(desc.inputs[1], &op_tensor, &op_cols, &tensors);
                // prefix_v (inputs[2]) is a DISTINCT cache tensor from
                // prefix_k (inputs[1]). `for_cache_tensor(t)` aliases
                // V=K — using it here made the lowering's V load read
                // the K cache (attn output == K, caught by the
                // full-tape interp diff). Inputs = [q, prefix_k,
                // prefix_v, new_k, new_v].
                let (prefix_v_t, _, _) = resolve(desc.inputs[2], &op_tensor, &op_cols, &tensors);
                // The head geometry rides on the AttnDecode op (minted from `model.bounds`);
                // the layout witness is just the cache-tensor identity pair.
                let layout = KvCacheLayout::for_cache_tensors(prefix_k_t, prefix_v_t);
                let producer = match k_cache_producer_node.get(&prefix_k_t) {
                    Some(&node_idx) => KvCacheProducer::from_rope_append(node_idx),
                    None => KvCacheProducer::pre_populated_ext(),
                };
                let softmax_state = SoftmaxStateId::new(next_softmax_state);
                next_softmax_state += 1;
                SubOp::AttnDecode {
                    geom,
                    scale,
                    valid_len,
                    layout,
                    producer,
                    softmax_state,
                    mask,
                }
            }
            SubOp::TanhSoftCap { cap } => SubOp::TanhSoftCap { cap },
            SubOp::RmsNormUnit { eps } => SubOp::RmsNormUnit { eps },
            SubOp::ScalarWeightMul => SubOp::ScalarWeightMul,
            SubOp::GateSplit { half_cols } => SubOp::GateSplit { half_cols },
            SubOp::GateApply => SubOp::GateApply,
            SubOp::Concat { cols } => SubOp::Concat { cols },
            SubOp::GateScale => SubOp::GateScale,
            SubOp::LoadRows { source, width } => SubOp::LoadRows { source, width },
            SubOp::EmbeddingGather { indices_kind } => SubOp::EmbeddingGather { indices_kind },
            SubOp::VisionRope => SubOp::VisionRope,
            SubOp::VarlenAttention { cu_kind } => SubOp::VarlenAttention { cu_kind },
            SubOp::EncoderAttn { geom, scale } => SubOp::EncoderAttn { geom, scale },
            SubOp::GatedDeltaNet => SubOp::GatedDeltaNet,
            SubOp::Mamba2 { inner } => SubOp::Mamba2 { inner },
            SubOp::Mean => SubOp::Mean,
            SubOp::AllReduce => SubOp::AllReduce,
            SubOp::AllGather { cols } => SubOp::AllGather { cols },
            SubOp::RouterNorm { eps, router } => SubOp::RouterNorm { eps, router },
            SubOp::RouterLogits { experts, router } => SubOp::RouterLogits { experts, router },
            SubOp::RouteSoftmax => SubOp::RouteSoftmax,
            SubOp::RouteSigmoidBias { router } => SubOp::RouteSigmoidBias { router },
            SubOp::RouteBias { router } => SubOp::RouteBias { router },
            SubOp::RouteArgsort => SubOp::RouteArgsort,
            SubOp::RouteTopK { k } => SubOp::RouteTopK { k },
            SubOp::RouteGatherScores => SubOp::RouteGatherScores,
            SubOp::RouteScale { scale } => SubOp::RouteScale { scale },
            SubOp::RouteRenorm => SubOp::RouteRenorm,
            SubOp::RouteExpertScale { router } => SubOp::RouteExpertScale { router },
            SubOp::ExpertSort { experts, k, bundle } => SubOp::ExpertSort { experts, k, bundle },
            SubOp::ExpertMatmul {
                proj,
                n,
                k,
                quant,
                bundle,
            } => SubOp::ExpertMatmul {
                proj,
                n,
                k,
                quant,
                bundle,
            },
            SubOp::ExpertGatedAct { act } => SubOp::ExpertGatedAct { act },
            SubOp::ExpertAct { act } => SubOp::ExpertAct { act },
            SubOp::ExpertUnsort => SubOp::ExpertUnsort,
            SubOp::ExpertCombine { hidden, shared } => SubOp::ExpertCombine { hidden, shared },
            SubOp::KvEncode { operand } => SubOp::KvEncode { operand },
            SubOp::KvStage { operand } => SubOp::KvStage { operand },
            SubOp::RotateRows { rows } => SubOp::RotateRows { rows },
            SubOp::AttnPackedKv => SubOp::AttnPackedKv,
            SubOp::SampleRowsGather => SubOp::SampleRowsGather,
            SubOp::SampleRowsScatter => SubOp::SampleRowsScatter,
            SubOp::AllRowsMatmul => SubOp::AllRowsMatmul,
        };

        if let SubOp::MatmulTile { n, weight } = subop {
            // k is structurally derived from the activation's
            // column count — it is not a separate field. See
            // SubOp::MatmulTile doc.
            let (in0_t, _r, in0_cols) = in0_named("MatmulTile");
            let k = in0_cols;
            let (w_t, _wr, _wc) = resolve(desc.inputs[1], &op_tensor, &op_cols, &tensors);
            // The activation's OWN registered rows — not the token count: a matmul
            // consuming a per-head view computes `m·heads` output rows.
            let act_rows = tensors[in0_t.0 as usize].rows;
            let act = TensorRegion {
                tensor: in0_t,
                region: Region {
                    rows: Range::new(0, act_rows),
                    cols: Range::new(0, k),
                },
            };
            // Gemms are emitted WHOLE: one MatmulTile per logical Linear
            // with the FULL [M, N] output. N+K tiling is owned by
            // `split_oversized_loads_pass` (the NK path), which strides
            // the B operand by the SOURCE matrix's full N — so a
            // column-slice of a row-major [K, N] weight can never be
            // mis-strided (the per-N-block-at-lowering bug the host
            // interpreter caught: the K-only path used the slice width
            // 128 as the row stride instead of N). `nb` still tiles
            // rmsnorm / rope / elementwise below.
            let whole = Range::new(0, n);
            let id = SubtileId(nodes.len() as u32);
            let mut mm_inputs = vec![
                act,
                // W is row-major [K, N] per FUF `gemm(x: [..., K],
                // w: [K, N])`; read all K rows of the full N.
                TensorRegion {
                    tensor: w_t,
                    region: Region {
                        rows: Range::new(0, k),
                        cols: whole,
                    },
                },
            ];
            if matches!(weight, crate::lower::GemmWeight::Fp8Dynamic) {
                // fp8 W8A8: carry the PER-CHANNEL weight_scale (desc.inputs[2], shape [N,1]) as the
                // MatmulTile's THIRD input, so `lower_matmul_node` sees arity-3 == fp8 and dequants via
                // `Fp8W8A8Dequant` (w_scale[n]). Dense matmul stays arity-2 [act, W] (byte-identical).
                let (ws_t, ws_r, ws_c) = resolve(desc.inputs[2], &op_tensor, &op_cols, &tensors);
                mm_inputs.push(TensorRegion {
                    tensor: ws_t,
                    region: Region {
                        rows: Range::new(0, ws_r),
                        cols: Range::new(0, ws_c),
                    },
                });
            }
            nodes.push(SubtileNode {
                id,
                op: subop,
                inputs: mm_inputs,
                output: TensorRegion {
                    tensor: out_t,
                    region: Region {
                        rows: Range::new(0, act_rows),
                        cols: whole,
                    },
                },
            });
        } else {
            // A pure elementwise op (silu/mul/add/silu·mul) is tiled by the
            // output column slice like the GEMM N-blocks, so a downstream
            // per-target lowering can dispatch the tiles independently.
            // rope / attn (head structure) and
            // rmsnorm (RMS reduction) stay WHOLE here — this is the bit-exact,
            // GPU-correct reference lowering + the live per-op-schedule path.
            // The head-tiling, split-K all-reduce and replication of the
            // tensor-parallel partition live in
            // [`crate::partition::lower_partitioned`], kept separate because
            // split-K reassociates (breaks this fn's bit-exact contract) and
            // because the partition needs the new GPU emit/player arms.
            let elementwise = matches!(
                subop,
                SubOp::Elementwise(EwKind::Silu | EwKind::Mul | EwKind::Add)
                    | SubOp::ScalarMul { .. }
                    | SubOp::SiluMul
            );
            let blocks = if elementwise {
                n_blocks(out_cols, nb)
            } else {
                vec![Range::new(0, out_cols)]
            };
            for blk in blocks {
                let mut inputs: Vec<TensorRegion> = desc
                    .inputs
                    .iter()
                    .map(|r| {
                        let (t, _, _) = resolve(*r, &op_tensor, &op_cols, &tensors);
                        if elementwise {
                            TensorRegion {
                                tensor: t,
                                region: Region {
                                    // The op's OWN registered rows — shape-preserving, so
                                    // every operand is read at the output's row count (the
                                    // token count rescaled by any upstream per-head view).
                                    rows: Range::new(0, out_rows),
                                    cols: blk,
                                },
                            }
                        } else {
                            whole(t, &tensors)
                        }
                    })
                    .collect();
                if let SubOp::AttnDecode {
                    producer,
                    valid_len,
                    ..
                } = &subop
                {
                    // Full-cache decode model (cache written by a RopeAppend THIS
                    // forward): prefix_k/prefix_v (inputs 1,2) are the full KV cache
                    // TENSOR [prefix(rows 0..dp) ++ new(row dp) ++ uninitialized
                    // capacity]. eval_node's prefix SEGMENT must be EXACTLY the
                    // `valid_len - 1` real prefix rows (`= decode_position`), and it
                    // reads the new row via the separate new_k/new_v segments — so
                    // it attends to `valid_len` positions TOTAL.
                    //
                    // The slice bound is `valid_len - 1`, NOT `cache_rows - 1`:
                    // tying it to the cache tensor's row count is the rigging the
                    // un-rigged honest test exposes (a cache sized [capacity, kv]
                    // with capacity > valid_len would otherwise drag uninitialized
                    // rows into the softmax). `valid_len` is the structural
                    // host-verification pin (= decode_position + 1), decoupled from
                    // capacity. A PrePopulatedExt cache is a READ-ONLY prefix
                    // (new_k a separate segment, no append) — NOT sliced (e.g. the
                    // RopeRotate-based bit-exact fixture, where prefix_k holds all
                    // l prefix rows and the whole cache is valid).
                    if matches!(producer, KvCacheProducer::SameForwardRopeAppend { .. }) {
                        let prefix_rows = (*valid_len).saturating_sub(1).max(1);
                        for idx in [1usize, 2usize] {
                            if let Some(r) = inputs.get_mut(idx) {
                                let rows = r.region.rows;
                                debug_assert!(
                                    prefix_rows <= rows.len,
                                    "AttnDecode valid_len-1 prefix ({prefix_rows}) exceeds \
                                     cache capacity ({}) — valid_len must be <= cache rows",
                                    rows.len,
                                );
                                r.region.rows = Range::new(rows.start, prefix_rows);
                            }
                        }
                    }
                }
                let id = SubtileId(nodes.len() as u32);
                nodes.push(SubtileNode {
                    id,
                    op: subop,
                    inputs,
                    output: TensorRegion {
                        tensor: out_t,
                        region: Region {
                            rows: Range::new(0, out_rows),
                            cols: blk,
                        },
                    },
                });
            }
        }
        op_tensor.push(out_t);
        op_cols.push(out_cols);
    }

    SubtileIR {
        tensors,
        num_sources,
        nodes,
        result: op_tensor[input.result],
        op_output: op_tensor,
    }
}

/// Rewrite a
/// `SubtileIR` so every `SubOp::RmsNorm` becomes
/// `SubOp::RmsNormReduce` (1 node, output `[m, 1]`) plus
/// `n_blocks(hidden, nb)` `SubOp::RmsNormApply` nodes (one per chunk
/// of the original output's `cols`).
///
/// Operates on a built `SubtileIR<F>` rather than inside
/// [`lower_region`] because the Metal `mega::serialize` backend
/// shares `lower_region` with the CUDA pipeline and only the CUDA
/// path's `lower_subtile_tape_to_tk_tape` knows the new SubOps. The
/// CUDA codegen probe runs `lower_region` then `decompose_rmsnorm`;
/// the Metal path leaves it whole.
///
/// Tensor renumbering: the inv_rms tensor for each replaced RmsNorm
/// is appended at the end of `tensors`, so existing `TensorId`
/// references in subsequent nodes are preserved verbatim.
///
/// Node renumbering: every replaced RmsNorm grows by `n_apply` nodes;
/// downstream nodes' positions shift accordingly. The only embedded
/// `SubtileId` reference outside the linear ordering is
/// [`KvCacheProducer::SameForwardRopeAppend::producer_node_idx`] on
/// `SubOp::AttnDecode`; this pass remaps it through `new_id_for_old`.
pub fn decompose_rmsnorm<F: RopeForm>(
    graph: &SubtileIR<F>,
    nb: std::num::NonZeroU32,
) -> SubtileIR<F> {
    // Pass 1: build new_id_for_old[i] = the new SubtileId that the
    // node at old position i maps to. For RmsNorm, this is the id of
    // the reduce node (the apply nodes follow immediately after).
    let mut new_id_for_old: Vec<u32> = Vec::with_capacity(graph.nodes.len());
    let mut next_new_id: u32 = 0;
    for node in &graph.nodes {
        new_id_for_old.push(next_new_id);
        if matches!(node.op, SubOp::RmsNorm { .. }) {
            let hidden = node.output.region.cols.len;
            let n_apply = n_blocks(hidden, nb).len() as u32;
            next_new_id += 1 + n_apply;
        } else {
            next_new_id += 1;
        }
    }

    // Pass 2: emit the rewritten nodes.
    let mut tensors = graph.tensors.clone();
    let mut nodes: Vec<SubtileNode<F>> = Vec::with_capacity(next_new_id as usize);
    for node in &graph.nodes {
        match node.op {
            SubOp::RmsNorm { eps, gain } => {
                let x_in = node.inputs[0];
                let gamma_in = node.inputs[1];
                let y_out = node.output;
                let m = y_out.region.rows.len;
                let hidden = y_out.region.cols.len;

                let inv_rms_t = TensorId(tensors.len() as u32);
                tensors.push(TensorShape { rows: m, cols: 1 });

                // 1 RmsNormReduce: reads x (whole), writes inv_rms (whole).
                nodes.push(SubtileNode {
                    id: SubtileId(nodes.len() as u32),
                    op: SubOp::RmsNormReduce {
                        eps,
                        minted: Minted::BY_LOWERING,
                    },
                    inputs: vec![x_in],
                    output: TensorRegion {
                        tensor: inv_rms_t,
                        region: Region {
                            rows: Range::new(0, m),
                            cols: Range::new(0, 1),
                        },
                    },
                });

                // N RmsNormApply: chunked along cols.
                let x_col_start = x_in.region.cols.start;
                let gamma_col_start = gamma_in.region.cols.start;
                let y_col_start = y_out.region.cols.start;
                for blk in n_blocks(hidden, nb) {
                    nodes.push(SubtileNode {
                        id: SubtileId(nodes.len() as u32),
                        // The convention rides the decomposition: the apply phase is
                        // where the gain multiplies, so it is the phase that has to know.
                        op: SubOp::RmsNormApply {
                            gain,
                            minted: Minted::BY_LOWERING,
                        },
                        inputs: vec![
                            TensorRegion {
                                tensor: x_in.tensor,
                                region: Region {
                                    rows: x_in.region.rows,
                                    cols: Range::new(x_col_start + blk.start, blk.len),
                                },
                            },
                            TensorRegion {
                                tensor: inv_rms_t,
                                region: Region {
                                    rows: Range::new(0, m),
                                    cols: Range::new(0, 1),
                                },
                            },
                            TensorRegion {
                                tensor: gamma_in.tensor,
                                region: Region {
                                    rows: gamma_in.region.rows,
                                    cols: Range::new(gamma_col_start + blk.start, blk.len),
                                },
                            },
                        ],
                        output: TensorRegion {
                            tensor: y_out.tensor,
                            region: Region {
                                rows: y_out.region.rows,
                                cols: Range::new(y_col_start + blk.start, blk.len),
                            },
                        },
                    });
                }
            }
            SubOp::AttnDecode {
                geom,
                scale,
                valid_len,
                layout,
                producer,
                softmax_state,
                mask,
            } => {
                // Remap the KvCacheProducer's stored producer node id
                // so an upstream RopeAppend's new SubtileId stays
                // referenceable after RmsNorm decomposition shifts ids.
                let new_producer = match producer {
                    KvCacheProducer::SameForwardRopeAppend {
                        producer_node_idx, ..
                    } => KvCacheProducer::from_rope_append(
                        new_id_for_old[producer_node_idx as usize],
                    ),
                    KvCacheProducer::PrePopulatedExt { .. } => producer,
                };
                nodes.push(SubtileNode {
                    id: SubtileId(nodes.len() as u32),
                    op: SubOp::AttnDecode {
                        geom,
                        scale,
                        valid_len,
                        layout,
                        mask,
                        producer: new_producer,
                        softmax_state,
                    },
                    inputs: node.inputs.clone(),
                    output: node.output,
                });
            }
            _ => {
                nodes.push(SubtileNode {
                    id: SubtileId(nodes.len() as u32),
                    op: node.op,
                    inputs: node.inputs.clone(),
                    output: node.output,
                });
            }
        }
    }

    SubtileIR {
        tensors,
        num_sources: graph.num_sources,
        nodes,
        result: graph.result,
        // The rewrite adds nodes; it never retires an op's output tensor.
        op_output: graph.op_output.clone(),
    }
}

/// Head-tile every
/// `SubOp::RopeRotate` and `SubOp::RopeAppend` node into
/// `head_blocks(total_cols, nb, head_dim)` head-aligned chunks.
///
/// Each emitted chunk reads a head-aligned slice of the rope's
/// activation input and writes the matching slice of the output
/// tensor. cos/sin and (for RopeAppend) cache tensors stay whole —
/// every chunk reads the same cos/sin row vec; cache writes are
/// indexed by `layout.layer × position` not by head, and per-block
/// region narrowing is handled at the lower_compute layer (audit
/// finding #18 — Patch 2 territory).
///
/// Chunk size: `heads_per_block = nb / head_dim` (clamped to ≥1 by
/// [`head_blocks`]). At `nb = 128` and `head_dim = 64` this is
/// 2 heads per block — exactly the substrate page width — so each
/// rope chunk consumes one Linear N-block 1:1 (no multi-writer
/// downstream).
///
/// `SubOp::AttnDecode` is NOT head-tiled by this pass — Patch 2 of
/// the handoff redesigns its KvCachePageShape, which subsumes head
/// tiling. AttnDecode's `KvCacheProducer::SameForwardRopeAppend`
/// node-id ref is remapped through `new_id_for_old` (points to the
/// FIRST emitted RopeAppend chunk; the remaining chunks are
/// barrier-synced via predecessors() region overlap).
pub fn head_tile_rope<F: RopeForm>(graph: &SubtileIR<F>, nb: std::num::NonZeroU32) -> SubtileIR<F> {
    // Pass 1: build new_id_for_old — each rope node grows by N
    // (= head_blocks().len()) chunks; everything else maps 1:1.
    let mut new_id_for_old: Vec<u32> = Vec::with_capacity(graph.nodes.len());
    let mut next_new_id: u32 = 0;
    for node in &graph.nodes {
        new_id_for_old.push(next_new_id);
        match node.op {
            SubOp::RopeRotate { head_dim, .. } | SubOp::RopeAppend { head_dim, .. } => {
                let total_cols = node.output.region.cols.len;
                let head_dim_nz = std::num::NonZeroU32::new(head_dim.get())
                    .expect("head_tile_rope: head_dim must be non-zero");
                next_new_id += head_blocks(total_cols, nb, head_dim_nz).len() as u32;
            }
            _ => next_new_id += 1,
        }
    }

    // Pass 2: emit the rewritten nodes.
    let tensors = graph.tensors.clone();
    let mut nodes: Vec<SubtileNode<F>> = Vec::with_capacity(next_new_id as usize);
    for node in &graph.nodes {
        match node.op {
            SubOp::RopeRotate { head_dim, _form: _ } => {
                let q_in = node.inputs[0];
                let cos_in = node.inputs[1];
                let sin_in = node.inputs[2];
                let total_cols = node.output.region.cols.len;
                let head_dim_nz = std::num::NonZeroU32::new(head_dim.get())
                    .expect("head_tile_rope: head_dim must be non-zero");
                let q_col_start = q_in.region.cols.start;
                let out_col_start = node.output.region.cols.start;
                for blk in head_blocks(total_cols, nb, head_dim_nz) {
                    nodes.push(SubtileNode {
                        id: SubtileId(nodes.len() as u32),
                        op: SubOp::RopeRotate {
                            head_dim,
                            _form: PhantomData,
                        },
                        inputs: vec![
                            TensorRegion {
                                tensor: q_in.tensor,
                                region: Region {
                                    rows: q_in.region.rows,
                                    cols: Range::new(q_col_start + blk.start, blk.len),
                                },
                            },
                            cos_in,
                            sin_in,
                        ],
                        output: TensorRegion {
                            tensor: node.output.tensor,
                            region: Region {
                                rows: node.output.region.rows,
                                cols: Range::new(out_col_start + blk.start, blk.len),
                            },
                        },
                    });
                }
            }
            SubOp::RopeAppend {
                head_dim,
                layer,
                attn,
                pairing,
                layout,
                _form: _,
            } => {
                // Arity 6: [K, cos, sin, V, K_cache, V_cache]. K and V
                // are head-tiled; cos/sin and the cache handles stay
                // whole.
                let k_in = node.inputs[0];
                let cos_in = node.inputs[1];
                let sin_in = node.inputs[2];
                let v_in = node.inputs[3];
                let kcache_in = node.inputs[4];
                let vcache_in = node.inputs[5];
                let total_cols = node.output.region.cols.len;
                let head_dim_nz = std::num::NonZeroU32::new(head_dim.get())
                    .expect("head_tile_rope: head_dim must be non-zero");
                let k_col_start = k_in.region.cols.start;
                let v_col_start = v_in.region.cols.start;
                let out_col_start = node.output.region.cols.start;
                for blk in head_blocks(total_cols, nb, head_dim_nz) {
                    nodes.push(SubtileNode {
                        id: SubtileId(nodes.len() as u32),
                        op: SubOp::RopeAppend {
                            head_dim,
                            layer,
                            attn,
                            pairing,
                            layout,
                            _form: PhantomData,
                        },
                        inputs: vec![
                            TensorRegion {
                                tensor: k_in.tensor,
                                region: Region {
                                    rows: k_in.region.rows,
                                    cols: Range::new(k_col_start + blk.start, blk.len),
                                },
                            },
                            cos_in,
                            sin_in,
                            TensorRegion {
                                tensor: v_in.tensor,
                                region: Region {
                                    rows: v_in.region.rows,
                                    cols: Range::new(v_col_start + blk.start, blk.len),
                                },
                            },
                            kcache_in,
                            vcache_in,
                        ],
                        output: TensorRegion {
                            tensor: node.output.tensor,
                            region: Region {
                                rows: node.output.region.rows,
                                cols: Range::new(out_col_start + blk.start, blk.len),
                            },
                        },
                    });
                }
            }
            SubOp::AttnDecode {
                geom,
                scale,
                valid_len,
                layout,
                producer,
                softmax_state,
                mask,
            } => {
                // AttnDecode stays whole — head tiling for it rides on
                // a KvCachePageShape redesign that has not landed.
                // Remap KvCacheProducer's stored producer node id
                // through new_id_for_old (points to the first
                // emitted RopeAppend chunk).
                let new_producer = match producer {
                    KvCacheProducer::SameForwardRopeAppend {
                        producer_node_idx, ..
                    } => KvCacheProducer::from_rope_append(
                        new_id_for_old[producer_node_idx as usize],
                    ),
                    KvCacheProducer::PrePopulatedExt { .. } => producer,
                };
                nodes.push(SubtileNode {
                    id: SubtileId(nodes.len() as u32),
                    op: SubOp::AttnDecode {
                        geom,
                        scale,
                        valid_len,
                        layout,
                        mask,
                        producer: new_producer,
                        softmax_state,
                    },
                    inputs: node.inputs.clone(),
                    output: node.output,
                });
            }
            _ => {
                nodes.push(SubtileNode {
                    id: SubtileId(nodes.len() as u32),
                    op: node.op,
                    inputs: node.inputs.clone(),
                    output: node.output,
                });
            }
        }
    }

    SubtileIR {
        tensors,
        num_sources: graph.num_sources,
        nodes,
        result: graph.result,
        // The rewrite adds nodes; it never retires an op's output tensor.
        op_output: graph.op_output.clone(),
    }
}

// ── Tests (canonical region IR) ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The LOGICAL half of Reshape's contract: it preserves the element
    /// sequence and changes only the extents. `eval_dag` is the oracle the
    /// pipeline already uses for numerics, so the identity is asserted against
    /// it rather than argued.
    ///
    /// The PHYSICAL half — that this identity does NOT make a placement alias
    /// sound, because the device layout is a function of the row count — is
    /// pinned separately by
    /// `sdsc_abstract::tests::reshape_is_not_a_placement_alias_under_stick_layout`.
    /// Both halves matter: the logical identity is exactly what tempts you into
    /// the physically-wrong alias.
    #[test]
    fn reshape_preserves_the_element_sequence() {
        // gemma-4-shaped per-head view: [2, 4*3] -> [2*4, 3].
        let (m, heads, hd) = (2usize, 4usize, 3usize);
        let src: Vec<f32> = (0..(m * heads * hd)).map(|i| i as f32 * 0.5).collect();

        let tensors = vec![
            TensorShape {
                rows: m as u32,
                cols: (heads * hd) as u32,
            },
            TensorShape {
                rows: (m * heads) as u32,
                cols: hd as u32,
            },
        ];
        let out_t = TensorId::from_index(1);
        let g = SubtileIR::<NeoX> {
            tensors,
            num_sources: 1,
            nodes: vec![SubtileNode {
                id: SubtileId::from_index(0),
                op: SubOp::Reshape {
                    rows: RowScale::Times(std::num::NonZeroU32::new(heads as u32).unwrap()),
                    cols: hd as u32,
                },
                inputs: vec![TensorRegion {
                    tensor: TensorId::from_index(0),
                    region: Region {
                        rows: Range::new(0, m as u32),
                        cols: Range::new(0, (heads * hd) as u32),
                    },
                }],
                output: TensorRegion {
                    tensor: out_t,
                    region: Region {
                        rows: Range::new(0, (m * heads) as u32),
                        cols: Range::new(0, hd as u32),
                    },
                },
            }],
            result: out_t,
            // Hand-authored fixture: there is no source op list to be the
            // provenance of, so the map is empty.
            op_output: Vec::new(),
        };
        assert!(validate(&g).is_ok(), "a reshape graph must validate");
        let bufs = eval_dag(&g, &[&src]);
        assert_eq!(
            result_buffer(&g, &bufs),
            &src[..],
            "reshape is the identity on the element sequence"
        );
    }

    /// ⭐ THE ROW LAW, end to end through `lower_region`. `desc.m` is the TOKEN count and the
    /// bridge never rescales it through a per-head view, so the row scale must ride the
    /// TENSORS: the split Reshape states `Times(heads)`, every shape-preserving consumer
    /// INHERITS operand 0's registered rows, and the flatten-back states `Times(1)` —
    /// registering everything at the token count instead understates the per-head domain
    /// (the reshape count law then fires on the flatten-back, and the buffer is reserved at
    /// a fraction of its size).
    #[test]
    fn a_per_head_chain_registers_its_re_laid_rows() {
        let nz = |k| std::num::NonZeroU32::new(k).unwrap();
        let (heads, hd) = (2u32, 64u32);
        let input = crate::lower::LoweringInput {
            exported: None,
            sources: vec![
                SourceShape { rows: 1, cols: 128 },
                SourceShape {
                    rows: 128,
                    cols: 128,
                },
                SourceShape { rows: 1, cols: 64 },
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::MatmulTile {
                        n: 128,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m: 1,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: SubOp::Reshape {
                        rows: RowScale::Times(nz(heads)),
                        cols: hd,
                    },
                    m: 1,
                    inputs: vec![InputRef::Op(0)],
                },
                OpDesc {
                    op: SubOp::RmsNormUnit { eps: 1e-6 },
                    m: 1,
                    inputs: vec![InputRef::Op(1)],
                },
                OpDesc {
                    op: SubOp::Reshape {
                        rows: RowScale::Times(nz(1)),
                        cols: heads * hd,
                    },
                    m: 1,
                    inputs: vec![InputRef::Op(2)],
                },
            ],
            result: 3,
        };
        let g = lower_region(&input, nz(64));
        assert!(validate(&g).is_ok(), "a per-head chain must validate");
        // The projections stay at the token count; the per-head view and its norm register
        // the RE-LAID rows; the flatten-back divides them back.
        let shape = |i: usize| (g.tensors[i].rows, g.tensors[i].cols);
        assert_eq!(shape(3), (1, 128), "the projection is [m, heads·hd]");
        assert_eq!(shape(4), (heads, hd), "the view is [m·heads, hd]");
        assert_eq!(shape(5), (heads, hd), "the norm inherits the view's rows");
        assert_eq!(
            shape(6),
            (1, heads * hd),
            "the flatten-back restores the flat row"
        );
        // And the node regions state the same extents — the regions are what every target's
        // lowering reads.
        let rows_of = |n: usize| g.nodes[n].output.region.rows.len;
        assert_eq!(
            (rows_of(0), rows_of(1), rows_of(2), rows_of(3)),
            (1, heads, heads, 1)
        );
    }

    /// The OTHER direction of the row law: the vision patch merger's `Over` view divides the
    /// token count, and the ops after it inherit the DIVIDED rows.
    #[test]
    fn a_merger_view_registers_its_divided_rows() {
        let nz = |k| std::num::NonZeroU32::new(k).unwrap();
        let input = crate::lower::LoweringInput {
            exported: None,
            sources: vec![SourceShape { rows: 4, cols: 64 }],
            ops: vec![
                OpDesc {
                    op: SubOp::Reshape {
                        rows: RowScale::Over(nz(2)),
                        cols: 128,
                    },
                    m: 4,
                    inputs: vec![InputRef::Ext(0)],
                },
                OpDesc {
                    op: SubOp::RmsNormUnit { eps: 1e-6 },
                    m: 4,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            result: 1,
        };
        let g = lower_region(&input, nz(64));
        assert!(validate(&g).is_ok(), "a merger chain must validate");
        assert_eq!((g.tensors[1].rows, g.tensors[1].cols), (2, 128));
        assert_eq!(
            (g.tensors[2].rows, g.tensors[2].cols),
            (2, 128),
            "the op after the merger inherits the DIVIDED rows, not the token count"
        );
    }
    use crate::lower::{InputRef, OpDesc};
    use crate::model_geometry::{KvHeads, QueryHeads};

    /// Patch 1 producer-side milestone + fixture-orientation guard.
    ///
    /// Drives the SubtileIR half of `codegen.rs`'s wavefront-cuda-probe
    /// over the production-shaped `one_layer_input` fixture at `nb = 128`
    /// (the value step (f) will flip `codegen.rs` to): `lower_region` →
    /// `decompose_rmsnorm` → `head_tile_rope` → `ValidatedGraph::new`
    /// (runs `validate`) → `lower_dag_to_tape` (asserts
    /// `validate_subtile_tape` at its exit). All four must pass.
    ///
    /// This is the guard for the fixture-weight-orientation fix: the
    /// MLP weights must be `[K, N]` (the `MatmulTile` / `eval_node`
    /// convention), not HF `[out, in] = [N, K]`. With the HF orientation
    /// the N-tiled gate/up/down `MatmulTile` blocks slice past the
    /// weight cols extent and `validate` rejects the graph (would have
    /// caught the latent fixture bug at any tiling).
    ///
    /// The TkTape consumer lift — `lower_subtile_tape_to_tk_tape`
    /// arms consuming the per-K-chunk predecessor pages a tiled
    /// producer now emits — is the next frontier (Patch 1 step (f)).
    #[test]
    #[cfg(feature = "spyre")]
    fn subtile_tape_nb128_builds_from_fixture() {
        use crate::lower::fuse_silu_mul;
        use crate::subtile_tape::lower_dag_to_tape;
        use std::num::NonZeroU32;

        let fused = fuse_silu_mul(&crate::fixtures::one_layer_input());
        let nb = NonZeroU32::new(128).expect("128 != 0");
        let rg_whole = lower_region(&fused, nb);
        let rg_decomposed = decompose_rmsnorm(&rg_whole, nb);
        let rg = head_tile_rope(&rg_decomposed, nb);
        let valid = ValidatedGraph::new(&rg)
            .expect("SubtileIR validates at nb=128 (fixture MLP weights must be [K, N])");
        // lower_dag_to_tape asserts validate_subtile_tape at its exit;
        // a non-empty tape means the multi-writer ComputeInput plumbing
        // (Patch 1 step (a)) accepts the N-tiled producers.
        let tape = lower_dag_to_tape(&valid);
        assert!(
            !tape.instrs().is_empty(),
            "lower_dag_to_tape produced a non-empty SubtileTape at nb=128"
        );
    }

    /// A consumer reading a whole N-block-tiled output depends on EVERY
    /// block; a reader of a leaf source has no dependency.
    #[test]
    fn tiled_elementwise_depends_on_matching_block() {
        let (m, n, k) = (1u32, 6u32, 8u32);
        let input = crate::lower::LoweringInput {
            exported: None,
            sources: vec![
                SourceShape { rows: m, cols: k },
                SourceShape { rows: n, cols: k },
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::MatmulTile {
                        n,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: SubOp::Elementwise(EwKind::Silu),
                    m,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            result: 1,
        };
        let g = lower_region(&input, std::num::NonZeroU32::new(2).unwrap());
        let preds = predecessors(&g);
        // 1 whole matmul (0) + 3 silu tiles (1,2,3). Gemms are emitted
        // whole, so each silu tile reads its N-block of the single gemm
        // output → all depend on the one matmul node.
        assert_eq!(g.nodes.len(), 4);
        assert!(preds[0].is_empty(), "matmul reads only sources");
        assert_eq!(preds[1], vec![SubtileId(0)], "silu tile 0 ← whole gemm");
        assert_eq!(preds[2], vec![SubtileId(0)], "silu tile 1 ← whole gemm");
        assert_eq!(preds[3], vec![SubtileId(0)], "silu tile 2 ← whole gemm");
    }

    /// SiluMul (binary input) with both inputs N-tiled at the same `nb`:
    /// SiluMul tile k must depend on exactly the matching block of EACH
    /// upstream producer (gate-block-k AND up-block-k, no others).
    ///
    /// Pins the invariant that
    /// elementwise N-tiling produces aligned per-block predecessor
    /// edges across both input positions of a binary elementwise op.
    #[test]
    fn silu_mul_nblock_aligns_with_two_gemms() {
        let (m, k, n) = (1u32, 4u32, 6u32);
        let input = crate::lower::LoweringInput {
            exported: None,
            sources: vec![
                SourceShape { rows: m, cols: k }, // 0: x  [m, k]
                SourceShape { rows: k, cols: n }, // 1: Wg [k, n] (gate)
                SourceShape { rows: k, cols: n }, // 2: Wu [k, n] (up)
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::MatmulTile {
                        n,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)], // gate = x @ Wg
                },
                OpDesc {
                    op: SubOp::MatmulTile {
                        n,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(2)], // up   = x @ Wu
                },
                OpDesc {
                    op: SubOp::SiluMul,
                    m,
                    inputs: vec![InputRef::Op(0), InputRef::Op(1)], // SiluMul(gate, up)
                },
            ],
            result: 2,
        };
        let g = lower_region(&input, std::num::NonZeroU32::new(2).unwrap());
        let preds = predecessors(&g);
        // 1 whole gate matmul (0) + 1 whole up matmul (1) + 3 silu_mul
        // tiles (2,3,4). Gemms are emitted whole, so each silu_mul tile
        // reads its N-block of the whole gate AND the whole up.
        assert_eq!(g.nodes.len(), 5, "expected 1+1+3 nodes");
        for k in 0..3u32 {
            let silu_id = 2 + k as usize;
            let want = vec![SubtileId(0), SubtileId(1)];
            assert_eq!(
                preds[silu_id], want,
                "silu_mul tile {} must depend on the whole gate (0) + whole up (1)",
                k,
            );
        }
    }

    /// Chunked RmsNorm decomposition feeding a tiled Mul consumer:
    /// `decompose_rmsnorm` rewrites RmsNorm into 1 RmsNormReduce + N
    /// RmsNormApply (one per chunk). Each downstream Mul tile lines
    /// up with one Apply tile (single-writer). Reduce is the sole
    /// writer of `inv_rms_t`, read by every Apply.
    ///
    /// Pins the rewrite's
    /// node count + per-tile predecessor edges. lower_region itself
    /// is unchanged (Metal mega::serialize-compatible); the CUDA
    /// codegen probe applies decompose_rmsnorm before
    /// `lower_dag_to_tape`.
    #[test]
    fn decompose_rmsnorm_rewrites_into_reduce_plus_apply_chunks() {
        let (m, k) = (1u32, 6u32);
        let input = crate::lower::LoweringInput {
            exported: None,
            sources: vec![
                SourceShape { rows: m, cols: k }, // 0: x     [m, k]
                SourceShape { rows: 1, cols: k }, // 1: gamma [1, k]
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::RmsNorm {
                        eps: 1e-6,
                        gain: GainConvention::Scale,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: SubOp::Elementwise(EwKind::Mul),
                    m,
                    inputs: vec![InputRef::Op(0), InputRef::Op(0)], // self-mul keeps it elementwise
                },
            ],
            result: 1,
        };
        let nb = std::num::NonZeroU32::new(2).unwrap();
        let g_whole = lower_region(&input, nb);
        // Pre-decomposition: 1 whole RmsNorm + 3 Mul tiles.
        assert_eq!(g_whole.nodes.len(), 4, "pre-decompose: 1 rms + 3 mul");

        let g = decompose_rmsnorm(&g_whole, nb);
        let preds = predecessors(&g);
        // Post-decomposition: 1 RmsNormReduce (id 0) + 3 RmsNormApply
        // (ids 1,2,3) + 3 Mul tiles (ids 4,5,6). Total 7 nodes.
        assert_eq!(
            g.nodes.len(),
            7,
            "post-decompose: 1 reduce + 3 apply + 3 mul tiles"
        );
        assert!(matches!(g.nodes[0].op, SubOp::RmsNormReduce { .. }));
        for i in 1..4 {
            assert!(matches!(g.nodes[i].op, SubOp::RmsNormApply { .. }));
        }
        // Reduce reads only sources (x).
        assert!(preds[0].is_empty(), "RmsNormReduce reads only sources");
        // Each Apply reads the inv_rms tensor (whole) → single
        // predecessor = RmsNormReduce. (x and gamma are sources, so
        // they contribute no node preds.)
        for (apply, p) in preds.iter().enumerate().take(4).skip(1) {
            assert_eq!(
                *p,
                vec![SubtileId(0)],
                "RmsNormApply {apply} reads inv_rms from the sole reduce writer",
            );
        }
        // Each Mul tile k reads cols [k*2..(k+1)*2] of out_t — exactly
        // the region written by RmsNormApply tile k (id 1+k).
        for tile in 0..3 {
            assert_eq!(
                preds[4 + tile],
                vec![SubtileId(1 + tile as u32)],
                "Mul tile {} depends on RmsNormApply tile {}",
                tile,
                tile,
            );
        }
    }

    /// Head-tiling rewrites a whole RopeRotate into head-aligned
    /// chunks. With nb=4 and head_dim=2, heads_per_block=2 — so 4 q
    /// heads (8 cols) become 2 chunks of 2 heads each.
    ///
    /// Each chunk reads a head-aligned slice of the rope's
    /// activation input; cos/sin stay whole; output is the matching
    /// slice of the rope output tensor. KvCacheProducer remap is
    /// exercised by the AttnDecode test below; here the upstream is
    /// a Gemm and downstream is a leaf result.
    #[test]
    fn head_tile_rope_splits_rope_rotate_into_head_blocks() {
        let (m, hd, hq, h) = (1u32, 2u32, 4u32, 4u32);
        let qdim = hq * hd; // 8
        let input = crate::lower::LoweringInput {
            exported: None,
            sources: vec![
                SourceShape { rows: m, cols: h }, // 0: x  [m, h]
                SourceShape {
                    rows: h,
                    cols: qdim,
                }, // 1: Wq [h, qdim]
                SourceShape { rows: 1, cols: hd }, // 2: cos
                SourceShape { rows: 1, cols: hd }, // 3: sin
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::MatmulTile {
                        n: qdim,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: SubOp::rope_rotate(HeadDim::new(hd)),
                    m,
                    inputs: vec![InputRef::Op(0), InputRef::Ext(2), InputRef::Ext(3)],
                },
            ],
            result: 1,
        };
        // nb = 4, head_dim = 2 → heads_per_block = 2; qdim/hd_per_blk
        // = 4/2 = 2 rope chunks.
        let nb = std::num::NonZeroU32::new(4).unwrap();
        let g_whole = lower_region(&input, nb);
        // Pre: 1 whole Gemm + 1 whole RopeRotate.
        assert_eq!(g_whole.nodes.len(), 2, "pre: 1 gemm + 1 rope");
        assert!(matches!(g_whole.nodes[1].op, SubOp::RopeRotate { .. }));

        let g = head_tile_rope(&g_whole, nb);
        // Post: 1 whole Gemm + 2 RopeRotate chunks.
        assert_eq!(g.nodes.len(), 3, "post: 1 gemm + 2 rope chunks");
        assert!(matches!(g.nodes[1].op, SubOp::RopeRotate { .. }));
        assert!(matches!(g.nodes[2].op, SubOp::RopeRotate { .. }));
        // Chunk 0 reads cols [0..4]; chunk 1 reads cols [4..8].
        assert_eq!(g.nodes[1].inputs[0].region.cols.start, 0);
        assert_eq!(g.nodes[1].inputs[0].region.cols.len, 4);
        assert_eq!(g.nodes[2].inputs[0].region.cols.start, 4);
        assert_eq!(g.nodes[2].inputs[0].region.cols.len, 4);
        // Output regions match.
        assert_eq!(g.nodes[1].output.region.cols.start, 0);
        assert_eq!(g.nodes[1].output.region.cols.len, 4);
        assert_eq!(g.nodes[2].output.region.cols.start, 4);
        assert_eq!(g.nodes[2].output.region.cols.len, 4);

        // Each rope chunk reads its head-aligned cols from the single
        // whole Gemm output (gemms emit whole now).
        let preds = predecessors(&g);
        assert_eq!(preds[1], vec![SubtileId(0)], "rope chunk 0 ← whole gemm");
        assert_eq!(preds[2], vec![SubtileId(0)], "rope chunk 1 ← whole gemm");
    }

    /// AttnDecode stays whole through `head_tile_rope`, but its
    /// `KvCacheProducer::SameForwardRopeAppend::producer_node_idx` is
    /// remapped through `new_id_for_old` so it points to the FIRST
    /// emitted RopeAppend chunk (other chunks are barrier-synced via
    /// region overlap in `predecessors()`).
    #[test]
    fn head_tile_rope_remaps_attn_decode_kv_cache_producer() {
        // Smallest fixture that exercises a RopeAppend → AttnDecode
        // chain. Uses 1 KV head, head_dim 4.
        let (m, hq, hkv, hd, l) = (1u32, 1u32, 1u32, 4u32, 3u32);
        let kvdim = hkv * hd; // = qdim (hq * hd) = 4
        let scale = 0.5f32;
        let input = crate::lower::LoweringInput {
            exported: None,
            sources: vec![
                SourceShape { rows: m, cols: hd }, // 0: q   [m, qdim]
                SourceShape {
                    rows: m,
                    cols: kvdim,
                }, // 1: k_in [m, kvdim]
                SourceShape { rows: 1, cols: hd }, // 2: cos
                SourceShape { rows: 1, cols: hd }, // 3: sin
                SourceShape {
                    rows: m,
                    cols: kvdim,
                }, // 4: v_in
                SourceShape {
                    rows: l,
                    cols: kvdim,
                }, // 5: prefixK
                SourceShape {
                    rows: l,
                    cols: kvdim,
                }, // 6: prefixV
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::rope_append(
                        HeadDim::new(hd),
                        0,
                        AttnMask::Causal,
                        RopeFormTag::NeoX,
                    ),
                    m,
                    inputs: vec![
                        InputRef::Ext(1),
                        InputRef::Ext(2),
                        InputRef::Ext(3),
                        InputRef::Ext(4),
                        InputRef::Ext(5),
                        InputRef::Ext(6),
                    ],
                },
                OpDesc {
                    // l prefix rows (fully valid) ++ 1 new = l+1 valid positions.
                    op: SubOp::attn_decode(
                        ModelAttnGeometry::mint(
                            QueryHeads::new(hq),
                            KvHeads::new(hkv),
                            HeadDim::new(hd),
                        )
                        .expect("the fixture's kv heads divide its query heads"),
                        scale,
                        l + 1,
                        AttnMask::Causal,
                    ),
                    m,
                    inputs: vec![
                        InputRef::Ext(0),
                        InputRef::Ext(5),
                        InputRef::Ext(6),
                        InputRef::Op(0), // rope_append's K output
                        InputRef::Ext(0),
                    ],
                },
            ],
            result: 1,
        };
        // nb = head_dim (1 head per block) → 1 RopeAppend chunk
        // (since num_kv_heads=1). The remap is degenerate but
        // exercises the code path.
        let nb = std::num::NonZeroU32::new(hd).unwrap();
        let g_whole = lower_region(&input, nb);
        let g = head_tile_rope(&g_whole, nb);
        // Find the AttnDecode node and verify producer points to the
        // RopeAppend's new id (which equals new_id_for_old[0] = 0).
        let attn = g
            .nodes
            .iter()
            .find(|n| matches!(n.op, SubOp::AttnDecode { .. }))
            .expect("AttnDecode node");
        match attn.op {
            SubOp::AttnDecode { producer, .. } => match producer {
                KvCacheProducer::SameForwardRopeAppend {
                    producer_node_idx, ..
                } => assert_eq!(
                    producer_node_idx, 0,
                    "AttnDecode producer must point to remapped RopeAppend id 0"
                ),
                KvCacheProducer::PrePopulatedExt { .. } => {
                    panic!("expected SameForwardRopeAppend producer")
                }
            },
            _ => unreachable!(),
        }
    }

    /// The un-rotated writer is a KV writer like the rope's: `lower_region` binds its cache
    /// pair (`[K, V, K_cache, V_cache]` operands 2 and 3) and the attention reading that cache
    /// names it as its producer, over the SAME layout witness. It carries its layer, and its
    /// host eval is the identity on K.
    #[test]
    fn kv_append_is_the_producer_of_the_cache_its_attention_reads() {
        let (m, hq, hkv, hd, l, layer) = (1u32, 2u32, 1u32, 4u32, 3u32, 5u32);
        let (qdim, kvdim) = (hq * hd, hkv * hd);
        let source = |rows, cols| SourceShape { rows, cols };
        let input = crate::lower::LoweringInput {
            exported: None,
            sources: vec![
                source(m, qdim),  // 0: q
                source(m, kvdim), // 1: k
                source(m, kvdim), // 2: v
                source(l, kvdim), // 3: prefixK
                source(l, kvdim), // 4: prefixV
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::kv_append(HeadDim::new(hd), layer, AttnMask::Causal),
                    m,
                    inputs: vec![
                        InputRef::Ext(1),
                        InputRef::Ext(2),
                        InputRef::Ext(3),
                        InputRef::Ext(4),
                    ],
                },
                OpDesc {
                    op: SubOp::attn_decode(
                        ModelAttnGeometry::mint(
                            QueryHeads::new(hq),
                            KvHeads::new(hkv),
                            HeadDim::new(hd),
                        )
                        .expect("the fixture's kv heads divide its query heads"),
                        0.5,
                        l + 1,
                        AttnMask::Causal,
                    ),
                    m,
                    inputs: vec![
                        InputRef::Ext(0),
                        InputRef::Ext(3),
                        InputRef::Ext(4),
                        InputRef::Op(0),
                        InputRef::Ext(2),
                    ],
                },
            ],
            result: 1,
        };
        let g = lower_region(&input, std::num::NonZeroU32::new(hd).unwrap());
        let (writer, attn) = (&g.nodes[0], &g.nodes[1]);
        let SubOp::KvAppend {
            layout: written,
            layer: at,
            ..
        } = writer.op
        else {
            panic!("expected KvAppend, got {:?}", writer.op);
        };
        assert_eq!(at, layer);
        assert_eq!(writer.op.layer_index(), Some(layer));
        assert_eq!(
            (written.cache_tensor(), written.v_cache_tensor()),
            (TensorId::from_index(3), TensorId::from_index(4))
        );
        let SubOp::AttnDecode {
            layout: read,
            producer,
            ..
        } = attn.op
        else {
            panic!("expected AttnDecode, got {:?}", attn.op);
        };
        assert_eq!(
            read, written,
            "the attention reads the cache the writer writes"
        );
        assert_eq!(producer, KvCacheProducer::from_rope_append(0));
        assert_eq!(writer.op.name(), "KvAppend");

        let k: Vec<f32> = (0..kvdim).map(|i| i as f32 - 1.5).collect();
        let zeros = |n: u32| vec![0f32; n as usize];
        let (q, v, pk, pv) = (
            zeros(qdim),
            zeros(kvdim),
            zeros(l * kvdim),
            zeros(l * kvdim),
        );
        let bufs = eval_dag(&g, &[&q, &k, &v, &pk, &pv]);
        let out = writer.output.tensor.0 as usize;
        assert_eq!(bufs[out], k, "the writer's K output is its K operand");
    }

    #[test]
    fn validate_rejects_uncovered_read() {
        let tensors = vec![
            TensorShape { rows: 1, cols: 4 }, // 0 source
            TensorShape { rows: 1, cols: 4 }, // 1 op output (never written)
        ];
        let bad: SubtileIR = SubtileIR {
            tensors,
            num_sources: 1,
            nodes: vec![SubtileNode {
                id: SubtileId(0),
                op: SubOp::Elementwise(EwKind::Silu),
                inputs: vec![TensorRegion {
                    tensor: TensorId(1), // reads op-output 1, which no prior node wrote
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                }],
                output: TensorRegion {
                    tensor: TensorId(1),
                    region: Region {
                        rows: Range::new(0, 1),
                        cols: Range::new(0, 4),
                    },
                },
            }],
            result: TensorId(1),
            // Hand-authored fixture: there is no source op list to be the
            // provenance of, so the map is empty.
            op_output: Vec::new(),
        };
        assert!(validate(&bad).is_err(), "use-before-def must be rejected");
    }

    // ── The chained head's mint ────────────────────────────────────────────
    //
    // ⛔ TEST FIXTURE, PER THE AUDITOR'S RULING: the synthetic pick producer
    // below is a `ScalarMul` standing in for the device pick op the emission
    // names (Front A's sampler). It exists ONLY in these `#[cfg(test)]`
    // graphs — no non-test build constructs it, and the mint itself never
    // checks a producer's KIND (the caller owns that fact; see
    // `mint_embed_chain`'s docs). The stand-in needs exactly the structure
    // the mint walks: an op output one row per request whose input 0 chains
    // back to the lm-head matmul.
    //
    // The toy graph, per (width, hidden): sources `embed [width, hidden]`
    // (t0) and the tied matrix `[hidden, VOCAB]` (t1); ops
    // `MatmulTile(n=VOCAB, m=width)` then the stand-in `ScalarMul(m=width)`
    // over its output. VOCAB 2048 pads to 4096 — a whole pick-group
    // multiple, the mint's extent law — and the stand-in's `[width, VOCAB]`
    // output stands in for the pick stick: the mint states the pick's ROWS,
    // never its lane format (the producer's own contract, the doors' law).
    // The value tests' matrix keeps TWO shape duties: column 0 is `e_0`
    // (`W[0][0] = 1`, zeros below — so the stand-in's word 0 IS the pick
    // `t`, feeding the stage's check) and row 0 is the column index
    // (`W[0][j] = j`).
    const FIXTURE_VOCAB: u32 = 2048;

    fn chained_head_fixture(width: u32, hidden: u32) -> crate::lower::LoweringInput {
        crate::lower::LoweringInput {
            exported: None,
            sources: vec![
                SourceShape {
                    rows: width,
                    cols: hidden,
                },
                SourceShape {
                    rows: hidden,
                    cols: FIXTURE_VOCAB,
                },
            ],
            ops: vec![
                OpDesc {
                    op: SubOp::MatmulTile {
                        n: FIXTURE_VOCAB,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m: width,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                // THE SYNTHETIC PICK PRODUCER — test fixture only.
                OpDesc {
                    op: SubOp::ScalarMul { scale: 1.0 },
                    m: width,
                    inputs: vec![InputRef::Op(0)],
                },
            ],
            result: 1,
        }
    }

    /// ⭐ THE EMBED-CHAIN MINT'S SURGERY, stated as the graph's own
    /// invariants, at every width the front serves (1 = the degenerate solo
    /// case; 2 and 8 = batched): the four persistent/staged sources INSERTED
    /// at `[num_sources, +4)` (shift four), every op-output tid — the result
    /// included — renumbered up by four while the source ids below the old
    /// boundary stay UNTOUCHED (the embed rows and the lm-head weight among
    /// them), the `EmbedRowGather` node INSERTED at position 0 with every
    /// whole-tensor consumer of the embed rows re-pointed onto its fresh
    /// `[width, hidden]` output, and the `ChainStage` appended at the tail
    /// reading the pick stick, both persistent sticks, and both tables. The
    /// widening pass ran first (its pad is the mint's widened-law assert).
    /// Pinned here so a surgery regression is a test failure, not a card
    /// divergence. A second geometry — hidden 4096, `k = 8`,
    /// `l_sticks = 2` — pins the multi-l-stick shapes the per-stick cut
    /// walks.
    #[test]
    fn the_embed_chain_mint_inserts_the_stick_sources_and_repoints_the_entry() {
        let nz = |k| std::num::NonZeroU32::new(k).unwrap();
        for width in [1u32, 2, 8] {
            let mut g = lower_region(&chained_head_fixture(width, 2), nz(64));
            // Pre-widen: 2 sources; t2 the matmul out, t3 the pick stick (the
            // result); op_output [t2, t3].
            assert_eq!(g.num_sources, 2);
            assert_eq!(g.result.0, 3);
            let widen =
                widen_tied_embed_weight(&mut g).expect("the toy's tail admits the widening");
            assert_eq!(
                (widen.vocab, widen.pad_vocab),
                (FIXTURE_VOCAB, 4096),
                "the pad is the pick-group multiple the mint's extent law demands"
            );

            let mint = mint_embed_chain(&mut g, TensorId(0), TensorId(3), FIXTURE_VOCAB);
            assert_eq!(
                (mint.hidden, mint.vocab, mint.width),
                (2, FIXTURE_VOCAB, width),
                "the gather's own law: hidden from the matrix rows, vocab from the \
                 caller's statement, the width from the embed rows"
            );
            assert_eq!(
                (mint.s_stick.0, mint.l_stick.0),
                (2, 3),
                "the persistent sticks are the NEW sources at the old boundary"
            );
            assert_eq!(g.num_sources, 6, "four sources were inserted");
            let table_rows = 4096 + SAMPLE_SENTINEL_OFFSET + 1;
            // hidden 2: k=1, l_sticks=1 — the degenerate's shapes.
            assert_eq!((g.tensors[2].rows, g.tensors[2].cols), (width, 32));
            assert_eq!((g.tensors[3].rows, g.tensors[3].cols), (width, 32));
            assert_eq!((g.tensors[4].rows, g.tensors[4].cols), (table_rows, 32));
            assert_eq!((g.tensors[5].rows, g.tensors[5].cols), (table_rows, 32));
            // The old op outputs moved +4: the matmul out t2 -> t6, the pick
            // stick (the result) t3 -> t7; the two real sources are untouched.
            assert_eq!(g.result.0, 4 + 3, "the result re-point survived the bump");
            assert_eq!(g.op_output, vec![TensorId(6), TensorId(7)]);
            assert_eq!((g.tensors[0].rows, g.tensors[0].cols), (width, 2));
            assert_eq!((g.tensors[1].rows, g.tensors[1].cols), (2, 4096));

            // The gather node at position 0: the s stick, the l stick, then
            // the WEIGHT SOURCE (untouched by the bump), the fresh
            // [width, hidden] rows out at t8.
            let gather = &g.nodes[0];
            assert!(matches!(
                gather.op,
                SubOp::EmbedRowGather {
                    hidden: 2,
                    replaces: TensorId(0)
                }
            ));
            let tids = gather
                .inputs
                .iter()
                .map(|tr| tr.tensor.0)
                .collect::<Vec<_>>();
            assert_eq!(
                tids,
                vec![2, 3, 1],
                "the s stick, the l stick, the weight source"
            );
            assert_eq!(gather.output.tensor.0, 8);
            assert_eq!((g.tensors[8].rows, g.tensors[8].cols), (width, 2));

            // The matmul's activation is the gather's output now; its weight
            // is the same source it always read.
            let matmul = g
                .nodes
                .iter()
                .find(|n| matches!(n.op, SubOp::MatmulTile { .. }))
                .unwrap();
            assert_eq!(
                matmul.inputs[0].tensor.0, 8,
                "the entry consumer re-pointed"
            );
            assert_eq!(matmul.inputs[1].tensor.0, 1, "the weight source untouched");

            // The stage at the tail: the (bumped) pick stick, both persistent
            // sticks, both tables; a fresh s-value stick out at t9.
            let stage = g.nodes.last().unwrap();
            assert!(matches!(
                stage.op,
                SubOp::ChainStage {
                    pad_vocab: 4096,
                    table_rows: 8257,
                    k: 1
                }
            ));
            let tids = stage
                .inputs
                .iter()
                .map(|tr| tr.tensor.0)
                .collect::<Vec<_>>();
            assert_eq!(
                tids,
                vec![7, 2, 3, 4, 5],
                "the pick's stick, the s stick, the l stick, the two tables"
            );
            assert_eq!(stage.output.tensor.0, 9);
            assert_eq!((g.tensors[9].rows, g.tensors[9].cols), (width, 32));

            // Dense ids after the position-0 insert, and the graph validates.
            for (i, node) in g.nodes.iter().enumerate() {
                assert_eq!(node.id.0 as usize, i, "dense node ids after the insert");
            }
            assert!(validate(&g).is_ok(), "the chained graph must validate");
        }

        // The multi-l-stick geometry: hidden 4096 — k = 8 (the LX law), the l
        // index TWO sticks wide, the l table two sticks per row.
        let mut g = lower_region(&chained_head_fixture(2, 4096), nz(64));
        widen_tied_embed_weight(&mut g).expect("the toy's tail admits the widening");
        let mint = mint_embed_chain(&mut g, TensorId(0), TensorId(3), FIXTURE_VOCAB);
        assert_eq!(
            (
                g.tensors[mint.l_stick.index()].rows,
                g.tensors[mint.l_stick.index()].cols
            ),
            (4, 32),
            "the l stick is ONE STICK WIDE — [width·l_sticks, 32]: row r's l_sticks sticks \
             contiguous (stick r·l_sticks + b), never a stick-scattered [width, 64]"
        );
        let l_tab = TensorId(mint.s_stick.0 + 3);
        assert_eq!(
            (g.tensors[l_tab.index()].rows, g.tensors[l_tab.index()].cols),
            (8257, 64),
            "the l table's row width is the pair law: l_tab cols == the l stick's cols"
        );
        assert!(matches!(
            g.nodes.last().unwrap().op,
            SubOp::ChainStage { k: 8, .. }
        ));
        assert!(
            validate(&g).is_ok(),
            "the multi-stick chained graph validates"
        );
    }

    /// ⭐ THE CHAIN IS THE ROW, TWO-ENDED — the graph-level parity the
    /// lowering's parity descends from: evaluating the CHAINED graph with the
    /// persistent sticks seeded the pick's decompose (`s` pattern, `l` ramp)
    /// per row equals evaluating the PLAIN graph with the embed rows seeded
    /// `W[:, t]` per row. The gather's own output buffer is the direct half
    /// of the compare; the result is the end-to-end half. And the STAGE's
    /// fresh output is the pick's own s-value stick — the pattern the
    /// device's s-gather writes, `k·s + (i mod k)` — pinned through the
    /// fixture's word-0 trick (column 0 = `e_0`, row 0 = the column index:
    /// the stand-in's word 0 IS the pick).
    #[test]
    fn the_embed_chain_is_the_two_ended_row() {
        let nz = |k| std::num::NonZeroU32::new(k).unwrap();
        let hidden = 128u32;
        let pad = 4096usize;
        let voc = FIXTURE_VOCAB as usize;
        // The tied matrix, RAW layout `[hidden, VOCAB]`: column 0 is `e_0`
        // (row 0 a 1, the rest zeros) so the stand-in's word 0 is the pick;
        // row 0 is the column index; every other entry distinct per (h, j).
        let w_raw: Vec<f32> = (0..hidden as usize * voc)
            .map(|i| {
                let (h, j) = (i / voc, i % voc);
                match (h, j) {
                    (0, 0) => 1.0,
                    (0, _) => j as f32,
                    (_, 0) => 0.0,
                    _ => h as f32 * 0.5 + j as f32 * 0.25 - 3.0,
                }
            })
            .collect();
        // The WIDENED layout `[hidden, pad]`: the real columns then the pad's
        // prepare-time zeros — the bytes the staging reserves.
        let w_wide: Vec<f32> = (0..hidden as usize * pad)
            .map(|i| {
                let (h, c) = (i / pad, i % pad);
                if c < voc { w_raw[h * voc + c] } else { 0.0 }
            })
            .collect();
        let col =
            |t: usize| -> Vec<f32> { (0..hidden as usize).map(|h| w_wide[h * pad + t]).collect() };
        for width in [1u32, 2, 8] {
            // Row r's seeded pick: a distinct in-vocab token per row, past
            // the first slab so the s join is non-trivial.
            let picks: Vec<usize> = (0..width as usize).map(|r| 130 + 233 * r).collect();
            let plain_rows: Vec<f32> = picks
                .iter()
                .flat_map(|&t| col(t).into_iter())
                .collect::<Vec<_>>();

            let plain = lower_region(&chained_head_fixture(width, hidden), nz(64));
            let bufs_plain = eval_dag(&plain, &[&plain_rows, &w_raw]);

            let mut chained = lower_region(&chained_head_fixture(width, hidden), nz(64));
            widen_tied_embed_weight(&mut chained).expect("the toy's tail admits the widening");
            mint_embed_chain(&mut chained, TensorId(0), TensorId(3), FIXTURE_VOCAB);
            // The host no longer seeds the rows (zeros — nothing reads them);
            // each stick row carries its request's decompose — the SEED laws
            // the device's own ChainStage writes (idempotent with them).
            let s_seed: Vec<f32> = picks
                .iter()
                .flat_map(|&t| {
                    chain_s_stick_pattern(1, (t >> 6) as u32, CHAIN_IDX_BLOCK as usize)
                        .into_iter()
                        .map(|x| x as f32)
                })
                .collect();
            let l_seed: Vec<f32> = picks
                .iter()
                .flat_map(|&t| {
                    chain_l_stick_ramp((t & 63) as u32, hidden)
                        .into_iter()
                        .map(|x| x as f32)
                })
                .collect();
            // The tables are unread by the reference (the door's check); the
            // sizes come off the minted graph, never a second derivation.
            let tab_len = |tid: u32| {
                (chained.tensors[tid as usize].rows * chained.tensors[tid as usize].cols) as usize
            };
            let tabs = vec![vec![0f32; tab_len(4)]; 2];
            let bufs_chained = eval_dag(
                &chained,
                &[
                    &vec![0f32; (width * hidden) as usize],
                    &w_wide,
                    &s_seed,
                    &l_seed,
                    &tabs[0],
                    &tabs[1],
                ],
            );

            // The direct half: the gather produced exactly the rows the host
            // used to seed.
            assert_eq!(
                bufs_chained[8], plain_rows,
                "the gather's rows are the matrix's columns at the seeded picks"
            );
            // The end-to-end half: the chained graph answers what the
            // host-seeded graph answers.
            assert_eq!(
                bufs_chained[chained.result.index()],
                bufs_plain[plain.result.index()],
                "the chained graph answers what the host-seeded graph answers"
            );
            // The stage's fresh output is the pick's own s-value stick: the
            // pattern `k·s + (i mod k)` at k = 1 — the splat of `t >> 6`.
            for r in 0..width as usize {
                let expect = vec![(picks[r] >> 6) as f32; CHAIN_IDX_BLOCK as usize];
                assert_eq!(
                    &bufs_chained[9]
                        [r * CHAIN_IDX_BLOCK as usize..(r + 1) * CHAIN_IDX_BLOCK as usize],
                    &expect[..],
                    "the stage's row is the pick's s-value stick, row {r}"
                );
            }
        }
    }

    /// ⭐ THE SENTINEL AND THE PAD BOTH STAGE A ZERO ROW — the refusal the
    /// host readback's `idx > vocab` check performed, moved into the op: a
    /// degenerate logits row answers an index past the vocab, and the gather
    /// must stage a zero row rather than read past the matrix (a fault on the
    /// device bus is not a refusal). TWO legs per the op's own law: the
    /// SENTINEL RAMP (word 0 the mark `hidden`, every word naming the
    /// transposed placement's zero tail) and a pick INSIDE `[vocab, pad)` —
    /// in extent, reading the pad columns' own zeros. Only the degenerate
    /// row is zero; its neighbours still gather.
    #[test]
    fn the_embed_chain_gather_clamps_the_sentinel_to_a_zero_row() {
        let nz = |k| std::num::NonZeroU32::new(k).unwrap();
        let hidden = 128u32;
        let pad = 4096usize;
        let voc = FIXTURE_VOCAB as usize;
        let w_wide: Vec<f32> = (0..hidden as usize * pad)
            .map(|i| {
                let (h, c) = (i / pad, i % pad);
                if c < voc {
                    h as f32 * 0.5 + c as f32 * 0.25 - 3.0
                } else {
                    0.0
                }
            })
            .collect();
        let col =
            |t: usize| -> Vec<f32> { (0..hidden as usize).map(|h| w_wide[h * pad + t]).collect() };
        let width = 3u32;
        let mut g = lower_region(&chained_head_fixture(width, hidden), nz(64));
        widen_tied_embed_weight(&mut g).expect("the toy's tail admits the widening");
        mint_embed_chain(&mut g, TensorId(0), TensorId(3), FIXTURE_VOCAB);
        // Row 0 carries a real pick; row 1 the sentinel ramp (the mark);
        // row 2 a pick inside [vocab, pad) — the pad columns' own zeros.
        let s_seed: Vec<f32> = [0u32, 0, 3000 >> 6]
            .iter()
            .flat_map(|&s| {
                chain_s_stick_pattern(1, s, CHAIN_IDX_BLOCK as usize)
                    .into_iter()
                    .map(|x| x as f32)
            })
            .collect();
        let l_seed: Vec<f32> = [
            chain_l_stick_ramp(5, hidden),
            chain_l_sentinel_ramp(hidden),
            chain_l_stick_ramp(3000 & 63, hidden),
        ]
        .iter()
        .flat_map(|ramp| ramp.iter().map(|&x| x as f32))
        .collect();
        let tab_len =
            |tid: u32| (g.tensors[tid as usize].rows * g.tensors[tid as usize].cols) as usize;
        let bufs = eval_dag(
            &g,
            &[
                &vec![0f32; (width * hidden) as usize],
                &w_wide,
                &s_seed,
                &l_seed,
                &vec![0f32; tab_len(4)],
                &vec![0f32; tab_len(5)],
            ],
        );
        let mut expect = col(5);
        expect.extend(vec![0f32; hidden as usize * 2]);
        assert_eq!(
            bufs[8], expect,
            "the sentinel ramp and the pad-region pick both stage zero rows — and \
             only their rows"
        );
    }

    /// ⭐ THE PATTERNED SLAB RE-JOINS AT k ABOVE ONE — the s stick's LANE
    /// PATTERN (word 0 = `k·s + 0`) answers the pick through
    /// `s = word0 / k`, EXACT at every k: at hidden 2048 the LX law splits
    /// the slab k = 4, and the seeded pattern's first lane `4·s` re-joins
    /// the slab the ramp's lane lands in. The direct half of the two-ended
    /// compare at the split geometry, one row per request.
    #[test]
    fn the_embed_chain_rejoins_the_patterned_slab_at_k_above_one() {
        let nz = |k| std::num::NonZeroU32::new(k).unwrap();
        let hidden = 2048u32;
        let pad = 4096usize;
        let voc = FIXTURE_VOCAB as usize;
        let k = chain_slab_entries(u64::from(hidden)) as u32;
        assert_eq!(
            k, 4,
            "granite-3.1-2b's own geometry: the LX law splits k = 4"
        );
        let w_wide: Vec<f32> = (0..hidden as usize * pad)
            .map(|i| {
                let (h, c) = (i / pad, i % pad);
                if c < voc {
                    h as f32 * 0.125 + c as f32 * 0.0625 - 3.0
                } else {
                    0.0
                }
            })
            .collect();
        let col =
            |t: usize| -> Vec<f32> { (0..hidden as usize).map(|h| w_wide[h * pad + t]).collect() };
        let width = 2u32;
        // Slab 5 lane 33 and slab 7 lane 60: word 0 = 4·s, the pattern's
        // first lane; the re-join divides it back.
        let picks = [(5u32, 33u32), (7, 60)];
        let mut g = lower_region(&chained_head_fixture(width, hidden), nz(64));
        widen_tied_embed_weight(&mut g).expect("the toy's tail admits the widening");
        mint_embed_chain(&mut g, TensorId(0), TensorId(3), FIXTURE_VOCAB);
        let s_seed: Vec<f32> = picks
            .iter()
            .flat_map(|&(s, _)| {
                chain_s_stick_pattern(k, s, CHAIN_IDX_BLOCK as usize)
                    .into_iter()
                    .map(|x| x as f32)
            })
            .collect();
        let l_seed: Vec<f32> = picks
            .iter()
            .flat_map(|&(_, l)| chain_l_stick_ramp(l, hidden).into_iter().map(|x| x as f32))
            .collect();
        let tab_len =
            |tid: u32| (g.tensors[tid as usize].rows * g.tensors[tid as usize].cols) as usize;
        let bufs = eval_dag(
            &g,
            &[
                &vec![0f32; (width * hidden) as usize],
                &w_wide,
                &s_seed,
                &l_seed,
                &vec![0f32; tab_len(4)],
                &vec![0f32; tab_len(5)],
            ],
        );
        let mut expect = col((64 * 5 + 33) as usize);
        expect.extend(col((64 * 7 + 60) as usize));
        assert_eq!(
            bufs[8], expect,
            "the patterned slab re-joins: word0/k is the slab, the ramp's word0 the lane"
        );
    }

    /// ⭐ THE SEEDS ARE PER-ROW AT THE LAUNCH'S OWN WIDTH — row r carries ITS OWN token's
    /// decompose, pad rows repeat the last real token's, and `rows == 1` is the solo
    /// spelling byte-for-byte. This is the pin for the W=1-hidden b2 defect: the card
    /// path's tape arm staged `tokens.first()` and ONE stick at every width, so row 1's
    /// index never landed and its slab gather read whatever was there.
    #[test]
    fn the_chain_seeds_carry_every_row_at_the_launch_width() {
        let hidden = 2048u32;
        let k = 4u32;
        let lanes = CHAIN_IDX_BLOCK as usize;
        let toks = [64u32 * 5 + 33, 64 * 7 + 60, 64 * 9 + 12];
        // PER-ROW: row r's stick is its own token's pattern/ramp, not row 0's.
        let s = chain_s_seed(k, &toks, 2);
        assert_eq!(
            s,
            [
                chain_s_stick_pattern(k, toks[0] >> 6, lanes),
                chain_s_stick_pattern(k, toks[1] >> 6, lanes)
            ]
            .concat(),
            "row 1's s stick carries row 1's token, not row 0's"
        );
        let l = chain_l_seed(&toks, hidden, 2);
        assert_eq!(
            l,
            [
                chain_l_stick_ramp(toks[0] & 63, hidden),
                chain_l_stick_ramp(toks[1] & 63, hidden)
            ]
            .concat(),
            "row 1's l sticks carry row 1's token, not row 0's"
        );
        // THE PAD-ROW CLAMP: rows past the real tokens repeat the last real
        // token's seed — the same law the embedding staging pins.
        let s_pad = chain_s_seed(k, &toks, 5);
        assert_eq!(
            s_pad[4 * lanes..],
            chain_s_stick_pattern(k, toks[2] >> 6, lanes),
            "a pad row repeats the last real token's pattern"
        );
        // THE SOLO DEGENERATE: at one row the seed IS the one stick, byte for
        // byte — the W=1 rung's own spelling, kept exact by construction.
        assert_eq!(
            chain_s_seed(k, &toks, 1),
            chain_s_stick_pattern(k, toks[0] >> 6, lanes)
        );
        assert_eq!(
            chain_l_seed(&toks, hidden, 1),
            chain_l_stick_ramp(toks[0] & 63, hidden)
        );
        // An empty token list is a refusal, never a zero stick (zero is a valid
        // entry and a valid lane).
        assert!(std::panic::catch_unwind(|| chain_s_seed(k, &[], 1)).is_err());
        assert!(std::panic::catch_unwind(|| chain_l_seed(&[], hidden, 1)).is_err());
    }
}
