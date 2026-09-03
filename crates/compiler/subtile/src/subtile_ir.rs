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
    /// `a - b`, with a `[m, 1]` broadcast on `b` (the mean-centering half of a LayerNorm).
    Sub,
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
    rope_form_seal::Sealed + Copy + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + 'static
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
    /// The cache was written by a `SubOp::RopeAppend` earlier in this
    /// same forward (the producer is `nodes[producer_node_idx]`).
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

// ── Sub-operations ─────────────────────────────────────────────────

/// How an RmsNorm reads its gain vector.
///
/// ⛔ NOT AN `f32` OFFSET. `LoweredOp::RmsNorm` carries `gain_offset: f32`, and a float admits
/// values — 0.5, NaN — that no checkpoint convention means and no kernel implements. There are
/// exactly two conventions in the world, so they get exactly two names, and the conversion is
/// the place a third one has to be rejected instead of silently scaling wrong.
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

    /// Read the convention off a `LoweredOp::RmsNorm`'s offset.
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
}

/// Which positions a decode-time attention may attend to.
///
/// ⛔ NOT A `bool`. `LoweredOp::AttnDecode` carries `sliding: bool`, and a bare boolean at a
/// struct literal is the shape that lets a `sliding` land in a `is_global` (they are negations
/// of each other, and both are in scope at the same call). Naming the two masks makes the swap
/// a type error instead of a model that quietly attends to the wrong span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnMask {
    /// Every position up to and including the query's own.
    Causal,
    /// The gemma class's local layers: causal, further restricted to the last `window`
    /// positions. The WINDOW SIZE is not here — like the tanh soft cap it is a model constant
    /// resolved at emission and has never ridden the tape.
    SlidingWindow,
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
impl<F: RopeForm> SubOp<F> {
    /// The layer this node belongs to, when it carries one.
    ///
    /// Only the KV-cache writer names its layer on the tape; everything else in a layer is
    /// identified by its per-layer WEIGHTS, not by an index. That is enough: the re-roll only
    /// needs to know how far the layer advances between two copies of a body, and any one
    /// layer-carrying op in the body measures that.
    pub fn layer_index(&self) -> Option<u32> {
        match self {
            SubOp::RopeAppend { layer, .. } => Some(*layer),
            _ => None,
        }
    }

    pub fn reroll_class_key(&self, h: &mut impl std::hash::Hasher) {
        use std::hash::Hash;
        std::mem::discriminant(self).hash(h);
        match self {
            // The weight's storage scheme IS part of the class: a 4-bit and an
            // 8-bit affine matmul are different programs. Without this, OptiQ's
            // mixed-precision layers all hash alike and re-roll onto one body.
            SubOp::MatmulTile { weight } => {
                std::mem::discriminant(weight).hash(h);
                if let crate::lower::GemmWeight::Affine { affine } = weight {
                    affine.bits().get().hash(h);
                    affine.group().get().hash(h);
                }
            }
            // No attributes: the discriminant is the whole class.
            SubOp::SumReduce
            | SubOp::Reshape
            | SubOp::SiluMul
            | SubOp::TanhSoftCap
            | SubOp::ScalarWeightMul
            | SubOp::GateApply
            | SubOp::GateScale
            | SubOp::VisionRope
            | SubOp::GatedDeltaNet
            | SubOp::Mean => {}
            SubOp::Elementwise(k) => std::mem::discriminant(k).hash(h),
            SubOp::ScalarMul { scale } => scale.to_bits().hash(h),
            SubOp::RmsNorm { eps, gain } => {
                eps.to_bits().hash(h);
                std::mem::discriminant(gain).hash(h);
            }
            SubOp::RmsNormReduce { eps } | SubOp::RmsNormUnit { eps } => eps.to_bits().hash(h),
            SubOp::RmsNormApply { gain } => std::mem::discriminant(gain).hash(h),
            SubOp::RopeRotate { head_dim, .. } => head_dim.get().hash(h),
            // `layer` and `layout`'s cache tensors are per-layer IDENTITY, not class.
            SubOp::RopeAppend { head_dim, .. } => head_dim.get().hash(h),
            // `layout`, `producer`'s node index and `softmax_state` are per-layer identity;
            // `mask` is the class (sliding-window vs full) and is exactly what gemma-3 needs.
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
            }
            SubOp::GateSplit { half_cols } => half_cols.hash(h),
            SubOp::LoadPixels { in_features } => in_features.hash(h),
            SubOp::LoadPosEmbeds { width } => width.hash(h),
            SubOp::EmbeddingGather { indices_kind } => indices_kind.hash(h),
            SubOp::VarlenAttention { cu_kind } => cu_kind.hash(h),
            SubOp::EncoderAttn { geom, scale } => {
                geom.nqh().get().hash(h);
                geom.nkvh().get().hash(h);
                geom.hd().get().hash(h);
                scale.to_bits().hash(h);
            }
            SubOp::GemmaMoe {
                num_experts,
                top_k,
                moe_inter,
                group_size,
                bits,
            } => (num_experts, top_k, moe_inter, group_size, bits).hash(h),
            SubOp::Moe {
                qwen_shared,
                num_experts,
                top_k,
                moe_inter,
                shared_inter,
                norm_topk,
                group_size,
                bits,
            } => (
                qwen_shared,
                num_experts,
                top_k,
                moe_inter,
                shared_inter,
                norm_topk,
                group_size,
                bits,
            )
                .hash(h),
        }
    }
}

/// The sub-operation a node performs. Generic over `F: RopeForm`
/// (default [`NeoX`]) so a SubtileIR's rope nodes share the same
/// pairing form by construction.
///
/// Every variant has a `cpu_golden`-backed host evaluation in
/// [`eval_node`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SubOp<F: RopeForm = NeoX> {
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
    MatmulTile { weight: crate::lower::GemmWeight },
    /// Elementwise sum of equal-shaped inputs — the split-K combine. All
    /// inputs and the output are `[out_rows, out_cols]`.
    SumReduce,
    /// Re-interpret `inputs[0]`'s elements under a new `[out_rows, out_cols]`.
    /// Element COUNT is preserved and row-major order is preserved; only the
    /// extents change. Host semantics are the identity.
    ///
    /// ⛔ IT IS NOT FREE ON A STICK-LAID-OUT DEVICE, and that is the whole
    /// reason it is an OP rather than a view. `sdsc_abstract::dev_off_stk`
    /// places element `(i, j)` at `(j/stk)*(a*stk) + i*stk + (j%stk)`, where
    /// `a` is the ROW COUNT — so changing the extents moves every element.
    /// A target must lower this to a RESTICKIFY (a real re-laying copy); it
    /// must NOT give the output the input's placement. Pinned by
    /// `sdsc_abstract::tests::reshape_is_not_a_placement_alias_under_stick_layout`.
    Reshape,
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
    /// not as an operand. Emitted from [`crate::lower::LoweredOp::ScalarMul`]
    /// for architectures with scalar activation multipliers (Granite's
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
    /// chunk of `cols`) when `nb < cols`; the whole form is kept as a
    /// canonical IR-level op for the eval_node host reference and the
    /// `LoweredOp::RmsNorm` translation path that pre-decomposition
    /// callers (validate-only tests, partition lowering) consume.
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
    RmsNormReduce { eps: f32 },
    /// Phase 2 of chunked RmsNorm: per-chunk apply.
    /// `inputs[0]` = x chunk `[m, chunk_cols]`,
    /// `inputs[1]` = inv_rms `[m, 1]` (the producing
    /// [`SubOp::RmsNormReduce`] node's output region; lower_compute
    /// reads the side-channelled smem-vec slot, not this tile-region),
    /// `inputs[2]` = gamma chunk `[1, chunk_cols]`.
    /// Output: `y[m, chunk_cols] = x * inv_rms[per row] * gamma[per col]`.
    RmsNormApply { gain: GainConvention },
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
    /// cache write are GPU-only.
    RopeAppend {
        head_dim: HeadDim,
        layer: u32,
        layout: KvCacheLayout,
        #[doc(hidden)]
        _form: PhantomData<F>,
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
        layout: KvCacheLayout,
        producer: KvCacheProducer,
        softmax_state: SoftmaxStateId,
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
    /// ⛔ THE CAP IS NOT HERE, AND THAT IS A REPORTED GAP, NOT A DESIGN. `LoweredOp::TanhSoftCap`
    /// carries no cap either: the value is a model constant resolved at emission, so the tape has
    /// never held it. `eval_node` therefore has no host reference for this op and says so instead
    /// of inventing a cap.
    TanhSoftCap,
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
    /// Qwen3.5-MoE shared-expert combine: `out = routed + shared · σ(g)`, `g` a `[m, 1]`
    /// row-broadcast. `inputs` = `[routed, shared, g]`.
    GateScale,
    /// Vision pixel load: the host-staged patchified pixel buffer lands in this op's slot.
    /// NO inputs; output `[T, in_features]`.
    LoadPixels { in_features: u32 },
    /// Vision position embeddings: the host-staged, per-image-grid interpolated table lands in
    /// this op's slot. NO inputs; output `[T, width]`.
    LoadPosEmbeds { width: u32 },
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
    /// Gemma-4 fused MoE block: router (own leading rmsnorm, temperature softmax, per-expert
    /// score scale) over `router_in`, GeGLU experts over `expert_in`.
    /// `inputs` = `[router_in, expert_in, router_bundle, experts_bundle]`.
    GemmaMoe {
        num_experts: u32,
        top_k: u32,
        moe_inter: u32,
        group_size: u32,
        bits: u32,
    },
    /// Fused mixture-of-experts: router → top-k → gathered expert MLPs → weighted sum, plus an
    /// optional shared-expert tail. `inputs` = `[x, moe_weight_bundle]`.
    Moe {
        /// Qwen order (`softmax → topk → gather`, optional renorm + shared expert) vs Mixtral
        /// order (`topk → softmax`).
        qwen_shared: bool,
        num_experts: u32,
        top_k: u32,
        moe_inter: u32,
        shared_inter: u32,
        norm_topk: bool,
        group_size: u32,
        bits: u32,
    },
    /// Row mean: `out[m, 1] = mean(x, axis = -1)`. `inputs` = `[x]`.
    Mean,
}

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
    /// but the WEIGHTS a step binds live on the source `LoweredOp` — its
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

// ── Host evaluation ────────────────────────────────────────────────

/// Gather a tensor region into a dense row-major `(buf, rows, cols)`.
fn gather<F: RopeForm>(
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
            let mut out = vec![0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0f32;
                    for l in 0..k {
                        // W is [K, N] row-major: w[l * N + j].
                        sum += a[i * k + l] * w[l * n + j];
                    }
                    out[i * n + j] = sum;
                }
            }
            out
        }
        SubOp::SumReduce => {
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
        SubOp::Reshape => {
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
                // on-card result. (`LoweredOp::GeluErf` is a DISTINCT op and
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
        SubOp::RmsNormReduce { eps } => {
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
        SubOp::RmsNormApply { gain } => {
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
        // RopeAppend's host eval is rotation only (identical to RopeRotate);
        // its V input + the paged-cache write are GPU-only.
        SubOp::RopeRotate { head_dim, _form: _ } | SubOp::RopeAppend { head_dim, .. } => {
            let (x, xr, xc) = gather(&node.inputs[0], graph, bufs);
            let (cos, _, cc) = gather(&node.inputs[1], graph, bufs);
            let (sin, _, sc) = gather(&node.inputs[2], graph, bufs);
            debug_assert_eq!((xr, xc), (out_rows, out_cols), "rope shape");
            let hd = head_dim.get() as usize;
            let half = hd / 2;
            let (rows, cols) = (xr as usize, xc as usize);
            debug_assert_eq!(cols % hd, 0, "rope cols not a multiple of head_dim");
            debug_assert!(cc as usize >= hd && sc as usize >= hd, "rope cos/sin width");
            let heads = cols / hd;
            let mut out = x.clone();
            for r in 0..rows {
                for h in 0..heads {
                    let base = r * cols + h * hd;
                    for d in 0..half {
                        let x0 = x[base + d];
                        let x1 = x[base + half + d];
                        // Apply the cos/sin table ELEMENT-WISE (cos[d] for the first half,
                        // cos[d+half] for the second), matching the SuperDSC on-card RoPE
                        // (`out = x*cos + rotate_half(x)*sin`, where the pointwise mul reads
                        // the full [hd] table per element). For REAL rope_cos_sin tables the
                        // two halves are duplicated (cos[d+half]==cos[d]), so this is a NO-OP
                        // for every real run; it only matters when cos/sin are NOT duplicated
                        // (the selftest's synthetic sources), where the old `cos[d]`-for-both
                        // form made the eval_dag golden diverge from the faithful on-card
                        // lowering (the spurious t341 RopeRotate cos≈0.45 selftest artifact).
                        out[base + d] = x0 * cos[d] - x1 * sin[d];
                        out[base + half + d] = x0 * sin[d + half] + x1 * cos[d + half];
                    }
                }
            }
            out
        }
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

        // ── Ops with NO host reference ──────────────────────────────
        //
        // ⛔ THESE PANIC, AND THE MESSAGE NAMES WHAT IS MISSING. `eval_node` is the
        // numeric ORACLE — the thing an on-card result is compared against. An op
        // whose value depends on state the host does not have (a runtime index
        // table, a staged pixel buffer, an ssm carry, an opaque weight bundle that
        // is not a gatherable tensor) has no oracle, and returning zeros or the
        // input unchanged would make the comparison PASS while proving nothing.
        // That is strictly worse than no oracle, so it is not on offer.
        SubOp::TanhSoftCap => panic!(
            "SubOp::TanhSoftCap has no host reference: the cap is a model constant \
             resolved at emission and has never been carried on the tape (neither \
             LoweredOp::TanhSoftCap nor Instruction::TanhSoftCap holds it), so there \
             is no value here to divide by"
        ),
        SubOp::GateSplit { .. } => panic!(
            "SubOp::GateSplit has no host reference: it has TWO outputs (q and gate) \
             and eval_node returns the buffer for ONE region, so the gate half would \
             be silently dropped"
        ),
        SubOp::LoadPixels { .. } | SubOp::LoadPosEmbeds { .. } => panic!(
            "SubOp::LoadPixels / LoadPosEmbeds have no host reference: the buffer is \
             STAGED BY THE HOST at runtime (patchified pixels, per-image-grid \
             interpolated position embeddings) and is not computed from operands"
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
        SubOp::GatedDeltaNet => panic!(
            "SubOp::GatedDeltaNet has no host reference: the conv/ssm state is \
             runtime-ambient per layer, so a single node's output is not a function of \
             its operands alone"
        ),
        SubOp::Moe { .. } | SubOp::GemmaMoe { .. } => panic!(
            "SubOp::Moe / GemmaMoe have no host reference: the router and expert \
             weights arrive as ONE opaque bundle source, which `gather` cannot read as \
             a tensor region"
        ),

        SubOp::AttnDecode {
            geom, scale, mask, ..
        } => {
            // ⛔ THE WINDOW SIZE IS NOT ON THE TAPE, so a sliding-window layer has no
            // host reference — evaluating it with the causal bound would silently
            // attend across the whole prefix and report agreement with a kernel that
            // does not do that.
            assert!(
                matches!(mask, AttnMask::Causal),
                "SubOp::AttnDecode has no host reference under AttnMask::SlidingWindow: \
                 the window size is a model constant resolved at emission and has never \
                 been carried on the tape"
            );
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
            debug_assert_eq!(node.inputs.len() % 2, 1, "attn inputs = Q + (K,V) pairs");
            // kv-head offset of this block (from the first K segment's column slice).
            let kvh_start = node.inputs[1].region.cols.start as usize / hd;
            // Concatenate K/V segments along sequence; each segment spans this
            // block's kv-heads (kv_count * hd wide).
            let mut k_all: Vec<f32> = Vec::new();
            let mut v_all: Vec<f32> = Vec::new();
            let mut kv_count = 0usize;
            let mut i = 1;
            while i < node.inputs.len() {
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
                    let causal_bound = seq_len - mq + qi;
                    let mut scores = vec![0f32; seq_len];
                    for (s, score) in scores.iter_mut().enumerate() {
                        if s > causal_bound {
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
                    let maxs = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for sc in scores.iter_mut() {
                        *sc = (*sc - maxs).exp();
                        sum += *sc;
                    }
                    for sc in scores.iter_mut() {
                        *sc /= sum;
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
        // Arity comes from THE SubOp registry (`crate::subops`) — one
        // row per op, no hand-maintained parallel match.
        let arity_ok = crate::subops::subop_arity_ok(&node.op, arity);
        if !arity_ok {
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

/// Out-columns of an op (mirrors `crate::lower`): GEMM → n, attention → the geometry's own
/// token-stream width, everything else preserves input-0 width.
pub(crate) fn op_out_cols(op: crate::lower::LoweredOp, operand_cols: impl Fn(usize) -> u32) -> u32 {
    // Derived from THE op registry (`crate::ops`) — one row per op, no
    // hand-maintained parallel match.
    crate::ops::lowered_op_out_cols(&op, operand_cols)
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
pub fn lower_region(
    input: &crate::lower::LoweringInput,
    nb: std::num::NonZeroU32,
) -> SubtileIR<NeoX> {
    use crate::lower::{InputRef, LoweredOp};
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
        // ⛔ OPERAND 0 IS OPTIONAL. The host-staged vision loads (`LoadPixels`,
        // `LoadPosEmbeds`) take NO operands — their buffer is delivered by the
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
        let out_cols = op_out_cols(desc.op, |k| {
            resolve(desc.inputs[k], &op_tensor, &op_cols, &tensors).2
        });
        let out_t = TensorId(tensors.len() as u32);
        tensors.push(TensorShape {
            rows: m,
            cols: out_cols,
        });

        match desc.op {
            LoweredOp::Gemm { n, weight } => {
                // k is structurally derived from the activation's
                // column count — it is not a separate field. See
                // LoweredOp::Gemm doc.
                let (in0_t, _r, in0_cols) = in0_named("Gemm");
                let k = in0_cols;
                let (w_t, _wr, _wc) = resolve(desc.inputs[1], &op_tensor, &op_cols, &tensors);
                let act = TensorRegion {
                    tensor: in0_t,
                    region: Region {
                        rows: Range::new(0, m),
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
                    let (ws_t, ws_r, ws_c) =
                        resolve(desc.inputs[2], &op_tensor, &op_cols, &tensors);
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
                    op: SubOp::MatmulTile { weight },
                    inputs: mm_inputs,
                    output: TensorRegion {
                        tensor: out_t,
                        region: Region {
                            rows: Range::new(0, m),
                            cols: whole,
                        },
                    },
                });
            }
            other => {
                // Populate per-variant typed witnesses. RopeAppend's
                // KvCacheLayout is keyed on its `K_cache` input
                // (desc.inputs[4] per the validate-arity contract);
                // AttnDecode's KvCacheLayout/KvCacheProducer are keyed
                // on its prefix-K input (desc.inputs[1] in the fused
                // decode shape).
                let subop: SubOp<NeoX> = match other {
                    LoweredOp::RmsNorm { eps, gain_offset } => {
                        // The (1+w) convention is CARRIED, not dropped and not folded into the
                        // weights at load: folding would make the loaded gain disagree with the
                        // checkpoint, so every consumer that re-reads it (dumps, the numeric
                        // reference, a second target) would see a different tensor.
                        let gain = GainConvention::from_offset(gain_offset).unwrap_or_else(|| {
                            panic!(
                                "rmsnorm gain_offset={gain_offset} is neither the w nor the \
                                 (1+w) convention, and no kernel implements a third"
                            )
                        });
                        SubOp::RmsNorm { eps, gain }
                    }
                    LoweredOp::Silu => SubOp::Elementwise(EwKind::Silu),
                    LoweredOp::GateSplit { half_cols } => SubOp::GateSplit { half_cols },
                    LoweredOp::GateApply => SubOp::GateApply,
                    LoweredOp::GateScale => SubOp::GateScale,
                    LoweredOp::EmbeddingGather { indices_kind } => {
                        SubOp::EmbeddingGather { indices_kind }
                    }
                    LoweredOp::LoadPosEmbeds { width } => SubOp::LoadPosEmbeds { width },
                    LoweredOp::LoadPixels { in_features } => SubOp::LoadPixels { in_features },
                    LoweredOp::VisionRope => SubOp::VisionRope,
                    LoweredOp::GeluErf => SubOp::Elementwise(EwKind::GeluErf),
                    LoweredOp::QuickGelu => SubOp::Elementwise(EwKind::QuickGelu),
                    LoweredOp::VarlenAttention { cu_kind } => SubOp::VarlenAttention { cu_kind },
                    LoweredOp::EncoderAttn { geom, scale } => SubOp::EncoderAttn { geom, scale },
                    LoweredOp::GatedDeltaNet => SubOp::GatedDeltaNet,
                    LoweredOp::GemmaMoe {
                        num_experts,
                        top_k,
                        moe_inter,
                        group_size,
                        bits,
                    } => SubOp::GemmaMoe {
                        num_experts,
                        top_k,
                        moe_inter,
                        group_size,
                        bits,
                    },
                    LoweredOp::Moe {
                        qwen_shared,
                        num_experts,
                        top_k,
                        moe_inter,
                        shared_inter,
                        norm_topk,
                        group_size,
                        bits,
                    } => SubOp::Moe {
                        qwen_shared,
                        num_experts,
                        top_k,
                        moe_inter,
                        shared_inter,
                        norm_topk,
                        group_size,
                        bits,
                    },
                    LoweredOp::RmsNormUnit { eps } => SubOp::RmsNormUnit { eps },
                    LoweredOp::ScalarWeightMul => SubOp::ScalarWeightMul,
                    LoweredOp::TanhSoftCap => SubOp::TanhSoftCap,
                    LoweredOp::Gelu => SubOp::Elementwise(EwKind::Gelu),
                    LoweredOp::Mul => SubOp::Elementwise(EwKind::Mul),
                    LoweredOp::ScalarMul { scale } => SubOp::ScalarMul { scale },
                    LoweredOp::SiluMul => SubOp::SiluMul,
                    LoweredOp::Add => SubOp::Elementwise(EwKind::Add),
                    // ⛔ THE OBVIOUS IMPLEMENTATION IS WRONG, SO THE MESSAGE
                    // SAYS WHICH ONE. "It is a view, so alias the buffer" is
                    // the first thing anyone tries and it silently corrupts:
                    // a device tensor is STICK-laid-out and
                    // `sdsc_abstract::dev_off_stk` places element (i, j) at
                    //     (j / stk) * (a * stk) + i * stk + (j % stk)
                    // where `a` is the ROW COUNT. Two tensors over the same
                    // bytes with different (rows, cols) therefore disagree on
                    // where every element lives. gemma-4's reshapes are
                    // per-head views ([M, heads*hd] -> [M*heads, hd]) — rows
                    // change, so an alias moves everything.
                    //
                    // Nor does the arrangement authority catch it:
                    // `declare_arrangement` keys on tensor NAME, and an alias
                    // gives the two views two names, so no conflict is seen.
                    //
                    // A reshape needs a RESTICKIFY — a real copy re-laying out
                    // from the old stick layout to the new. Spyre already has
                    // that machinery; this pipeline just has no op for it yet.
                    LoweredOp::Reshape { .. } => SubOp::Reshape,
                    // Row-broadcast bias: elementwise add whose second
                    // operand is a `[1, n]` source. Exact at decode
                    // (m = 1, both rows); m > 1 relies on the same
                    // broadcast row-stride handling `Add` has for
                    // `[1, n]` operands.
                    LoweredOp::BiasAdd => SubOp::Elementwise(EwKind::Add),
                    LoweredOp::RopeRotate { head_dim } => SubOp::RopeRotate {
                        head_dim,
                        _form: PhantomData,
                    },
                    LoweredOp::Mean => SubOp::Mean,
                    LoweredOp::Sub => SubOp::Elementwise(EwKind::Sub),
                    LoweredOp::RopeAppend {
                        head_dim,
                        layer,
                        is_global: _,
                        interleaved: _,
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
                        // on the lowered op itself — stamped from
                        // `model.bounds` upstream — and flow into the
                        // byte-offset math via the SubOp fields. No
                        // separate shape-table to check against.
                        let layout = KvCacheLayout::for_cache_tensors(k_cache_t, v_cache_t);
                        k_cache_producer_node.insert(k_cache_t, nodes.len() as u32);
                        SubOp::RopeAppend {
                            head_dim,
                            layer,
                            layout,
                            _form: PhantomData,
                        }
                    }
                    LoweredOp::AttnDecode {
                        geom,
                        scale,
                        valid_len,
                        sliding,
                    } => {
                        let mask = if sliding {
                            AttnMask::SlidingWindow
                        } else {
                            AttnMask::Causal
                        };
                        let (prefix_k_t, _, _) =
                            resolve(desc.inputs[1], &op_tensor, &op_cols, &tensors);
                        // prefix_v (inputs[2]) is a DISTINCT cache tensor from
                        // prefix_k (inputs[1]). `for_cache_tensor(t)` aliases
                        // V=K — using it here made the lowering's V load read
                        // the K cache (attn output == K, caught by the
                        // full-tape interp diff). Inputs = [q, prefix_k,
                        // prefix_v, new_k, new_v].
                        let (prefix_v_t, _, _) =
                            resolve(desc.inputs[2], &op_tensor, &op_cols, &tensors);
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
                    LoweredOp::Gemm { .. } => unreachable!("gemm handled above"),
                };
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
                    other,
                    LoweredOp::Silu
                        | LoweredOp::Mul
                        | LoweredOp::ScalarMul { .. }
                        | LoweredOp::Add
                        | LoweredOp::SiluMul
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
                                        rows: Range::new(0, m),
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
                                rows: Range::new(0, m),
                                cols: blk,
                            },
                        },
                    });
                }
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
                    op: SubOp::RmsNormReduce { eps },
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
                        op: SubOp::RmsNormApply { gain },
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
                op: SubOp::Reshape,
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
    use crate::lower::{InputRef, LoweredOp, OpDesc};
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
    #[cfg(any(feature = "spyre", feature = "superdsc"))]
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
            sources: vec![
                SourceShape { rows: m, cols: k },
                SourceShape { rows: n, cols: k },
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm {
                        n,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Silu,
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
            sources: vec![
                SourceShape { rows: m, cols: k }, // 0: x  [m, k]
                SourceShape { rows: k, cols: n }, // 1: Wg [k, n] (gate)
                SourceShape { rows: k, cols: n }, // 2: Wu [k, n] (up)
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::Gemm {
                        n,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)], // gate = x @ Wg
                },
                OpDesc {
                    op: LoweredOp::Gemm {
                        n,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(2)], // up   = x @ Wu
                },
                OpDesc {
                    op: LoweredOp::SiluMul,
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
            sources: vec![
                SourceShape { rows: m, cols: k }, // 0: x     [m, k]
                SourceShape { rows: 1, cols: k }, // 1: gamma [1, k]
            ],
            ops: vec![
                OpDesc {
                    op: LoweredOp::RmsNorm {
                        eps: 1e-6,
                        gain_offset: 0.0,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::Mul,
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
        for apply in 1..4 {
            assert_eq!(
                preds[apply],
                vec![SubtileId(0)],
                "RmsNormApply {} reads inv_rms from the sole reduce writer",
                apply,
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
                    op: LoweredOp::Gemm {
                        n: qdim,
                        weight: crate::lower::GemmWeight::Dense,
                    },
                    m,
                    inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
                },
                OpDesc {
                    op: LoweredOp::RopeRotate {
                        head_dim: HeadDim::new(hd),
                    },
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
        let (qdim, kvdim) = (hq * hd, hkv * hd); // both = 4
        let scale = 0.5f32;
        let input = crate::lower::LoweringInput {
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
                    op: LoweredOp::RopeAppend {
                        head_dim: HeadDim::new(hd),
                        layer: 0,
                        is_global: true,
                        interleaved: false,
                    },
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
                    op: LoweredOp::AttnDecode {
                        geom: ModelAttnGeometry::mint(
                            QueryHeads::new(hq),
                            KvHeads::new(hkv),
                            HeadDim::new(hd),
                        )
                        .expect("the fixture's kv heads divide its query heads"),
                        scale,
                        // l prefix rows (fully valid) ++ 1 new = l+1.
                        valid_len: l + 1,
                        sliding: false,
                    },
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
}
