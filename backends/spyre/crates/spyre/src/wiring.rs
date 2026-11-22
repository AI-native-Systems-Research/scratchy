// SPDX-License-Identifier: Apache-2.0
//! The per-model launch wiring, as SPYRE'S OWN TYPES — emitted by `#[forward]`
//! as a `static`, never parsed.
//!
//! ⛔ WHAT THIS REPLACES, AND WHY IT IS NOT A FILE FORMAT.
//!
//! Every fact below is a compile-time constant the macro already holds. It used
//! to hold them as the TYPED `SourceBinding` enum, flatten them to JSON strings
//! (`{"id":7,"role":"prefix_k","layer":3}`), bake the string into the binary,
//! and have the worker `serde_json`-parse it back and `match role.as_str()`
//! into the same distinction it started as. A compile-time enum round-tripped
//! through runtime strings, on every model load.
//!
//! That round-trip is not merely wasteful, it is where model facts go to
//! disagree: the string form has no exhaustiveness, so a new role reaches the
//! worker as an unmatched `_ => {}` instead of a compile error, and the
//! `"{disk}.weight"` naming rule it carried could not express a bare
//! `nn.Parameter` (gemma-4's `layer_scalar`) or a non-default decoder root.
//!
//! So the roles are an ENUM here and the macro emits values of it. Adding a
//! role breaks every match that must learn it, at build time, in one place.

use scratchy_tensors::GpuTensor;

// The query's extent pair and its two halves, re-exported for the wiring's own emission: the
// models crate (where `#[forward]` expands) names every wiring type through this crate's root,
// and `PagedStep` carries the pair ([`PagedStep::extent`]) — so its literal needs the types in
// reach through the same prefix every other wiring type uses.
pub use ktir_core::ddl_binds::ElementArrangement;
pub use scratchy_spyre_bundle::rung::{QueryExtent, RungCap, RungWidth};

/// The staged tensor of a linear layer, whatever its quant variant.
///
/// ⛔ NOT `LinearLayer::dense_weight()`, WHICH PANICS ON EVERY QUANT VARIANT.
/// That accessor exists for CUTLASS, which only takes dense bf16. Spyre stages
/// BYTES and retiles them for the PT array, so a quantized weight is not a
/// special case here — it is the point (fp8 W8A8 is the bandwidth win). A
/// `Result` rather than a panic because a variant with no single backing
/// tensor (a GGML concat of several) is a real "this model cannot be staged"
/// answer, not a bug to abort on.
pub fn linear_weight(l: &scratchy_layers::layers::LinearLayer) -> anyhow::Result<GpuTensor> {
    use scratchy_layers::layers::LinearLayer as L;
    Ok(match l {
        L::Dense(x) => x.weight,
        L::Fp8(x) => x.weight,
        other => anyhow::bail!(
            "spyre: linear weight variant {} has no single staged tensor",
            std::any::type_name_of_val(other)
        ),
    })
}

/// The fp8 code tensor — the 1-byte payload staged verbatim.
pub fn fp8_weight(l: &scratchy_layers::layers::Fp8AnyLinear) -> GpuTensor {
    use scratchy_layers::layers::Fp8AnyLinear as F;
    match l {
        F::Std(x) => x.weight,
        F::Block(x) => x.weight,
    }
}

/// The fp8 per-channel scale — the sibling of [`fp8_weight`], bound at the
/// `WeightScale` source of the same gemm.
pub fn fp8_scale(l: &scratchy_layers::layers::Fp8AnyLinear) -> GpuTensor {
    use scratchy_layers::layers::Fp8AnyLinear as F;
    match l {
        F::Std(x) => x.weight_scale,
        // The block-scaled variant names it `weight_scale_inv` — it stores the
        // RECIPROCAL, per the block-fp8 checkpoint convention.
        F::Block(x) => x.weight_scale_inv,
    }
}

// ── The MoE bundle tensors, one accessor per (bundle, tensor) ─────────────────
//
// A MoE bundle field holds SEVERAL tensors where every other weight kind holds one, so
// `superdsc_weights`'s per-source arms reach INSIDE the field through these. They return the
// tensor VERBATIM — staging (orientation, device-width pad) is the worker's, decided by the
// source's baked shape, exactly as for a single-tensor weight.

/// The gemma-4 router's dense `[num_experts, hidden]` projection.
pub fn gemma_router_gate(l: &scratchy_layers::layers_moe::GemmaRouterLayer) -> GpuTensor {
    l.gate
}

/// The gemma-4 router's `[num_experts]` per-expert score scale.
pub fn gemma_router_per_expert_scale(
    l: &scratchy_layers::layers_moe::GemmaRouterLayer,
) -> GpuTensor {
    l.per_expert_scale
}

/// The gemma-4 router's `[hidden]` RMSNorm gain on its input.
pub fn gemma_router_scale(l: &scratchy_layers::layers_moe::GemmaRouterLayer) -> GpuTensor {
    l.scale
}

/// The SwitchGLU experts' gate-projection codes, stacked `[E, out, in]`.
pub fn switch_glu_gate_w(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_gate_w
}

/// The SwitchGLU experts' gate-projection per-channel scale, stacked `[E, out, 1]`.
pub fn switch_glu_gate_s(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_gate_scales
}

pub fn switch_glu_up_w(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_up_w
}

pub fn switch_glu_up_s(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_up_scales
}

pub fn switch_glu_down_w(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_down_w
}

pub fn switch_glu_down_s(l: &scratchy_layers::layers_moe::SwitchGluExpertsLayer) -> GpuTensor {
    l.expert_down_scales
}

impl Wiring {
    /// The query-row CAPACITY this bundle was baked at — the embedding activation's placement rows.
    ///
    /// ⛔⛔⛔ THIS IS A CAPACITY, NOT THE ROW COUNT OF A LAUNCH, AND IT IS NOT WHAT THE TAPE
    /// STAGES. A decode step fills ONE row; a prompt chunk fills `mq`, which is at most this. I
    /// briefly made `forward_shape` take its `rows` from here on the reasoning that "the bundle
    /// already knows" — it knows how many rows it can HOLD. On granite that is 96, so every decode
    /// forward staged 96 rows of embedding, 96 rotary rows and a 96-row causal mask where one row
    /// was read, and the card stopped responding.
    ///
    /// Kept because the launch row count must be checked AGAINST it: a launch wider than the bake
    /// stages past the placement. That check is [`Wiring::forward_shape`]'s.
    pub fn baked_row_capacity(&self) -> RowCapacity {
        RowCapacity((self.tensor_shapes[self.embed_src as usize].0 as usize).max(1))
    }

    /// THE FORWARD TAPE FOR THIS BUNDLE.
    ///
    /// ⭐ EVERY FIELD IS GENERATED DATA. The shape used to be assembled in the worker out of a
    /// `BundleMeta` the worker had itself built by scanning the artifact; here it is a projection
    /// of the emitted wiring, so there is nothing for a second opinion to disagree with.
    pub fn forward_shape(
        &self,
        // ⛔ THE ROWS THIS LAUNCH FILLS — 1 for a decode step, `mq` for a prompt chunk. NOT the
        // bundle's capacity: see [`Self::baked_row_capacity`], which this is checked against.
        rows: StagedRows,
    ) -> crate::forward_tape::ForwardShape {
        self.forward_shape_within(self.baked_row_capacity(), rows, self.paged)
    }

    /// [`Self::forward_shape`] with the capacity and the paged step supplied by the CALLER rather
    /// than read off this wiring.
    ///
    /// ⭐ WHY A CALLER MAY KNOW BETTER. A decode batch rung is the same tape baked at `B` rows and is
    /// addressed by fingerprint, not through a wiring — so `Wirings.decode` carries every rung's
    /// tensor ids correctly while carrying only the single-request graph's ROW COUNT and step. A
    /// launch on the `B`-row rung is legitimate and [`Self::baked_row_capacity`] answers 1 for it, so
    /// checking against the wiring rejects a correct launch; and its step is the batch's, at the walk
    /// of the body that runs ([`PagedStep::on_body`]).
    ///
    /// ⛔ THE GUARD IS NOT WEAKENED, IT IS POINTED AT THE RIGHT NUMBER. `RowCapacity` still has no
    /// integer door: the only way to obtain one is this wiring's own placement
    /// ([`Self::baked_row_capacity`]) or a baked rung width
    /// ([`RowCapacity::of_baked_rung`]) — both artifact-derived. A caller cannot widen the check by
    /// arithmetic.
    pub fn forward_shape_within(
        &self,
        // The rows the bundle ACTUALLY RUNNING was baked to hold. Equal to
        // [`Self::baked_row_capacity`] whenever the wiring describes that bundle.
        cap: RowCapacity,
        // ⛔ THE ROWS THIS LAUNCH FILLS — 1 for a decode step, `mq` for a prompt chunk.
        rows: StagedRows,
        // The steps of the body that runs — this wiring's own for its own bundle.
        paged: ClassSteps,
    ) -> crate::forward_tape::ForwardShape {
        assert!(
            cap.admits(rows),
            "forward tape asked for {} row(s) from a bundle baked to hold {}: every host buffer \
             would be staged at a shape the placement cannot take",
            rows.get(),
            cap.stated(),
        );
        crate::forward_tape::ForwardShape {
            rows: rows.get(),
            embed_src: self.embed_src,
            cos_srcs: self.cos_srcs,
            sin_srcs: self.sin_srcs,
            paged,
            phase: self.phase,
        }
    }
}

/// ⭐⭐⭐ WHETHER A BODY GATHERS — read off its own launch groups ([`Self::of_launch_groups`]), the
/// only door: a fact that differs per BODY, never asked of a bundle or a caller.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GathersKv(bool);

impl GathersKv {
    /// ⭐⭐⭐ THE **PER-BODY** ANSWER, off the body's own LAUNCH GROUPS — each group's
    /// `bundle::KvShifts::gathered`, which the emitter stamps on the fold group it built the gather into.
    ///
    /// ⛔ WHY A THIRD DOOR RATHER THAN REUSING THE LAYOUT ONE. [`Self::of_layout`] answers about a
    /// BUNDLE: `KV_BLOCK_INDEX_TID` is placed if ANY body of it gathers. A decode bundle holds a whole
    /// sk_bucket ladder of bodies plus their fold-fused twins, and which one a step runs is the
    /// selector's answer from the live context length — so the bundle's placement cannot answer "does
    /// the body I am about to launch gather", and answering it from the bundle is exactly the mistake
    /// this file has now recorded three times (see `superdsc_exec::Executor::step_body`).
    ///
    /// The ops are the right authority for the same reason `latch_paged_geometry` reads `page_slots` off
    /// them: a bake fact rides on the launch group it describes, so it cannot be true of a body that
    /// does not carry it.
    pub fn of_launch_groups(gathered: impl IntoIterator<Item = bool>) -> Self {
        GathersKv(gathered.into_iter().any(|g| g))
    }

    pub const fn get(self) -> bool {
        self.0
    }
}

// ── THE TWO ROW COUNTS, WHICH ARE NOT THE SAME NUMBER ───────────────────────

/// ⭐⭐⭐ THE TOKEN ROWS **ONE LAUNCH STAGES**.
///
/// ⛔ NAMED `StagedRows`, NOT `StagedRows`, BECAUSE THAT NAME IS TAKEN — `sdsc_abstract::
/// LaunchRows` is a borrowed view of a decode batch's rows (which request sits in which slot),
/// an entirely different thing that this same file already imports. Two types with one name, in
/// one module, differing only by crate path is the collision this whole newtype exists to
/// prevent; introducing one while fixing another would be its own joke. One for a decode step, `mq` for a prompt chunk.
///
/// ⛔ THIS IS NOT [`RowCapacity`], AND THE DIFFERENCE HAS NOW COST TWO BUGS IN ONE FILE.
/// A bundle is baked to HOLD some number of rows; a launch FILLS some number, at most that. Both
/// are `usize`, both are called "rows", and granite's decode bundle holds 96 while a decode step
/// fills 1 — so passing the wrong one is not a crash, it is a plausible number:
///
///   * `ForwardShape::rows` took the capacity, so every decode step staged 96 rows of embedding,
///     96 rotary rows and a 96-row causal mask into buffers one row is read from.
///   * `ConstantEnv::rows` took the capacity, so `reductions = rows > 1` was TRUE on the decode
///     path and bound two reduce constants the pre-tape decode binder never bound.
///
/// Both were written by the same hand, in the same file, from the same assumption — which is why
/// the fix is a TYPE and not two corrected call sites. There is no `From<usize>` and no
/// conversion from a capacity: the only way in is [`Self::of_launch`], at the site that knows how
/// many rows it is about to stage.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct StagedRows(usize);

impl StagedRows {
    /// The rows this launch fills. `None` at zero — a forward that stages nothing is not a
    /// forward, and zero is the value this hardware reads as a maximum.
    pub fn of_launch(rows: usize) -> Option<StagedRows> {
        (rows >= 1).then_some(StagedRows(rows))
    }

    /// Exactly one row — a decode step. Named, so the common case cannot be spelled with a bare
    /// integer that a capacity could also be spelled with.
    pub const ONE: StagedRows = StagedRows(1);

    pub const fn get(self) -> usize {
        self.0
    }
}

/// ⭐ THE ROWS A BUNDLE WAS BAKED TO HOLD — its embedding activation's placement rows.
///
/// ⛔ A CEILING, NOT A COUNT. It answers "how wide a launch may this bundle take", and the only
/// thing it may be used for is checking a [`StagedRows`] against it. It deliberately has no
/// accessor that yields a bare `usize` usable as a row count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RowCapacity(usize);

impl RowCapacity {
    /// Does this bundle admit a launch that wide?
    pub const fn admits(self, rows: StagedRows) -> bool {
        rows.0 <= self.0
    }
    /// For diagnostics only — a message saying how wide the bundle is.
    pub const fn stated(self) -> usize {
        self.0
    }

    /// ⭐ THE CAPACITY OF A **BAKED LADDER RUNG**, which is NOT readable from the wiring it runs
    /// through.
    ///
    /// A decode batch rung is the SAME TAPE baked at `B` rows, and the emitter says so outright:
    /// *"a rung is the same tape baked at B rows and is addressed by FINGERPRINT, not through this
    /// wiring"* (`codegen.rs`, the `slot.0` guard). So one `Wirings.decode` describes every rung's
    /// tensor ids correctly while describing only the single-request graph's ROW COUNT — and
    /// [`Wiring::baked_row_capacity`], which reads the embedding placement, therefore answers 1 for
    /// a launch that is legitimately `B` rows wide.
    ///
    /// ⛔ THIS IS THE REGRESSION THAT BROKE BATCHED DECODE. Before the KTIR unification the wiring
    /// slot was last-write-wins, and because rungs bake ASCENDING it happened to hold the WIDEST
    /// rung's wiring — so a batched launch passed the check by accident. Tightening that slot to the
    /// single-request graph (a correct fix: `ktir_decode_cb_*` was stamping `m_cap = 96`, making
    /// every single-token decode compute 96 activation rows) left the batched path checking its
    /// width against a wiring that no longer describes it, and a 2-row launch became
    /// *"forward tape asked for 2 row(s) from a bundle baked to hold 1"*.
    ///
    /// Minted ONLY from a [`RungWidth`](scratchy_spyre_bundle::rung::RungWidth) — a width that
    /// came from a manifest's own `RungSeqs`, i.e. from the artifact rather than from a host guess —
    /// so this cannot become a door for waving an arbitrary integer past the guard.
    pub fn of_baked_rung(width: scratchy_spyre_bundle::rung::RungWidth) -> RowCapacity {
        RowCapacity(width.count())
    }
}

/// One layer's KV pools — the K and V cache tensors its paged attention gathers from and its KV
/// write scatters into (see [`PagedStep`]).
///
/// ⛔ THE WORKER USED TO REDISCOVER THIS BY SEARCHING. It walked every manifest node for one whose
/// args mentioned the mask tensor and inferred the rest positionally. Emitted directly, it is two
/// integers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LayerWiring {
    pub prefix_k: u32,
    pub prefix_v: u32,
}

/// Model geometry, baked.
///
/// ⛔ THE WORKER USED TO RE-READ THIS FROM `config.json` AT RUNTIME
/// (`num_attention_heads`, `rope_theta`, `head_dim`, `num_hidden_layers`), each
/// behind an `unwrap_or` default — a SECOND source of truth against the bundle
/// that was lowered from the compiler's own bounds. When the two disagreed the
/// bundle won silently and the output was garbage. Baked from the same bounds
/// the tape was lowered at, they cannot disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Geometry {
    pub hidden: u32,
    pub kv_dim: u32,
    pub vocab: u32,
    pub layers: u32,
    pub head_dim: u32,
    /// ⭐⭐⭐⭐⭐ THE QUERY-HEAD COUNT, baked — `num_attention_heads` from the same
    /// bounds the tape was lowered at.
    ///
    /// ⛔⛔⛔ NOT `hidden / head_dim`, WHICH IS WHAT EVERY HOST STAGING SITE USED TO
    /// DERIVE. That quotient is the head count only when the Q projection is SQUARE
    /// (`hidden == nqh·hd`) — true of granite/llama and false of gemma-4, whose
    /// `hidden` (3840) is not a multiple of any class's `nqh·hd` (16·256 = 4096,
    /// 16·512 = 8192). A host that stages the causal mask or the prefix mask tiled
    /// to `hidden/head_dim` head rows binds only that many of the baked `nqh·mq`
    /// rows — the bind guard refuses OVER-binds only, so the shortfall is silent,
    /// and the ops read the unstaged tail of an ADDITIVE mask as ZERO, i.e. as
    /// "no mask". MEASURED, gemma-4 tiny-allglobal (hidden 128, nqh 4, hd 64):
    /// the causal mask covered heads 0-1 and heads 2-3 ran with NO causal mask,
    /// every layer, prefill — card dump `sc` rows 22..43 unmasked, heads 0-1 exact.
    /// The fix is the count itself, baked here from the bounds, so the staged row
    /// count and the baked placement cannot disagree.
    pub nqh: u32,
    /// Rotary base, as bits — `f32` is not structurally-eq, so a `const`-able
    /// struct cannot hold one and still derive `Eq`. Read via [`Self::rope_theta`].
    pub rope_theta_bits: u32,
}

impl Geometry {
    #[inline]
    pub const fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }
}

/// ⭐ WHICH ROTARY TABLE A COS/SIN SOURCE CARRIES — the class of the layer(s) it
/// rotates. A uniform model has ONE class; a hybrid-attention arch (gemma-4:
/// sliding layers vs every-6th global layers) has TWO, and a table built from
/// the wrong one is silent garbage — the rotation multiplies Q/K by wrong
/// frequencies, which is fluent-but-wrong output, not a fault.
///
/// ⛔ NOT A `bool`. `local`/`global` are negations of each other and both are in
/// scope at every construction site — the exact shape that lets one land in the
/// other. The two classes here have DIFFERENT ARITY of parameters (the
/// proportional class carries its own rotated-dim count), which a bool cannot.
///
/// Every class's frequency law, from mlx `rope_utils.py` (the reference):
/// - `Default` — `freq_i = θ^(-2i/hd)` for the full head dim, NeoX pairing
///   (d, d+hd/2).
/// - `Proportional` — mlx `ProportionalRoPE`: `freq_i = θ^(-2i/hd)` for
///   `2i < rotated`, INFINITE past it (the pass-through lanes — identity, the
///   tail stays unrotated). The exponent's denominator is the FULL hd, not
///   `rotated`: that is the class's defining difference.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RotaryKind {
    /// The base class: full rotary at the class's own head dim and θ.
    Default {
        /// The class's head dim — the width of one head in THIS table's
        /// consumers (sliding: `head_dim`; a uniform model: `head_dim`).
        head_dim: u32,
        /// The class's rope base θ.
        theta_bits: u32,
    },
    /// mlx `ProportionalRoPE` — partial rotary whose exponent denominator is
    /// the FULL head dim (gemma-4 global layers: hd 512, rotated 128, θ 1e6).
    Proportional {
        head_dim: u32,
        theta_bits: u32,
        /// How many leading dims of the head actually rotate; the rest are
        /// pass-through (freq = ∞ ⇒ cos 1, sin 0).
        rotated: u32,
    },
}

impl RotaryKind {
    /// The class's head dim — the width one row of its table spans.
    pub const fn head_dim(self) -> u32 {
        match self {
            Self::Default { head_dim, .. } | Self::Proportional { head_dim, .. } => head_dim,
        }
    }

    /// The class's rope base θ, as bits (see [`Geometry::rope_theta_bits`]).
    pub const fn theta(self) -> f32 {
        match self {
            Self::Default { theta_bits, .. } | Self::Proportional { theta_bits, .. } => {
                f32::from_bits(theta_bits)
            }
        }
    }
}

/// ⭐ THE PAGED STEP — what a paged forward's host fills each step, and the pool it fills it against.
///
/// ONE BIND: which tensor, and the rows it gets — a computed activation owns its rows, a baked
/// table is borrowed straight into the bind.
pub type Bind = (
    scratchy_spyre_bundle::place::PlaceId,
    std::borrow::Cow<'static, [f32]>,
);

/// The attention reads its keys and values out of each layer's K and V POOLS (the cache tensors,
/// laid as the build's pool lays a page, [`scratchy_triton_splice::PageOrder`]), gathering the
/// pages a request's block table names;
/// the KV writers scatter the step's rows into them first. spyre-inference's own operands carry the
/// step (`crate::paged_step` builds each as their metadata builder does): the PAGE INDEX TABLE
/// (`[walk, 32]` int32, block `i`'s page at column 0), the MASK STACK (`[walk, rows, page_slots]`
/// additive, one tile per walked block), the KV WRITE INDEX (one `[rows]` int32 index per write
/// entry, [`scratchy_triton_splice::PageOrder::write_entries`]), the QUERY ROW INDEX (`[rows]`
/// int32); a
/// DECODE BATCH's attention reads `rep_row_ids`, `chunk_page_ids` and `mask_by_chunk` instead
/// ([`Self::paged_rows`], [`Self::batch`]). The pools themselves are never re-staged: the device writes them and they
/// persist across steps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PagedStep {
    pub mask: u32,
    pub page_table: u32,
    pub write_rows: u32,
    pub query_rows: u32,
    pub rep_rows: u32,
    pub chunk_pages: u32,
    pub chunk_masks: u32,
    /// Pages in each layer's pools — the scheduler's block count.
    pub pages: u32,
    /// Token slots per page — the scheduler's block size.
    pub page_slots: u32,
    /// Query rows one forward of this bundle carries.
    pub rows: u32,
    /// KV heads per layer.
    pub nkvh: u32,
    /// The query's row width, `num_heads · head_size` — what a decode batch's fold cuts a query
    /// row into whole sticks by ([`crate::paged_step::BatchChunking::of`]).
    pub q_width: u32,
    /// The head's width — whether a decode batch's fold may cut a page finer than its kv heads
    /// ([`crate::paged_step::BatchChunking::of`]).
    pub head_size: u32,
    /// Blocks of the page table this bundle's attention walks — the bake's
    /// `scratchy_triton_splice::walk_pages` for the rung, so at most [`Self::pages`].
    pub walk: u32,
    /// What the rows are: one sequence's, or a decode batch's — one token of each of `rows`
    /// sequences, read by `batched_decode_head_major` ([`Self::batch`]).
    pub paged_rows: crate::paged_step::PagedRows,
    /// ⭐ THE QUERY'S EXTENT PAIR — the rung's own `(per-request width, capacity)`, minted once
    /// at the bake from the rung (never a division of [`Self::rows`]) and threaded here so the
    /// card stager and the layout state the per-request laws off the ONE pair the splice already
    /// checks against the graph ([`QueryExtent`]'s own doc). The pair's held product IS
    /// `rows` — the two are one fact in two spellings, pinned at the one mint
    /// ([`crate::lower_subtile_tape_to_ktir::paged_step_of_ir`]'s agreement check) and maintained
    /// by [`Self::on_body`]'s own re-mint, so a `PagedStep` can never carry a pair its rows
    /// disagree with.
    pub extent: QueryExtent,
    /// ⭐ WHICH KEYS A ROW ATTENDS — the class's mask: every position up to the row's own, or (a
    /// hybrid model's local layers) only the last `window` of them. The host's mask stack states
    /// it; the kernels read it as data.
    pub attends: AttnMask,
}

pub use scratchy_ir::model_facts::AttnClass;
pub use scratchy_subtile::subtile_ir::AttnMask;

/// ⭐ THE PAGED STEP OF EACH ATTENTION CLASS — a uniform model's one class, a hybrid model's local
/// (sliding) class beside its full one: each class's step its own tables, kv heads, head dim and
/// mask (`scratchy_triton_splice::PagedStepTids::of`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct ClassSteps {
    pub full: Option<PagedStep>,
    pub sliding: Option<PagedStep>,
}

impl ClassSteps {
    /// The classes' steps the graph has, the full class first.
    pub fn iter(self) -> impl Iterator<Item = PagedStep> {
        self.full.into_iter().chain(self.sliding)
    }

    /// Class `class`'s step.
    pub const fn of(self, class: AttnClass) -> Option<PagedStep> {
        match class {
            AttnClass::Full => self.full,
            AttnClass::Sliding => self.sliding,
        }
    }

    /// The facts every class's step shares (rows, extent, slots, capacity): the first class's step.
    pub fn first(self) -> Option<PagedStep> {
        self.iter().next()
    }

    /// Every class's step mapped through `f`.
    pub fn map(self, f: impl Fn(PagedStep) -> Option<PagedStep>) -> Option<ClassSteps> {
        let each = |s: Option<PagedStep>| -> Option<Option<PagedStep>> {
            match s {
                Some(s) => f(s).map(Some),
                None => Some(None),
            }
        };
        Some(ClassSteps {
            full: each(self.full)?,
            sliding: each(self.sliding)?,
        })
    }

    /// The positions every class's walk reaches.
    pub fn reach(self) -> Option<usize> {
        self.iter().map(PagedStep::reach).min()
    }
}

impl PagedStep {
    /// The blocks the scheduler may hand out — every page but the null page
    /// ([`crate::paged_step::page_of_block`]).
    pub const fn host_blocks(self) -> u32 {
        self.pages - 1
    }

    /// The pool's token capacity: every page's slots.
    pub const fn capacity(self) -> usize {
        self.pages as usize * self.page_slots as usize
    }

    /// ⭐ THE BUNDLE'S REACH — the positions its attention's walk covers. A step whose positions end at
    /// or below this can run on the bundle from ANY start: the mask stack decides which walked keys each
    /// row attends, so a prompt chunk over a cached prefix is this bundle's to run whenever the prefix
    /// and the chunk both sit inside the walk. A step that ends past it would lose keys.
    pub const fn reach(self) -> usize {
        self.walk as usize * self.page_slots as usize
    }

    /// A decode batch's chunking of this bundle's walk (`batched_decode_chunking`), derived from the
    /// rows and the walk so a rung's walk cannot leave a stale chunking behind; `None` for one
    /// sequence's rows.
    pub const fn batch(self) -> Option<crate::paged_step::BatchChunking> {
        match self.paged_rows {
            crate::paged_step::PagedRows::Batch => Some(crate::paged_step::BatchChunking::of(
                scratchy_triton_splice::DecodeWalk {
                    seqs: self.rows,
                    blocks: self.walk,
                    pages: self.pages,
                },
                scratchy_triton_splice::DecodeHeads {
                    kv_heads: self.nkvh,
                    q_width: self.q_width,
                    head_size: self.head_size,
                },
            )),
            crate::paged_step::PagedRows::Sequence => None,
        }
    }

    /// ⭐ THE PAGE INDEX TABLE'S PITCH — the int32 words one block's entry takes: one int32 stick a
    /// block (spyre-inference's `[walk, INT32_ELEMS_PER_STICK]`, the page at column 0) wherever the
    /// walk takes one page a trip, one word a block (`[walk]`, contiguous) where a prompt's walk
    /// takes several ([`scratchy_triton_splice::trip_pages`]): a trip's pages are one gather's index.
    pub fn page_pitch(self) -> usize {
        let trips = matches!(self.paged_rows, crate::paged_step::PagedRows::Sequence)
            && self.rows > 1
            && scratchy_triton_splice::trip_pages(self.walk) > 1;
        if trips {
            1
        } else {
            crate::paged_step::INT32_ELEMS_PER_STICK
        }
    }

    /// The page index table's shape: `[walk, pitch]`, or `[walk]` at a pitch of one word.
    pub fn page_table_shape(self) -> Vec<usize> {
        match self.page_pitch() {
            1 => vec![self.walk as usize],
            pitch => vec![self.walk as usize, pitch],
        }
    }

    /// ⭐ THE STEP'S CARD PLACEMENTS — `(tid, seat, bytes)` for every source this bundle's attention
    /// reads, sized for the pool's whole walk (a body walking fewer blocks stages and reads a
    /// prefix). The one statement the bake places by (`compute_bundle_layout`) and the card stagers
    /// ([`crate::paged_step::card_step_sources`], [`crate::paged_step::card_batch_sources`],
    /// [`crate::paged_step::card_statics`]) stage within; `self` is the bundle's own step, at its
    /// whole walk.
    pub fn card_placements(self) -> Vec<(u32, CardSeat, u64)> {
        let int32_stick = u64::from(scratchy_triton_splice::INT32_STICK);
        let sticks = |words: u64| words.div_ceil(int32_stick) * int32_stick * 4;
        let slots = u64::from(self.page_slots);
        let mut v = vec![(
            self.write_rows,
            CardSeat::Step,
            sticks(u64::from(self.rows * self.nkvh)),
        )];
        match self.batch() {
            Some(c) => {
                let (chunks, entries) = (u64::from(c.num_chunks), u64::from(c.entries()));
                v.extend([
                    (
                        self.rep_rows,
                        CardSeat::Step,
                        sticks(u64::from(c.index_rows())),
                    ),
                    (
                        self.chunk_pages,
                        CardSeat::Step,
                        sticks(chunks * u64::from(c.index_rows())),
                    ),
                    (
                        self.chunk_masks,
                        CardSeat::Step,
                        chunks * entries * slots * 2,
                    ),
                ]);
            }
            None => {
                // ⭐ ONE ROW PER WINDOW, AT ITS OWN WINDOW TID — window `r`'s three tables (page
                // table, mask stack, query rows) at
                // [`scratchy_spyre_bundle::reserved_tids::paged_window_tid`] `r`, each its own region
                // sized the ONE-REQUEST size (the pool's whole walk, as ever), the windows
                // packed in window order after the write rows. Window 0's tids ARE the step's
                // own sentinels, so at one request the rows are exactly today's — same tids,
                // same sizes, same order, byte for byte.
                let w = u64::from(self.extent.per_request().get());
                let windows = u64::from(self.extent.requests().get());
                // At either pitch: a rung of the same pool may walk at the other.
                let walk = u64::from(self.pages.max(self.walk));
                let table = walk * crate::paged_step::INT32_ELEMS_PER_STICK as u64 * 4;
                let class = scratchy_triton_splice::paged_class(self.attends.class());
                let window_tid = |r: u64, table: u32| {
                    scratchy_spyre_bundle::reserved_tids::paged_window_tid(class, r as u32, table)
                };
                for r in 0..windows {
                    v.extend([
                        (window_tid(r, 1), CardSeat::Step, table),
                        (window_tid(r, 2), CardSeat::Step, walk * w * slots * 2),
                        (window_tid(r, 0), CardSeat::Step, sticks(w)),
                    ]);
                }
            }
        }
        v
    }

    /// ⭐ THE STEP OF THE BODY THAT RUNS — this bundle's pool and tids at a rung's own rows (a batch or
    /// prompt-width rung is the same tape baked at `rows` rows), walking the blocks the body sweeps:
    /// `swept` slots, whole pages for every body (a `walk_buckets` rung, or the ceiling's walk).
    /// `None` when `swept` is no walk of this pool.
    ///
    /// ⭐ THE EXTENT IS THE LAUNCH'S OWN PAIR, by the launch's own kind — never a comparison
    /// against the wiring's stated pair. The wiring states ONE representative of a tape the
    /// ladders bake at several widths (the prefill ladder's rungs all bind the same tensor ids,
    /// so one wiring serves them; `rows` here is the rung actually launching), and the runtime
    /// truth is what the launch itself is: a `Sequence` launch is ONE request of `rows` rows —
    /// `(rows, 1)` — and a `Batch` launch is one row per request — `(1, rows)`. Both are the
    /// same law the bake's codegen minted each rung's own pair by, so a launch on the
    /// representative's own width re-derives the wiring's pair exactly.
    pub fn on_body(
        self,
        rows: u32,
        paged_rows: crate::paged_step::PagedRows,
        swept: u32,
    ) -> Option<PagedStep> {
        let pages = swept / self.page_slots;
        let walk = scratchy_triton_splice::whole_trips(pages);
        let extent = match paged_rows {
            crate::paged_step::PagedRows::Sequence => {
                QueryExtent::of_baked_rung(RungWidth::of_baked_rows(rows)?, RungCap::ONE)?
            }
            crate::paged_step::PagedRows::Batch => {
                QueryExtent::of_baked_rung(RungWidth::ONE, RungCap::of_baked_seqs(rows)?)?
            }
        };
        (rows > 0 && pages > 0 && swept.is_multiple_of(self.page_slots) && pages <= self.pages)
            .then_some(PagedStep {
                rows,
                paged_rows,
                walk,
                extent,
                ..self
            })
    }

    /// ⭐ THE PACKED LAUNCH'S EXTENT — the wiring's OWN stated pair, kept verbatim.
    ///
    /// A packed rung's wiring is not a representative of a tape baked at several widths: the
    /// bake minted the pair `(w, s)` for THIS rung ([`QueryExtent`]'s own law), and a wave
    /// launch IS that pair — `s` requests of `w` rows each, neither the one request of
    /// [`Self::on_body`]'s `Sequence` law nor the one-row-per-request of its `Batch` law, so
    /// there is no pair to re-derive from the launch's rows. Only the walk is the launch's to
    /// derive, from the sweep of the body the session selected — exactly `on_body`'s own law.
    pub fn on_body_packed(self, swept: u32) -> Option<PagedStep> {
        let pages = swept / self.page_slots;
        (self.extent.requests().get() > 1
            && pages > 0
            && swept.is_multiple_of(self.page_slots)
            && pages <= self.pages)
            .then_some(PagedStep {
                walk: scratchy_triton_splice::whole_trips(pages),
                ..self
            })
    }
}

/// Where a paged step's card source sits ([`PagedStep::card_placements`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardSeat {
    /// Beside the activations: staged every step, and moved by a walk's relaunch.
    Step,
    /// In the segment NEITHER relaunch shifts — not a layer's (which moves the weight and KV
    /// segments to that layer's copies) and not a walk's (which moves the step's tables): staged once,
    /// at load, and read at its placement by every layer and every trip.
    Static,
}

/// Everything the worker needs to launch one baked bundle, emitted whole.
pub struct Wiring {
    /// Tensor id of the logits.
    pub result: u32,
    /// ⭐ THE RESULT'S OWN DTYPE — the fact that decides how the worker reads the
    /// rung's answer back ([`Self::result_is_index`] is the one question asked of
    /// it). Stated by the result producer's own kernel signature when a spliced
    /// kernel produced the result, and the f16 row every native lm-head tail
    /// emits otherwise.
    ///
    /// ⛔ THE ROW REMAINS A PLACED TENSOR, READABLE BY TID, when the result is an
    /// index stick: the answer re-point is the greedy contract's redirect of the
    /// ANSWER (the host reads the picked token, not the row it was picked from),
    /// never a deletion of the row. Any consumer that needs the row can still
    /// name its tid and read it; the mint widens it, places it, and leaves every
    /// writer exactly where it was.
    pub result_dtype: scratchy_tensors::DType,
    /// Tensor ids `0..num_sources` are caller-filled.
    pub num_sources: u32,
    /// The AttnDecode runtime length mask, if the model has a maskable decode.
    pub attn_mask: Option<u32>,
    /// Each attention class's paged step — see [`ClassSteps`]. Empty for a graph with no attention.
    pub paged: ClassSteps,
    /// Baked `valid_len - 1`; a worker overrides per generated token.
    pub decode_position: u32,
    /// `[rows, cols]` per tensor id.
    pub tensor_shapes: &'static [(u32, u32)],
    /// The host-gathered hidden row's source id.
    ///
    /// ⛔ THE ANSWER, NOT THE DATA TO COMPUTE IT FROM. This used to be a
    /// `&[Source]` role array that the worker SCANNED at every model load
    /// (`.position(|s| matches!(s, EmbeddedHidden))`) — a compile-time fact
    /// re-derived at runtime, which is the same tell as the JSON `role`
    /// strings this replaced, just with a nicer type. The roles are still an
    /// exhaustive enum where they belong: in the macro, over `SourceBinding`.
    pub embed_src: u32,
    /// Rotary `cos` / `sin` sources as `(id, full column width, rotary class)`.
    /// GQA gives more than one width (Q vs K), so this is a list, not a pair;
    /// a hybrid-attention arch (gemma-4) additionally gives CLASSES different
    /// rotary tables, so every source carries its own [`RotaryKind`].
    pub cos_srcs: &'static [(u32, u32, RotaryKind)],
    pub sin_srcs: &'static [(u32, u32, RotaryKind)],
    /// Per-layer AttnDecode wiring, in layer order.
    pub layers: &'static [LayerWiring],
    pub geometry: Geometry,
    /// The ScalarMul weight folds the baked programs carry: `(tensor id, multiplier)` for each
    /// matmul weight the worker must scale at load — `(x·W)·s == x·(W·s)`, so the folded
    /// multiplier's device op is gone and the scale rides the staged weight bytes instead
    /// ([`crate::spyre_load`]'s `stage_bound_weights` applies it, the one runtime half of the fold).
    ///
    /// ⛔ THE MULTIPLY IS POST-STAGING, PRE-CARD — the same staged fp16 bytes the un-folded bundle
    /// would have shipped, times the multiplier, once, at load. Not a per-forward cost and not a
    /// requantize: dense weights scale in place, fp8 dequant rows scale their `ws` row.
    pub weight_scale_folds: &'static [(u32, f32)],
    /// ⭐ THE LOAD-TIME CONSTANTS — `(tid, kind)` for every resident buffer the graph's own
    /// producer minted below the `SourceBinding` layer: staged ONCE at load through the kind's
    /// builder ([`LoadConst::build`]), never a per-step source, never baked as literals. A second
    /// spliced kernel with staged tables adds one [`LoadConst`] kind and rides this same field. The
    /// ids are the graph's own tensor ids (the ids the placement places).
    pub load_consts: &'static [(u32, LoadConst)],
    /// ⭐ THE SAMPLER'S PER-STEP SEAT — `Some` exactly when the rung carries the token draw (the
    /// minted `Sample` node), naming the sources the draw reads FRESH EVERY STEP. The seat's tids
    /// count as filled only when bound THIS step: a held seed row is a held token.
    pub sample_seat: Option<SampleSeat>,
    /// ⭐ THE FORWARD'S OWN PHASE — projected into every [`crate::forward_tape::ForwardShape`]
    /// this wiring makes. `Decode` carries the chained head's sticks ([`ChainWiring`], the mint's
    /// payload); `Prefill` is every unchained program.
    ///
    /// ⛔ THIS ONE FIELD IS THE WORKER'S WHOLE CONTRACT with the chain: on `Decode`, the per-step
    /// embed-row push is REPLACED by the sticks' seed — the gather reads the sticks, never a
    /// host-pushed row. A decode wiring carries `Decode` BY CONSTRUCTION: the expansion's mint
    /// refuses any decode rung that would not chain.
    pub phase: crate::forward_tape::ForwardPhase,
}

/// The chained head's stick sources — the persistent state the mint threaded between steps
/// ([`Wiring::phase`]'s `Decode` payload). `s_src`/`l_src` are the sources the `ChainStage` WRITES
/// each step and the `EmbedRowGather` READS the next: the slab words and the lane one-hot.
/// `replaces` is the row the chain REPLACED — the caller-filled `[rows, hidden]` source whose host
/// filler the chain removed, read by the layout stamp's resident excuse and the bake's binding
/// check; the worker never names it. `block` is the table columns one rank's gather reads (the
/// mint's own `block`): with the rank a session plays as, it is the rank's
/// [`scratchy_subtile::subtile_ir::ChainBlock`] ([`chain_block`]) — the one fact the tables and the
/// seeds both decompose a pick through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainWiring {
    pub s_src: u32,
    pub l_src: u32,
    pub replaces: u32,
    pub block: u32,
}

/// ⭐ THE BLOCK A SESSION'S CHAINED HEAD GATHERS — `width` columns at the origin of the rank `me`
/// plays as; the whole table for a session that plays no rank of a sharded world (`None`). Bound
/// at load from the session's own seat, never baked: one bake serves every rank, and the rank
/// lives only in the bytes each rank stages (its tables, its seeds).
pub fn chain_block<T: crate::bundle_code::collective::World>(
    width: u32,
    me: Option<crate::bundle_code::collective::Rank<T>>,
) -> scratchy_subtile::subtile_ir::ChainBlock {
    use scratchy_subtile::subtile_ir::ChainBlock;
    me.map_or_else(
        || ChainBlock::whole(width),
        |r| ChainBlock::of_rank(width, r),
    )
}

/// ⭐ THE GRAPH'S OWN CHAIN FACT — read once off the mint by
/// [`crate::lower_subtile_tape_to_ktir::chain_of_ir`], the one derivation. `Chained` carries the
/// payload BY TYPE; `Unchained` is every graph the mint did not touch (prefill walks, the emulator
/// tapes of untied models) — the graph's own state, never a silent fallback.
#[derive(Clone, Copy, Debug)]
pub enum ChainFact {
    Chained { chain: ChainWiring },
    Unchained,
}

impl ChainFact {
    /// The forward phase this fact projects to: a chained graph is a decode.
    pub const fn phase(self) -> crate::forward_tape::ForwardPhase {
        match self {
            ChainFact::Chained { chain } => crate::forward_tape::ForwardPhase::Decode { chain },
            ChainFact::Unchained => crate::forward_tape::ForwardPhase::Prefill,
        }
    }
}

/// ⭐ A LOAD-TIME CONSTANT'S KIND — the shared builder that makes its bytes. A kind names a
/// BUILDER ([`Self::build`]), never a consumer: everything downstream (the bake's numeric gate, the
/// worker's load staging) iterates [`Wiring::load_consts`] op-blind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadConst {
    /// One of the sampler's twenty operand tables, 1-based in the kernel's operand order past the
    /// row, the seed words and the temperature (which are never load constants). Bytes from
    /// [`sample_staged_tables`] — the one derivation every stager reads.
    SampleTable {
        vocab: u32,
        /// The rung's own row count — the mint's `rows` fact: the uniform `[rows, 1]` tables stage
        /// one word per row.
        rows: u32,
        operand: u8,
        /// ⭐ THE BAKE'S OWN ARRANGEMENT FACT — the table a widened (`dl16tofp32`, lane-permuted)
        /// value reads lane-for-lane is staged ALREADY permuted
        /// ([`spyre_superdsc::arrangement::staged_arrangements`] decided it off the lowered
        /// program); [`LoadConst::build`] is pure in it, so every stager stages the same bytes.
        arrangement: ElementArrangement,
    },
    /// One of the chained head's two decompose tables, 1-based over the stage's table inputs (the
    /// s table, then the l table). Bytes from [`chain_staged_tables`]. `table_rows` is the extent
    /// the mint stated; `block` the mint's own block width ([`ChainWiring::block`]) — the rank's
    /// origin is bound when the session stages the table ([`Self::build`]'s `me`).
    ChainStageTable {
        table_rows: u32,
        block: u32,
        operand: u8,
    },
}

impl LoadConst {
    /// The const's bytes — row-major and dense, the layout the kernel's own view of the table
    /// states — its true dtype and its `[rows, cols]` shape. ONE derivation per kind, shared by
    /// every stager. `me` is the rank the staging session plays as (`None` alone): the chain's
    /// tables are its rank's [`chain_block`]; every other kind is the same on every rank.
    pub fn build<T: crate::bundle_code::collective::World>(
        &self,
        me: Option<crate::bundle_code::collective::Rank<T>>,
    ) -> (Vec<u8>, scratchy_tensors::DType, [usize; 2]) {
        match *self {
            LoadConst::ChainStageTable {
                table_rows,
                block,
                operand,
            } => chain_staged_tables(table_rows, chain_block(block, me))
                .into_iter()
                .nth(usize::from(operand) - 1)
                .expect("the mint emits operand 1..=2 — the registry's own row"),
            LoadConst::SampleTable {
                vocab,
                rows,
                operand,
                arrangement,
            } => {
                let (bytes, dt, shape) = sample_staged_tables(vocab, rows)
                    .into_iter()
                    .nth(usize::from(operand) - 1)
                    .expect("the mint emits operand 1..=20 — the registry's own row");
                let bytes = match (dt, arrangement) {
                    (_, ElementArrangement::Standard) => bytes,
                    // The stagger permutes an f32 word stream — the widen's own output.
                    (scratchy_tensors::DType::F32, ElementArrangement::Dl16ToFp32) => {
                        let words: Vec<f32> = bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|w| f32::from_le_bytes(*w))
                            .collect();
                        spyre_superdsc::arrangement::stagger::<ktir_core::ddl_binds::Baked>(&words)
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect()
                    }
                    (other, ElementArrangement::Dl16ToFp32) => panic!(
                        "the sampler's table {operand} is {other:?}, but a widen's stagger permutes \
                         f32 words — the arrangement pass marked a table no widen reads beside"
                    ),
                };
                (bytes, dt, shape)
            }
        }
    }
}

/// ⭐ THE SAMPLER'S PER-STEP SEAT — the sources the token draw reads fresh every step, the mint's
/// own facts ([`crate::lower_subtile_tape_to_ktir::sample_seat_of_ir`]), never recomputed.
///
/// ⛔ COUNTED AS FILLED ONLY WHEN BOUND THIS STEP: the seed's step word is the step's own
/// freshness, so a decode launch without a fresh bind of every seat operand refuses loudly — a held
/// seed row is a held TOKEN.
#[derive(Clone, Copy, Debug)]
pub struct SampleSeat {
    /// The four `[rows, 1]` i32 word tables of the seed quadruple `(seed_lo, seed_hi, step,
    /// greedy_gate)`, in that order — one word per row, dense, as the kernel's view reads them. No
    /// row word: the hash key is request-owned alone, so a request draws the same noise in any row
    /// of any batched launch.
    pub seed_words: [u32; 4],
    /// The `[rows, 1]` f32 temperature reciprocal, staged every step with the seed words.
    pub invt: u32,
    /// The rung's own row count — of both per-row operands and of the draw's grid.
    pub rows: u32,
    /// ⭐ THE LOGITS ROW's tid — the Sample node's `inputs[0]`, kept placed and readable by tid
    /// when the mint re-points the rung's result at the idx stick.
    pub row: u32,
    /// The emitted vocab — the sentinel threshold the readback refuses at or past.
    pub vocab: u32,
    /// `sample_pad_vocab(vocab)` — the width the whole-row tables share.
    pub pad_vocab: u32,
    /// The idx stick's row stride in i32 words, as the kernel's own view of it states: the
    /// readback reads `rows · idx_row_words` words and row `r`'s pick at word `r · idx_row_words`.
    pub idx_row_words: u32,
}

/// ⭐ THE CHAINED HEAD'S TWO DECOMPOSE TABLES — one row per answerable pick `t` (the sampler's
/// sentinel the last), in the stage's operand order: the s table `[table_rows, CHAIN_IDX_BLOCK]`
/// int32, row `t` the slab of `t`'s column within `block` ([`chain_s_row`]); the l table
/// `[table_rows, SAMPLE_BLOCK]` fp16, row `t` the lane one-hot ([`chain_l_row`], the zero row for
/// a pick `block` does not hold). The one derivation every stager reads; the host seeds are the
/// same rows ([`chain_s_seed`], [`chain_l_seed`]), so a seed is idempotent with the device's own
/// writes.
///
/// [`chain_s_row`]: scratchy_subtile::subtile_ir::chain_s_row
/// [`chain_l_row`]: scratchy_subtile::subtile_ir::chain_l_row
/// [`chain_s_seed`]: scratchy_subtile::subtile_ir::chain_s_seed
/// [`chain_l_seed`]: scratchy_subtile::subtile_ir::chain_l_seed
pub fn chain_staged_tables(
    table_rows: u32,
    block: scratchy_subtile::subtile_ir::ChainBlock,
) -> Vec<(Vec<u8>, scratchy_tensors::DType, [usize; 2])> {
    use scratchy_subtile::subtile_ir::{CHAIN_IDX_BLOCK, SAMPLE_BLOCK, chain_l_row, chain_s_row};
    let rows = table_rows as usize;
    vec![
        (
            (0..table_rows)
                .flat_map(|t| chain_s_row(t, block))
                .flat_map(u32::to_le_bytes)
                .collect(),
            scratchy_tensors::DType::I32,
            [rows, CHAIN_IDX_BLOCK as usize],
        ),
        (
            (0..table_rows)
                .flat_map(|t| chain_l_row(t, block))
                .flat_map(|w| half::f16::from_f32(w as f32).to_le_bytes())
                .collect(),
            scratchy_tensors::DType::F16,
            [rows, SAMPLE_BLOCK as usize],
        ),
    ]
}

/// ⭐ THE SAMPLER'S TWENTY STAGED TABLES — the kernel's own host contract
/// (`kernels/sample_fwd.py`'s docstring), stated ONCE for every stager: the bake-time numeric gate
/// and the worker's load-time staging read this same fn, so a table defect cannot hide on one leg
/// only. Functions of the vocab (and the rung's rows) alone, constant across steps, staged ONCE.
///
/// ⭐ THE SELECTION TABLES ARE F32 — the selection's own dtype: the score is a sum `x·inv_t + g`,
/// and an f16 compare would round near-ties into ties the host reference never sees.
///
/// In the kernel's OPERAND ORDER — `bramp, lramp, binc, linc, ones, sixtyfour, padoff, gramp,
/// ginc, hc1, hc2, hc3, c256, lane, fm8, ftop, fzero, fhalf, fm22, cmin` — each entry is
/// `(bytes, dtype, [rows, cols])`, row-major and dense:
///
/// * `bramp` — strictly DECREASING in the block index within each group and periodic across
///   groups (`GROUP_BLOCKS - (b % GROUP_BLOCKS)`), so the second `max` lands on the first block
///   holding the row's maximum inside the first maximal group. 0 on the pad lanes.
/// * `lramp` — strictly decreasing within each block (`BLOCK - l`), so the third `max` lands on
///   the first lane of that block.
/// * `binc` / `linc` — the INCREASING block-within-group / lane index tables the one-hot reads the
///   selected numbers back out through; on the pad lanes `GROUP_BLOCKS` / `BLOCK`, the sentinel's
///   own values.
/// * `ones` / `sixtyfour` — the `[1, BLOCK]` widen tables: the ones broadcast the selected numbers
///   to the output stick, the scale row carries the `·64` of the wide integer combine.
/// * `padoff` — THE PAD LAW: 0.0 on `[0, vocab)`, f32::MIN (−FLT_MAX) on the pads — the fp32
///   reduce template's own "negInf" (`summeanmaxexx2_fp32.ddl:71`), never IEEE −inf (an
///   exponent-all-ones lane reads as NaN on the card and poisons the max). Added to the score
///   LAST, so a pad lane never wins whatever the row's pad bytes hold.
/// * `gramp` / `ginc` — the two-level ramp's outer level: decreasing in the group index
///   (`G - g`), 0 on the pads; and the increasing group index, `G` on the pads.
/// * `hc1` / `hc2` / `hc3` — the counter hash's multipliers `0x9E3779B1` / `0x85EBCA77` /
///   `0xC2B2AE3D` (odd, so a bijection mod 2^32), `[rows, 1]`.
/// * `c256` — the wrap-fold's 2^8, `[rows, 1]`.
/// * `lane` — the per-lane index `0..PAD_V-1` as f32 (exact below 2^24).
/// * `fm8` / `ftop` / `fzero` / `fhalf` / `fm22` — the fold's constants `2^-8`, `2^23 - 1`, `0`,
///   `0.5`, `2^-22`, `[rows, 1]` each.
/// * `cmin` — the fp16 island's clamp `2^-14`, the log's domain guard, `[rows, 1]`.
///
/// The SENTINEL (all-NaN row): the one-hot falls on the pad lanes, where the tables stage
/// `ginc = G`, `binc = GROUP_BLOCKS`, `linc = BLOCK`, so the idx is
/// [`scratchy_subtile::subtile_ir::sample_sentinel`], past the last real lane.
pub fn sample_staged_tables(
    vocab: u32,
    rows: u32,
) -> Vec<(Vec<u8>, scratchy_tensors::DType, [usize; 2])> {
    let pad_v = scratchy_subtile::subtile_ir::sample_pad_vocab(vocab) as usize;
    let block = scratchy_subtile::subtile_ir::SAMPLE_BLOCK as usize;
    let group_blocks = scratchy_subtile::subtile_ir::SAMPLE_GROUP_BLOCKS as usize;
    let groups = scratchy_subtile::subtile_ir::sample_groups(pad_v as u32) as usize;
    let v = vocab as usize;
    let rows = rows as usize;
    // A whole-row f32 table under a per-lane rule: lane `i < vocab` takes `real(i)`, a pad lane
    // takes `pad`.
    let f32_row = |real: &dyn Fn(usize) -> usize, pad: usize| -> Vec<u8> {
        (0..pad_v)
            .flat_map(|i| {
                let x = if i < v { real(i) } else { pad };
                (x as f32).to_le_bytes()
            })
            .collect()
    };
    // A uniform `[rows, 1]` table: the same word at every row.
    let i32_col = |x: i32| -> Vec<u8> { (0..rows).flat_map(|_| x.to_le_bytes()).collect() };
    let f32_col = |x: f32| -> Vec<u8> { (0..rows).flat_map(|_| x.to_le_bytes()).collect() };
    use scratchy_tensors::DType::{F16, F32, I32};
    vec![
        // bramp
        (
            f32_row(&|i| group_blocks - (i / block) % group_blocks, 0),
            F32,
            [1, pad_v],
        ),
        // lramp
        (f32_row(&|i| block - i % block, 0), F32, [1, pad_v]),
        // binc
        (
            f32_row(&|i| (i / block) % group_blocks, group_blocks),
            F32,
            [1, pad_v],
        ),
        // linc
        (f32_row(&|i| i % block, block), F32, [1, pad_v]),
        // ones
        (
            (0..block).flat_map(|_| 1.0f32.to_le_bytes()).collect(),
            F32,
            [1, block],
        ),
        // sixtyfour
        (
            (0..block).flat_map(|_| 64.0f32.to_le_bytes()).collect(),
            F32,
            [1, block],
        ),
        // padoff
        (
            (0..pad_v)
                .flat_map(|i| (if i < v { 0.0 } else { f32::MIN }).to_le_bytes())
                .collect(),
            F32,
            [1, pad_v],
        ),
        // gramp
        (
            f32_row(&|i| groups - i / (block * group_blocks), 0),
            F32,
            [1, pad_v],
        ),
        // ginc
        (
            f32_row(&|i| i / (block * group_blocks), groups),
            F32,
            [1, pad_v],
        ),
        // hc1 / hc2 / hc3
        (i32_col(-1640531535), I32, [rows, 1]),
        (i32_col(-2048144777), I32, [rows, 1]),
        (i32_col(-1028477379), I32, [rows, 1]),
        // c256
        (i32_col(256), I32, [rows, 1]),
        // lane
        (
            (0..pad_v).flat_map(|i| (i as f32).to_le_bytes()).collect(),
            F32,
            [1, pad_v],
        ),
        // fm8 / ftop / fzero / fhalf / fm22
        (f32_col(0.00390625), F32, [rows, 1]),
        (f32_col(8388607.0), F32, [rows, 1]),
        (f32_col(0.0), F32, [rows, 1]),
        (f32_col(0.5), F32, [rows, 1]),
        (f32_col(2f32.powi(-22)), F32, [rows, 1]),
        // cmin
        (
            (0..rows)
                .flat_map(|_| half::f16::from_f32(2f32.powi(-14)).to_le_bytes())
                .collect(),
            F16,
            [rows, 1],
        ),
    ]
}

/// The per-model wiring set: one [`Wiring`] per baked PROGRAM.
///
/// ⛔ NOT ONE WIRING. Tensor ids are numbered per program, so the prefill graph
/// has its own — binding decode ids into a prefill launch is silent corruption,
/// not an error. The worker selects by phase.
pub struct Wirings {
    pub decode: Wiring,
    /// `None` when the model bakes no batched-prefill program.
    pub prefill: Option<Wiring>,
    /// ⭐ THE PACKED PREFIX RUNGS' OWN WIRINGS `[(width, seqs, Wiring)]` — one per
    /// `(width, seqs)` pair the bake shipped (`PagedKvPool::packed_prefix_rungs`'s table), each
    /// numbered by ITS OWN graph's tensor ids and carrying its own staged constants: a packed
    /// rung's graph is lowered at `width · seqs` rows with its sampler folded to `seqs` rows, so
    /// neither the solo prefill wiring nor the decode one can stage it. The worker picks the
    /// entry whose pair matches the rung it loaded. EMPTY ⇒ no packed rungs baked (every model
    /// whose ladder bakes none, and every build before the packed table existed).
    pub packed_prefill: &'static [(u32, u32, Wiring)],
}

/// How each rank loads a source — the shared analysis's slice law, re-exported so the tokens
/// `#[forward]` emits name one path and a loader slices through
/// `scratchy_subtile::tensor_parallel::slice_source`, the one implementation of it.
pub use scratchy_subtile::tensor_parallel::SourceShard;

/// ⭐ ONE SHARDED WORLD'S FORWARD, AS `#[forward]` BAKED IT — what a session spanning the cards of
/// world `T` loads instead of the one-rank wiring.
///
/// ⛔ THE WORLD IS THE TYPE. `T` is the build's one `tp/N` world: a session for one world cannot be
/// handed another world's forward, because the two are different types — the shapes it stages, the
/// window its consumes read and the peers it sends to are all `T`'s. No value selects a world at
/// load.
pub struct TpForward<T: crate::bundle_code::collective::World> {
    /// The rank-local decode bundle — the same on every rank (one bake serves the world).
    pub decode: crate::manifest::KtirBundleData,
    /// The rank-local batched-prefill bundle, when the model bakes one.
    pub prefill: Option<crate::manifest::KtirBundleData>,
    /// The world's ladders, as the one-rank bundle group carries its own
    /// ([`crate::manifest::SengraphBundleGroup`]): the prefix-free prefill rungs, the decode batch
    /// rungs and the prefix-capable prefill rungs — every rung the one-rank world serves, baked for
    /// this world, so a sharded session serves every prompt length and live request count on its
    /// own world's bundles.
    pub prefill_rungs: &'static [(u32, &'static str)],
    pub decode_rungs: &'static [(
        crate::manifest::RungSeqs,
        crate::manifest::SweptCols,
        crate::manifest::BundleSentinel,
    )],
    pub prefill_prefix: &'static [(u32, u32, &'static str)],
    /// Their launch wirings, numbered by the rank-local forwards' own tensor ids.
    pub wirings: Wirings,
    /// Per source of the one-rank forward (same indices): the slice each rank loads.
    pub sources: &'static [SourceShard],
    /// The world every rank of which runs this forward.
    pub world: core::marker::PhantomData<T>,
}

/// Why rank `me`'s share of a source could not be cut from its checkpoint bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SliceRefusal {
    /// The sharded forward states no slice law for this source.
    NotASource { id: u32, sources: usize },
    /// The whole tensor's bytes are not its one-rank shape's.
    Size { id: u32, bytes: usize, want: usize },
    /// The element width has no byte-exact cut here.
    Element { id: u32, bytes: usize },
    /// The split axis does not divide by the world.
    Indivisible {
        id: u32,
        shard: SourceShard,
        rows: u32,
        cols: u32,
    },
    /// A head-block split of a gemm weight — a law the analysis gives caches, not weights.
    HeadSplitGemm { id: u32 },
}

impl std::fmt::Display for SliceRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotASource { id, sources } => write!(
                f,
                "source {id} has no slice law (the sharded forward states {sources})"
            ),
            Self::Size { id, bytes, want } => write!(
                f,
                "source {id} is {bytes} bytes on disk, but its one-rank shape spans {want}"
            ),
            Self::Element { id, bytes } => {
                write!(f, "source {id}: a {bytes}-byte element has no cut")
            }
            Self::Indivisible {
                id,
                shard,
                rows,
                cols,
            } => write!(
                f,
                "source {id} [{rows}, {cols}] does not split {shard:?} across the world"
            ),
            Self::HeadSplitGemm { id } => {
                write!(f, "source {id} is a gemm weight with a head-block split")
            }
        }
    }
}

impl std::error::Error for SliceRefusal {}

impl<T: crate::bundle_code::collective::World> TpForward<T> {
    /// ⭐ THE BYTES RANK `me` STAGES FOR SOURCE `id`: its share of the whole checkpoint tensor
    /// `full`, cut by the slice law the analysis derived for that source ([`Self::sources`]) —
    /// `scratchy_subtile::tensor_parallel::slice_source`, the one implementation of it. `full`
    /// is BORROWED: the cut reads the rank's share out of it and builds only that, so a viewed
    /// checkpoint (the mmap) is never materialized whole — a sharded leg once copied the whole
    /// model once per rank here (measured at tp/8: 8 full-model host copies, ~1 s of the boot).
    ///
    /// `whole` is the source's one-rank shape (the one-rank wiring's `tensor_shapes`) and `elem`
    /// its element width. A gemm weight sits on disk as `[n, k]`, the transpose of its logical
    /// `[k, n]`, so its row and column laws trade places on the bytes.
    ///
    /// # Errors
    /// [`SliceRefusal`], by name.
    pub fn slice_source_bytes(
        &self,
        me: crate::bundle_code::collective::Rank<T>,
        id: u32,
        full: &[u8],
        elem: usize,
        whole: (u32, u32),
        is_gemm: bool,
    ) -> Result<Vec<u8>, SliceRefusal> {
        let shard = *self
            .sources
            .get(id as usize)
            .ok_or(SliceRefusal::NotASource {
                id,
                sources: self.sources.len(),
            })?;
        cut_source_bytes::<T>(shard, me, id, full, elem, whole, is_gemm)
    }

    /// ⭐ EVERY RANK'S SHARE OF SOURCE `id` FROM ONE PASS OVER ITS BYTES — the all-ranks
    /// form of [`Self::slice_source_bytes`], the same slice law and the same refusals,
    /// cut by [`scratchy_subtile::tensor_parallel::slice_source_all_ranks`] so each
    /// source byte is read exactly once however wide the world is. The per-rank cuts each
    /// walk the whole source (`W` concurrent boots re-read every checkpoint page `W`
    /// times over — measured at tp/8 as the weights window's whole residual), and the
    /// ranks without the allocator's store slot had to materialize every whole tensor
    /// before cutting; this cut reads the mmap view once, in source order, and builds
    /// every rank's share from that single read. [`AllRanksCut::Whole`] and
    /// [`AllRanksCut::Partial`] name the shared-backing laws the caller serves from one
    /// handle instead of `W` copies.
    ///
    /// # Errors
    /// [`SliceRefusal`], by name.
    pub fn slice_source_bytes_all_ranks(
        &self,
        id: u32,
        full: &[u8],
        elem: usize,
        whole: (u32, u32),
        is_gemm: bool,
    ) -> Result<AllRanksCut, SliceRefusal> {
        let shard = *self
            .sources
            .get(id as usize)
            .ok_or(SliceRefusal::NotASource {
                id,
                sources: self.sources.len(),
            })?;
        cut_source_bytes_all_ranks::<T>(shard, id, full, elem, whole, is_gemm)
    }
}

/// [`TpForward::slice_source_bytes`] once the source's law is known.
fn cut_source_bytes<T: crate::bundle_code::collective::World>(
    shard: SourceShard,
    me: crate::bundle_code::collective::Rank<T>,
    id: u32,
    full: &[u8],
    elem: usize,
    whole: (u32, u32),
    is_gemm: bool,
) -> Result<Vec<u8>, SliceRefusal> {
    use scratchy_subtile::subtile_ir::SourceShape;
    use scratchy_subtile::tensor_parallel::slice_source;
    if shard == SourceShard::Whole {
        return Ok(full.to_vec());
    }
    let (rows, cols) = whole;
    let want = rows as usize * cols as usize * elem;
    if full.len() != want {
        return Err(SliceRefusal::Size {
            id,
            bytes: full.len(),
            want,
        });
    }
    let w = T::RANKS;
    let divides = match shard {
        SourceShard::Whole
        | SourceShard::Partial
        | SourceShard::ColBlocks { .. }
        | SourceShard::RowBlocks { .. } => true,
        SourceShard::Rows => rows.is_multiple_of(w),
        SourceShard::Cols => cols.is_multiple_of(w),
        SourceShard::HeadRows { heads } => {
            heads.get().is_multiple_of(w) && rows.is_multiple_of(heads.get())
        }
    };
    if !divides {
        return Err(SliceRefusal::Indivisible {
            id,
            shard,
            rows,
            cols,
        });
    }
    // The bytes' own layout: a gemm weight's disk order is the transpose of its logical one.
    let (shape, shard) = if is_gemm {
        let disk = SourceShape {
            rows: cols,
            cols: rows,
        };
        match shard {
            SourceShard::Rows => (disk, SourceShard::Cols),
            SourceShard::Cols => (disk, SourceShard::Rows),
            SourceShard::ColBlocks { block } => (disk, SourceShard::RowBlocks { block }),
            SourceShard::RowBlocks { block } => (disk, SourceShard::ColBlocks { block }),
            // Rank 0 holds the value and every other rank zeros, in either byte order.
            SourceShard::Whole | SourceShard::Partial => (disk, shard),
            SourceShard::HeadRows { .. } => return Err(SliceRefusal::HeadSplitGemm { id }),
        }
    } else {
        (SourceShape { rows, cols }, shard)
    };
    fn cut<const N: usize, T: crate::bundle_code::collective::World>(
        full: &[u8],
        shape: SourceShape,
        shard: SourceShard,
        rank: crate::bundle_code::collective::Rank<T>,
    ) -> Vec<u8> {
        let (elems, _) = full.as_chunks::<N>();
        slice_source(elems, shape, shard, rank).into_flattened()
    }
    match elem {
        1 => Ok(cut::<1, T>(full, shape, shard, me)),
        2 => Ok(cut::<2, T>(full, shape, shard, me)),
        4 => Ok(cut::<4, T>(full, shape, shard, me)),
        bytes => Err(SliceRefusal::Element { id, bytes }),
    }
}

/// ⭐ EVERY RANK'S SHARE OF A SOURCE, from ONE pass over its bytes — the all-ranks form
/// of [`SliceRefusal`]'s cut. `Whole` and `Partial` name the laws whose every-rank share
/// is the SAME bytes (all of it, or the value on rank 0 and zeros elsewhere), so the
/// caller can hand every rank one shared backing instead of the `W` copies the per-rank
/// cut builds; `Split` carries each rank's own buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllRanksCut {
    /// Every rank holds all of it — one backing serves every rank.
    Whole,
    /// Rank 0 holds the value and every other rank zeros (as many bytes as the source).
    Partial,
    /// `shares[r]` is rank `r`'s share.
    Split(Vec<Vec<u8>>),
}

/// [`TpForward::slice_source_bytes_all_ranks`] once the source's law is known.
fn cut_source_bytes_all_ranks<T: crate::bundle_code::collective::World>(
    shard: SourceShard,
    id: u32,
    full: &[u8],
    elem: usize,
    whole: (u32, u32),
    is_gemm: bool,
) -> Result<AllRanksCut, SliceRefusal> {
    use scratchy_subtile::subtile_ir::SourceShape;
    use scratchy_subtile::tensor_parallel::slice_source_all_ranks;
    if shard == SourceShard::Whole {
        return Ok(AllRanksCut::Whole);
    }
    let (rows, cols) = whole;
    let want = rows as usize * cols as usize * elem;
    if full.len() != want {
        return Err(SliceRefusal::Size {
            id,
            bytes: full.len(),
            want,
        });
    }
    let w = T::RANKS;
    let divides = match shard {
        SourceShard::Whole
        | SourceShard::Partial
        | SourceShard::ColBlocks { .. }
        | SourceShard::RowBlocks { .. } => true,
        SourceShard::Rows => rows.is_multiple_of(w),
        SourceShard::Cols => cols.is_multiple_of(w),
        SourceShard::HeadRows { heads } => {
            heads.get().is_multiple_of(w) && rows.is_multiple_of(heads.get())
        }
    };
    if !divides {
        return Err(SliceRefusal::Indivisible {
            id,
            shard,
            rows,
            cols,
        });
    }
    if shard == SourceShard::Partial {
        return Ok(AllRanksCut::Partial);
    }
    // The bytes' own layout: a gemm weight's disk order is the transpose of its logical
    // one — the same mapping the per-rank cut applies, rank-independent.
    let (shape, shard) = if is_gemm {
        let disk = SourceShape {
            rows: cols,
            cols: rows,
        };
        match shard {
            SourceShard::Rows => (disk, SourceShard::Cols),
            SourceShard::Cols => (disk, SourceShard::Rows),
            SourceShard::ColBlocks { block } => (disk, SourceShard::RowBlocks { block }),
            SourceShard::RowBlocks { block } => (disk, SourceShard::ColBlocks { block }),
            // The caller serves Whole and Partial from one shared backing, in either
            // byte order.
            SourceShard::Whole | SourceShard::Partial => (disk, shard),
            SourceShard::HeadRows { .. } => return Err(SliceRefusal::HeadSplitGemm { id }),
        }
    } else {
        (SourceShape { rows, cols }, shard)
    };
    fn cut<const N: usize, T: crate::bundle_code::collective::World>(
        full: &[u8],
        shape: SourceShape,
        shard: SourceShard,
    ) -> Vec<Vec<u8>> {
        let (elems, _) = full.as_chunks::<N>();
        slice_source_all_ranks::<[u8; N], T>(elems, shape, shard)
            .into_iter()
            .map(Vec::into_flattened)
            .collect()
    }
    let split = match elem {
        1 => cut::<1, T>(full, shape, shard),
        2 => cut::<2, T>(full, shape, shard),
        4 => cut::<4, T>(full, shape, shard),
        bytes => return Err(SliceRefusal::Element { id, bytes }),
    };
    Ok(AllRanksCut::Split(split))
}

impl Wiring {
    /// Element count of tensor `id`.
    #[inline]
    pub fn tensor_len(&self, id: u32) -> usize {
        let (r, c) = self.tensor_shapes[id as usize];
        r as usize * c as usize
    }

    /// The KV capacity the bundle was baked at — its pools' token slots.
    #[inline]
    pub fn capacity(&self) -> Option<usize> {
        self.paged.first().map(PagedStep::capacity)
    }
}

#[cfg(test)]
mod tp_slice_tests {
    use super::*;
    use crate::bundle_code::collective::{Rank, TpWorld, ranks};
    type W2 = TpWorld<2>;
    type W4 = TpWorld<4>;

    /// A gemm weight's bytes sit on disk as `[n, k]`: its K-split (`Rows`) takes a column block of
    /// those bytes and its N-split (`Cols`) a row block, each rank its own, and a `Whole` source
    /// passes through untouched — a multi-byte element moving as one.
    #[test]
    fn a_rank_stages_its_block_of_the_disk_bytes() {
        let (r0, r1) = (Rank::<W2>::of::<0>(), Rank::<W2>::of::<1>());
        // Logical [k=2, n=4]; on disk [n=4, k=2], element (n, k) = 10·n + k.
        let disk: Vec<u8> = (0..4u8)
            .flat_map(|n| (0..2u8).map(move |k| 10 * n + k))
            .collect();
        let cut =
            |shard, me| cut_source_bytes::<W2>(shard, me, 7, disk.as_slice(), 1, (2, 4), true);
        assert_eq!(cut(SourceShard::Rows, r0), Ok(vec![0, 10, 20, 30]));
        assert_eq!(cut(SourceShard::Rows, r1), Ok(vec![1, 11, 21, 31]));
        assert_eq!(cut(SourceShard::Cols, r0), Ok(vec![0, 1, 10, 11]));
        assert_eq!(cut(SourceShard::Cols, r1), Ok(vec![20, 21, 30, 31]));
        assert_eq!(cut(SourceShard::Whole, r1), Ok(disk.clone()));
        // A two-byte element moves as one: a norm row [1, 4] split by columns.
        let row: Vec<u8> = (0..8).collect();
        assert_eq!(
            cut_source_bytes::<W2>(SourceShard::Cols, r1, 3, row.as_slice(), 2, (1, 4), false),
            Ok(vec![4, 5, 6, 7])
        );
    }

    /// An uneven vocabulary's padded N-split (`ColBlocks`) is a row block of the disk bytes like
    /// the even one, the last rank's rows past the vocabulary staged zero — the columns its local
    /// projection computes and the all-gather crops.
    #[test]
    fn a_padded_block_stages_zeros_past_the_source() {
        let (r0, r1) = (Rank::<W2>::of::<0>(), Rank::<W2>::of::<1>());
        // Logical [k=2, n=3]; on disk [n=3, k=2], element (n, k) = 10·n + k.
        let disk: Vec<u8> = (0..3u8)
            .flat_map(|n| (0..2u8).map(move |k| 10 * n + k))
            .collect();
        let block = SourceShard::ColBlocks {
            block: std::num::NonZeroU32::new(2).unwrap(),
        };
        let cut = |me| cut_source_bytes::<W2>(block, me, 7, disk.as_slice(), 1, (2, 3), true);
        assert_eq!(cut(r0), Ok(vec![0, 1, 10, 11]));
        assert_eq!(cut(r1), Ok(vec![20, 21, 0, 0]));
    }

    /// Every cut it cannot make byte-exactly is refused by name.
    #[test]
    fn a_cut_it_cannot_make_is_refused() {
        let r0 = Rank::<W2>::of::<0>();
        assert_eq!(
            cut_source_bytes::<W2>(SourceShard::Cols, r0, 1, &[0; 6], 1, (2, 4), false),
            Err(SliceRefusal::Size {
                id: 1,
                bytes: 6,
                want: 8
            })
        );
        assert_eq!(
            cut_source_bytes::<W2>(SourceShard::Cols, r0, 1, &[0; 3], 1, (1, 3), false),
            Err(SliceRefusal::Indivisible {
                id: 1,
                shard: SourceShard::Cols,
                rows: 1,
                cols: 3
            })
        );
        assert_eq!(
            cut_source_bytes::<W2>(SourceShard::Rows, r0, 1, &[0; 24], 3, (2, 4), true),
            Err(SliceRefusal::Element { id: 1, bytes: 3 })
        );
        let heads = SourceShard::HeadRows {
            heads: std::num::NonZeroU32::new(2).unwrap(),
        };
        assert_eq!(
            cut_source_bytes::<W2>(heads, r0, 1, &[0; 8], 1, (2, 4), true),
            Err(SliceRefusal::HeadSplitGemm { id: 1 })
        );
    }

    /// The cut at a wider world: a `[2, 8]` norm row-pair split four ways by columns gives each of
    /// the four ranks its own two-column block, and a split that does not divide by four refuses.
    #[test]
    fn four_ranks_each_stage_their_block() {
        let row: Vec<u8> = (0..16).collect();
        let got: Vec<Vec<u8>> = ranks::<W4>()
            .map(|r| {
                cut_source_bytes::<W4>(SourceShard::Cols, r, 3, row.as_slice(), 1, (2, 8), false)
                    .unwrap()
            })
            .collect();
        assert_eq!(
            got,
            [
                vec![0, 1, 8, 9],
                vec![2, 3, 10, 11],
                vec![4, 5, 12, 13],
                vec![6, 7, 14, 15]
            ]
        );
        let r0 = Rank::<W4>::of::<0>();
        assert!(matches!(
            cut_source_bytes::<W4>(SourceShard::Cols, r0, 3, &[0; 12], 1, (2, 6), false),
            Err(SliceRefusal::Indivisible { .. })
        ));
    }

    /// ⭐ THE ALL-RANKS CUT IS THE PER-RANK CUT, at the bytes, over the same fixtures the
    /// per-rank law is pinned on — every law, both orientations, both element widths,
    /// every world under test. The one-pass scatter cut may change HOW the source is
    /// read, never WHICH bytes a rank holds.
    #[test]
    fn the_all_ranks_cut_is_the_per_rank_cut_at_the_bytes() {
        /// One law over one source: its bytes, element width, one-rank shape, whether it
        /// is a gemm weight (disk order transposed), and the law itself.
        struct Case<'a> {
            bytes: &'a [u8],
            elem: usize,
            whole: (u32, u32),
            is_gemm: bool,
            shard: SourceShard,
        }
        // Logical [k=2, n=4] gemm (disk [n=4, k=2]), a [1, 4] two-byte norm row, a
        // padded [k=2, n=3] gemm (ColBlocks/RowBlocks), and a [2, 8] row-pair.
        let disk: Vec<u8> = (0..4u8)
            .flat_map(|n| (0..2u8).map(move |k| 10 * n + k))
            .collect();
        let row: Vec<u8> = (0..8).collect();
        let padded: Vec<u8> = (0..3u8)
            .flat_map(|n| (0..2u8).map(move |k| 10 * n + k))
            .collect();
        let wide: Vec<u8> = (0..16).collect();
        let block = SourceShard::ColBlocks {
            block: std::num::NonZeroU32::new(2).unwrap(),
        };
        let heads = SourceShard::HeadRows {
            heads: std::num::NonZeroU32::new(2).unwrap(),
        };
        let cases = [
            Case {
                bytes: &disk,
                elem: 1,
                whole: (2, 4),
                is_gemm: true,
                shard: SourceShard::Rows,
            },
            Case {
                bytes: &disk,
                elem: 1,
                whole: (2, 4),
                is_gemm: true,
                shard: SourceShard::Cols,
            },
            Case {
                bytes: &disk,
                elem: 1,
                whole: (2, 4),
                is_gemm: true,
                shard: SourceShard::Whole,
            },
            Case {
                bytes: &row,
                elem: 2,
                whole: (1, 4),
                is_gemm: false,
                shard: SourceShard::Cols,
            },
            Case {
                bytes: &row,
                elem: 2,
                whole: (1, 4),
                is_gemm: false,
                shard: SourceShard::Whole,
            },
            Case {
                bytes: &padded,
                elem: 1,
                whole: (2, 3),
                is_gemm: true,
                shard: block,
            },
            Case {
                bytes: &padded,
                elem: 1,
                whole: (2, 3),
                is_gemm: true,
                shard: SourceShard::RowBlocks {
                    block: std::num::NonZeroU32::new(2).unwrap(),
                },
            },
            Case {
                bytes: &wide,
                elem: 1,
                whole: (2, 8),
                is_gemm: false,
                shard: SourceShard::Cols,
            },
            Case {
                bytes: &wide,
                elem: 1,
                whole: (2, 8),
                is_gemm: false,
                shard: SourceShard::Rows,
            },
            Case {
                bytes: &wide,
                elem: 1,
                whole: (2, 8),
                is_gemm: false,
                shard: SourceShard::Partial,
            },
            Case {
                bytes: &wide,
                elem: 1,
                whole: (4, 4),
                is_gemm: false,
                shard: heads,
            },
        ];
        for Case {
            bytes,
            elem,
            whole,
            is_gemm,
            shard,
        } in cases
        {
            macro_rules! pin {
                ($t:ty) => {{
                    let all =
                        cut_source_bytes_all_ranks::<$t>(shard, 7, bytes, elem, whole, is_gemm);
                    for r in ranks::<$t>() {
                        let per = cut_source_bytes::<$t>(shard, r, 7, bytes, elem, whole, is_gemm);
                        let got = match &all {
                            Ok(AllRanksCut::Split(v)) => Ok(v[r.get() as usize].clone()),
                            Ok(AllRanksCut::Whole) => Ok(bytes.to_vec()),
                            Ok(AllRanksCut::Partial) => Ok(if r.get() == 0 {
                                bytes.to_vec()
                            } else {
                                vec![0u8; bytes.len()]
                            }),
                            Err(e) => Err(e.clone()),
                        };
                        assert_eq!(
                            got,
                            per,
                            "{shard:?} elem {elem} whole {whole:?} gemm {is_gemm}: rank {}",
                            r.get()
                        );
                    }
                }};
            }
            pin!(W2);
            pin!(W4);
        }
    }
}
