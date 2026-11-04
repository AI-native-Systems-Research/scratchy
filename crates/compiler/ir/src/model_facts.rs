// SPDX-License-Identifier: Apache-2.0
//! ⭐ A CANONICAL'S MODEL FACTS, AS ONE TYPED VALUE.
//!
//! Every quantity is a newtype with a private field: a query-head count is not a kv-head count, and
//! an epsilon is not a scale. Its `const fn` constructor checks its law, and `get()` is the only way
//! out. A fact a model may lack is an `Option` of its group (a body without attention has no
//! [`Attention`]), never a `0`, `-1` or `0.0` standing in for "absent".
//!
//! The fill derives a [`ModelFacts`] once per canonical, from its config, at the canonical's
//! tensor-parallel width (heads sharded). The bake reads that value, so a config that breaks a law
//! fails the build in the constructor that states it.

pub use crate::KvCodec;
pub use scratchy_tensors::{MetalDtype, ScaleDtype};

/// A positive count or extent: the constructor refuses zero.
macro_rules! positive {
    ($($(#[$m:meta])* $name:ident($t:ty)),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name($t);
        impl $name {
            /// The quantity, refused at zero.
            pub const fn new(v: $t) -> Self {
                assert!(v > 0, concat!(stringify!($name), " is a positive quantity"));
                Self(v)
            }
            /// The quantity as a number.
            pub const fn get(self) -> $t {
                self.0
            }
        }
    )*};
}

positive! {
    /// Query heads (`num_attention_heads`, sharded).
    QueryHeads(u32),
    /// Key/value heads (`num_key_value_heads`, sharded).
    KvHeads(u32),
    /// One head's width.
    HeadDim(u32),
    /// The leading lanes of a head that rotary rotates.
    RotDim(u32),
    /// Positions per paged-KV block.
    BlockSize(u32),
    /// The residual stream's width.
    HiddenSize(usize),
    /// The feed-forward's inner width.
    IntermediateSize(usize),
    /// The lm_head's rows.
    VocabSize(usize),
    /// The positions a sliding-window layer attends to.
    AttnWindow(u32),
    /// The drafts each sequence of a verify step carries.
    NumDrafts(u32),
    /// The blocks of rows a target captures for its DFlash2 drafter.
    CaptureSlots(u32),
    /// The channels each dynamic-convolution weight of a DFlash2 drafter covers.
    ConvGroup(u32),
    /// The rank of a DFlash2 drafter's selector codebooks.
    SelectorRank(u32),
    /// A Gated-DeltaNet head count (key or value side).
    GdnHeads(u32),
    /// A Gated-DeltaNet head's width (key or value side).
    GdnHeadDim(u32),
    /// The Gated-DeltaNet short convolution's width.
    ConvKernel(u32),
    /// The Gated-DeltaNet convolution's channels.
    ConvDim(usize),
    /// A vision tower's patch grid side (square).
    PatchGridSide(u32),
    /// A vision tower's post-encoder average pool kernel.
    PoolKernel(u32),
    /// The input width of a vision tower's patch embedding.
    InFeatures(usize),
}

/// A finite `f32`: the constructor refuses NaN and infinity, and, where `$positive`, a value not
/// above zero.
macro_rules! scalar {
    ($($(#[$m:meta])* $name:ident: $positive:literal),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name(f32);
        impl $name {
            /// The scalar, refused when not finite (or, for a positive quantity, not above zero).
            pub const fn new(v: f32) -> Self {
                assert!(v.is_finite(), concat!(stringify!($name), " is finite"));
                assert!(!$positive || v > 0.0, concat!(stringify!($name), " is positive"));
                Self(v)
            }
            /// The scalar as a number.
            pub const fn get(self) -> f32 {
                self.0
            }
        }
    )*};
}

scalar! {
    /// The attention softmax's scale.
    AttnScale: true,
    /// A tanh soft cap on logits.
    SoftCap: true,
    /// The RMSNorm epsilon.
    RmsNormEps: true,
    /// The offset added to every norm's gain at run time (zero-centred checkpoints: 1).
    NormWeightOffset: false,
}

/// A token id: zero is a token like any other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TokenId(u32);

impl TokenId {
    /// The id.
    pub const fn new(v: u32) -> Self {
        Self(v)
    }
    /// The id as a number.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The KV page of a model whose config states none (`block_size`): the paged pool's default.
pub const DEFAULT_KV_PAGE: BlockSize = BlockSize::new(16);

/// The layers that hold a KV cache, when they are a proper subset of the decoder's (a hybrid's
/// attention layers among its recurrent ones). `'static` on both paths: the emitted const's table is
/// a literal, and the bake's is leaked once per canonical by the fill, in the build-script process.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvLayers(&'static [u32]);

impl KvLayers {
    /// The table, refused unless strictly ascending.
    pub const fn new(layers: &'static [u32]) -> Self {
        let mut i = 1;
        while i < layers.len() {
            assert!(
                layers[i - 1] < layers[i],
                "KvLayers: the layers are strictly ascending"
            );
            i += 1;
        }
        Self(layers)
    }
    /// The layers.
    pub const fn get(self) -> &'static [u32] {
        self.0
    }
}

/// One attention class's head geometry. Its constructor divides the GQA group out once, after
/// proving the division exact, so no reader re-checks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadGeometry {
    q_heads: QueryHeads,
    kv_heads: KvHeads,
    head_dim: HeadDim,
    rot_dim: RotDim,
    page: BlockSize,
}

impl HeadGeometry {
    /// The geometry, refused when the kv heads do not divide the query heads (an attention over it
    /// would read another head's keys) or the rot window is wider than the head.
    pub const fn new(
        q_heads: QueryHeads,
        kv_heads: KvHeads,
        head_dim: HeadDim,
        rot_dim: RotDim,
        page: BlockSize,
    ) -> Self {
        assert!(
            q_heads.get().is_multiple_of(kv_heads.get()),
            "HeadGeometry: the kv heads divide the query heads"
        );
        assert!(
            rot_dim.get() <= head_dim.get(),
            "HeadGeometry: the rot window fits its head"
        );
        Self {
            q_heads,
            kv_heads,
            head_dim,
            rot_dim,
            page,
        }
    }
    /// The query heads.
    pub const fn q_heads(self) -> QueryHeads {
        self.q_heads
    }
    /// The kv heads.
    pub const fn kv_heads(self) -> KvHeads {
        self.kv_heads
    }
    /// One head's width.
    pub const fn head_dim(self) -> HeadDim {
        self.head_dim
    }
    /// The rot window.
    pub const fn rot_dim(self) -> RotDim {
        self.rot_dim
    }
    /// The class's KV page.
    pub const fn page(self) -> BlockSize {
        self.page
    }
    /// Query heads per kv head, exact by the constructor's law.
    pub const fn gqa(self) -> u32 {
        self.q_heads.get() / self.kv_heads.get()
    }
    /// `q_heads · head_dim`.
    pub const fn q_width(self) -> usize {
        self.q_heads.get() as usize * self.head_dim.get() as usize
    }
}

/// An MRoPE section: the rotary pairs the temporal, height and width bands take.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MropeSection([u32; 3]);

impl MropeSection {
    /// The section, as the config states it.
    pub const fn new(section: [u32; 3]) -> Self {
        Self(section)
    }
    /// The bands' pairs.
    pub const fn get(self) -> [u32; 3] {
        self.0
    }
    /// The pairs of all three bands: each a pair of rotated lanes.
    pub const fn pairs(self) -> u32 {
        self.0[0] + self.0[1] + self.0[2]
    }
}

/// Which attention class a layer is: a sliding-window layer reads the base class, a full layer the
/// global class (the same class on a uniform model).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttnClass {
    /// A sliding-window layer.
    Sliding,
    /// A full (causal) layer.
    Full,
}

/// How the full (global) class's rope pairs its lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopePairing {
    /// Within the rot window (NeoX, full or partial).
    Standard,
    /// Across the head's halves (Gemma 4's proportional rope).
    Proportional,
}

/// Where the keys are rotated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvRope {
    /// At the KV write.
    AtWrite,
    /// On each read of the cache.
    OnRead,
}

/// How an attention's keys and queries are rotated: the global class's lane pairing, where the
/// keys are rotated, and an MRoPE section when the rows' positions are three-dimensional.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rope {
    pairing: RopePairing,
    kv_rope: KvRope,
    mrope: Option<MropeSection>,
}

impl Rope {
    /// The rotary facts.
    pub const fn new(pairing: RopePairing, kv_rope: KvRope, mrope: Option<MropeSection>) -> Self {
        Self {
            pairing,
            kv_rope,
            mrope,
        }
    }
}

/// A decoder's attention: its base class (every layer's, or the sliding layers' on a hybrid) and
/// its global class (equal to the base on a uniform model).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attention {
    base: HeadGeometry,
    global: HeadGeometry,
    scale: AttnScale,
    softcap: Option<SoftCap>,
    window: Option<AttnWindow>,
    rope: Rope,
}

impl Attention {
    /// The attention, refused when its classes disagree on the query heads (every layer projects
    /// the same query heads; only the kv side differs by class), or when an MRoPE section does not
    /// fill each class's rot window (its bands' pairs are half the window's lanes).
    pub const fn new(
        base: HeadGeometry,
        global: HeadGeometry,
        scale: AttnScale,
        softcap: Option<SoftCap>,
        window: Option<AttnWindow>,
        rope: Rope,
    ) -> Self {
        assert!(
            base.q_heads.get() == global.q_heads.get(),
            "Attention: both classes project the same query heads"
        );
        if let Some(section) = rope.mrope {
            assert!(
                2 * section.pairs() == base.rot_dim.get()
                    && 2 * section.pairs() == global.rot_dim.get(),
                "Attention: an MRoPE section's pairs are half each class's rot window"
            );
        }
        Self {
            base,
            global,
            scale,
            softcap,
            window,
            rope,
        }
    }
    /// The base class.
    pub const fn base(self) -> HeadGeometry {
        self.base
    }
    /// The global class.
    pub const fn global(self) -> HeadGeometry {
        self.global
    }
    /// The class a layer of `class` reads.
    pub const fn class(self, class: AttnClass) -> HeadGeometry {
        match class {
            AttnClass::Sliding => self.base,
            AttnClass::Full => self.global,
        }
    }
    /// The query heads every layer projects.
    pub const fn q_heads(self) -> QueryHeads {
        self.base.q_heads
    }
    /// The softmax scale.
    pub const fn scale(self) -> AttnScale {
        self.scale
    }
    /// The tanh cap on the attention logits, for a model that caps them.
    pub const fn softcap(self) -> Option<SoftCap> {
        self.softcap
    }
    /// The sliding layers' window; `None` on a model without sliding layers.
    pub const fn window(self) -> Option<AttnWindow> {
        self.window
    }
    /// The global class's rope pairing.
    pub const fn pairing(self) -> RopePairing {
        self.rope.pairing
    }
    /// Where the keys are rotated.
    pub const fn kv_rope(self) -> KvRope {
        self.rope.kv_rope
    }
    /// The MRoPE section; `None` on a 1-D rope.
    pub const fn mrope(self) -> Option<MropeSection> {
        self.rope.mrope
    }
}

/// How a vision tower's 2-D rope pairs its lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisionRopePairing {
    /// Rotate-half (the Qwen-VL towers).
    RotateHalf,
    /// Adjacent pairs (MoonViT / LocateAnything).
    Interleaved,
}

/// A vision tower's encoder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vision {
    heads: QueryHeads,
    head_dim: HeadDim,
    in_features: InFeatures,
    pairing: VisionRopePairing,
    scale: AttnScale,
    patch_grid_side: Option<PatchGridSide>,
    pool_kernel: Option<PoolKernel>,
}

impl Vision {
    /// The tower.
    pub const fn new(
        heads: QueryHeads,
        head_dim: HeadDim,
        in_features: InFeatures,
        pairing: VisionRopePairing,
        scale: AttnScale,
        patch_grid_side: Option<PatchGridSide>,
        pool_kernel: Option<PoolKernel>,
    ) -> Self {
        Self {
            heads,
            head_dim,
            in_features,
            pairing,
            scale,
            patch_grid_side,
            pool_kernel,
        }
    }
    /// Its attention heads.
    pub const fn heads(self) -> QueryHeads {
        self.heads
    }
    /// One head's width.
    pub const fn head_dim(self) -> HeadDim {
        self.head_dim
    }
    /// `heads · head_dim`.
    pub const fn q_width(self) -> usize {
        self.heads.get() as usize * self.head_dim.get() as usize
    }
    /// Its patch embedding's input width.
    pub const fn in_features(self) -> InFeatures {
        self.in_features
    }
    /// Its 2-D rope pairing.
    pub const fn pairing(self) -> VisionRopePairing {
        self.pairing
    }
    /// Its softmax scale.
    pub const fn scale(self) -> AttnScale {
        self.scale
    }
    /// Its patch grid side, for a tower that pools over one.
    pub const fn patch_grid_side(self) -> Option<PatchGridSide> {
        self.patch_grid_side
    }
    /// Its post-encoder pool kernel, for a tower that pools.
    pub const fn pool_kernel(self) -> Option<PoolKernel> {
        self.pool_kernel
    }
}

/// Gated-DeltaNet (or Mamba-2) layers' geometry. Mamba-2 reads as the same geometry: its groups
/// are the key heads, its heads the value heads, its state size the key head's width and its head
/// dim the value head's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Gdn {
    k_heads: GdnHeads,
    v_heads: GdnHeads,
    k_dim: GdnHeadDim,
    v_dim: GdnHeadDim,
    conv_kernel: ConvKernel,
    conv_dim: ConvDim,
}

impl Gdn {
    /// The layers' geometry, refused unless the short convolution covers exactly its streams: the
    /// q and k streams (`2 · k_heads · k_dim`) beside the v stream (`v_heads · v_dim`). Mamba-2's
    /// conv covers its `x` (`heads · head_dim`) beside its B and C (`2 · groups · state`): the same
    /// law under the reading above.
    pub const fn new(
        k_heads: GdnHeads,
        v_heads: GdnHeads,
        k_dim: GdnHeadDim,
        v_dim: GdnHeadDim,
        conv_kernel: ConvKernel,
        conv_dim: ConvDim,
    ) -> Self {
        let streams = 2 * k_heads.get() as usize * k_dim.get() as usize
            + v_heads.get() as usize * v_dim.get() as usize;
        assert!(
            conv_dim.get() == streams,
            "Gdn: the convolution covers the q, k and v streams side by side"
        );
        Self {
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            conv_kernel,
            conv_dim,
        }
    }
    /// Key heads.
    pub const fn k_heads(self) -> GdnHeads {
        self.k_heads
    }
    /// Value heads.
    pub const fn v_heads(self) -> GdnHeads {
        self.v_heads
    }
    /// A key head's width.
    pub const fn k_dim(self) -> GdnHeadDim {
        self.k_dim
    }
    /// A value head's width.
    pub const fn v_dim(self) -> GdnHeadDim {
        self.v_dim
    }
    /// The short convolution's width.
    pub const fn conv_kernel(self) -> ConvKernel {
        self.conv_kernel
    }
    /// The convolution's channels.
    pub const fn conv_dim(self) -> ConvDim {
        self.conv_dim
    }
}

/// A DFlash2 drafter's own geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Drafter {
    conv_group: ConvGroup,
    selector_rank: SelectorRank,
    mask_token: TokenId,
}

impl Drafter {
    /// The drafter's geometry.
    pub const fn new(
        conv_group: ConvGroup,
        selector_rank: SelectorRank,
        mask_token: TokenId,
    ) -> Self {
        Self {
            conv_group,
            selector_rank,
            mask_token,
        }
    }
    /// Channels per dynamic-convolution weight.
    pub const fn conv_group(self) -> ConvGroup {
        self.conv_group
    }
    /// The selector codebooks' rank.
    pub const fn selector_rank(self) -> SelectorRank {
        self.selector_rank
    }
    /// The token its block's proposal rows embed.
    pub const fn mask_token(self) -> TokenId {
        self.mask_token
    }
}

/// How a head's target routes the rows it verifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetRouting {
    /// Through a dense feed-forward.
    Dense,
    /// Through experts.
    Experts,
}

/// What a head reads of the target it drafts for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    hidden: HiddenSize,
    routing: TargetRouting,
}

impl Target {
    /// The target, as its head reads it.
    pub const fn new(hidden: HiddenSize, routing: TargetRouting) -> Self {
        Self { hidden, routing }
    }
    /// The width of one of its hidden rows.
    pub const fn hidden(self) -> HiddenSize {
        self.hidden
    }
    /// How it routes the rows it verifies.
    pub const fn routing(self) -> TargetRouting {
        self.routing
    }
}

/// The parts a [`ModelFacts`] is built from: each group already law-checked by its own constructor.
/// [`ModelFacts::new`] checks the laws between them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FactsParts {
    pub metal_dtype: MetalDtype,
    pub scale_dtype: ScaleDtype,
    pub hidden: HiddenSize,
    pub intermediate: Option<IntermediateSize>,
    pub vocab: Option<VocabSize>,
    pub eps: RmsNormEps,
    pub norm_offset: NormWeightOffset,
    pub final_softcap: Option<SoftCap>,
    pub kv_codec: KvCodec,
    /// The layers holding KV, when not every layer does.
    pub kv_layers: Option<KvLayers>,
    pub attention: Option<Attention>,
    pub vision: Option<Vision>,
    pub gdn: Option<Gdn>,
    pub drafter: Option<Drafter>,
    pub spec_drafts: Option<NumDrafts>,
    pub capture_slots: Option<CaptureSlots>,
    pub target: Option<Target>,
}

/// ⭐ ONE CANONICAL'S FACTS, at its tensor-parallel width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelFacts(FactsParts);

impl ModelFacts {
    /// The facts, refused when its groups break a law between them: a TurboQuant KV codec codes an
    /// attention's keys and values, so it needs an attention.
    pub const fn new(parts: FactsParts) -> Self {
        if let KvCodec::TurboQuant(_) = parts.kv_codec {
            assert!(
                parts.attention.is_some(),
                "ModelFacts: a TurboQuant KV codec is an attention's"
            );
        }
        Self(parts)
    }
    /// The parts, to restate the facts with one group changed (`ModelFacts::new(FactsParts {
    /// .., ..facts.parts() })`), which re-checks every law.
    pub const fn parts(self) -> FactsParts {
        self.0
    }
    /// The metal element dtype.
    pub const fn metal_dtype(self) -> MetalDtype {
        self.0.metal_dtype
    }
    /// The norm gains' dtype.
    pub const fn scale_dtype(self) -> ScaleDtype {
        self.0.scale_dtype
    }
    /// The residual stream's width (a vision tower's: its embedding width).
    pub const fn hidden(self) -> HiddenSize {
        self.0.hidden
    }
    /// The feed-forward's inner width, for a body with a feed-forward.
    pub const fn intermediate(self) -> Option<IntermediateSize> {
        self.0.intermediate
    }
    /// The vocabulary, for a body with an lm_head.
    pub const fn vocab(self) -> Option<VocabSize> {
        self.0.vocab
    }
    /// The RMSNorm epsilon.
    pub const fn eps(self) -> RmsNormEps {
        self.0.eps
    }
    /// The norms' run-time gain offset.
    pub const fn norm_offset(self) -> NormWeightOffset {
        self.0.norm_offset
    }
    /// The final logits' tanh cap, for a model that caps them.
    pub const fn final_softcap(self) -> Option<SoftCap> {
        self.0.final_softcap
    }
    /// The KV cache's codec.
    pub const fn kv_codec(self) -> KvCodec {
        self.0.kv_codec
    }
    /// The decoder's attention.
    pub const fn attention(self) -> Option<Attention> {
        self.0.attention
    }
    /// The vision tower.
    pub const fn vision(self) -> Option<Vision> {
        self.0.vision
    }
    /// The Gated-DeltaNet layers.
    pub const fn gdn(self) -> Option<Gdn> {
        self.0.gdn
    }
    /// The DFlash2 drafter's geometry.
    pub const fn drafter(self) -> Option<Drafter> {
        self.0.drafter
    }
    /// The drafts a verify step carries.
    pub const fn spec_drafts(self) -> Option<NumDrafts> {
        self.0.spec_drafts
    }
    /// The blocks a target captures for its drafter.
    pub const fn capture_slots(self) -> Option<CaptureSlots> {
        self.0.capture_slots
    }
    /// What a head reads of its target.
    pub const fn target(self) -> Option<Target> {
        self.0.target
    }
    /// The layers holding KV, when not every layer does.
    pub const fn kv_layers(self) -> Option<KvLayers> {
        self.0.kv_layers
    }
}

/// ⭐ THE RAW SPELLINGS: the facts as `CanonicalParams`' numeric consts spell them, for the readers that
/// still take numbers (the runtime, cuda's interpreter): `0` (or `-1`, `0.0`, `false`) where the model
/// lacks the group. The one place these spellings are derived; `CanonicalParams`' defaults call them.
///
/// A vision-only body's text consts are its tower's (the fill's long-standing rule): its head dim,
/// heads and query width are the tower's, its residual stream the tower's embedding.
pub mod raw {
    use super::*;

    /// `HEAD_DIM`: the base class's, else the tower's.
    pub const fn head_dim(f: ModelFacts) -> u32 {
        match (f.0.attention, f.0.vision) {
            (Some(a), _) => a.base.head_dim.get(),
            (None, Some(v)) => v.head_dim.get(),
            (None, None) => 0,
        }
    }
    /// `NUM_Q_HEADS`: the decoder's, else the tower's.
    pub const fn num_q_heads(f: ModelFacts) -> u32 {
        match (f.0.attention, f.0.vision) {
            (Some(a), _) => a.base.q_heads.get(),
            (None, Some(v)) => v.heads.get(),
            (None, None) => 0,
        }
    }
    /// `NUM_KV_HEADS`: the base class's.
    pub const fn num_kv_heads(f: ModelFacts) -> u32 {
        match f.0.attention {
            Some(a) => a.base.kv_heads.get(),
            None => 0,
        }
    }
    /// `Q_SIZE`: the decoder's query width, else the tower's.
    pub const fn q_size(f: ModelFacts) -> usize {
        match (f.0.attention, f.0.vision) {
            (Some(a), _) => a.base.q_width(),
            (None, Some(v)) => v.q_width(),
            (None, None) => 0,
        }
    }
    /// `KV_SIZE`: one K or V projection's width, the base class's.
    pub const fn kv_size(f: ModelFacts) -> usize {
        match f.0.attention {
            Some(a) => a.base.kv_heads.get() as usize * a.base.head_dim.get() as usize,
            None => 0,
        }
    }
    /// `INTERMEDIATE_SIZE`.
    pub const fn intermediate_size(f: ModelFacts) -> usize {
        match f.0.intermediate {
            Some(i) => i.get(),
            None => 0,
        }
    }
    /// `ATTN_SCALE`.
    pub const fn attn_scale(f: ModelFacts) -> f32 {
        match f.0.attention {
            Some(a) => a.scale.get(),
            None => 0.0,
        }
    }
    /// `ATTN_SOFTCAP`.
    pub const fn attn_softcap(f: ModelFacts) -> f32 {
        match f.0.attention {
            Some(Attention {
                softcap: Some(c), ..
            }) => c.get(),
            _ => 0.0,
        }
    }
    /// `SLIDING_WINDOW`: `-1` without sliding layers.
    pub const fn sliding_window(f: ModelFacts) -> i32 {
        match f.0.attention {
            Some(Attention {
                window: Some(w), ..
            }) => w.get() as i32,
            _ => -1,
        }
    }
    /// `FINAL_LOGIT_SOFTCAPPING`.
    pub const fn final_logit_softcapping(f: ModelFacts) -> f32 {
        match f.0.final_softcap {
            Some(c) => c.get(),
            None => 0.0,
        }
    }
    /// `MROPE_SECTION`.
    pub const fn mrope_section(f: ModelFacts) -> Option<[u32; 3]> {
        match f.0.attention {
            Some(Attention {
                rope: Rope { mrope: Some(s), .. },
                ..
            }) => Some(s.get()),
            _ => None,
        }
    }
    /// `KV_CACHE_LAYERS`.
    pub const fn kv_cache_layers(f: ModelFacts) -> Option<&'static [u32]> {
        match f.0.kv_layers {
            Some(l) => Some(l.get()),
            None => None,
        }
    }
    /// `VISION_NUM_HEADS`.
    pub const fn vision_num_heads(f: ModelFacts) -> u32 {
        match f.0.vision {
            Some(v) => v.heads.get(),
            None => 0,
        }
    }
    /// `VISION_HEAD_DIM`.
    pub const fn vision_head_dim(f: ModelFacts) -> u32 {
        match f.0.vision {
            Some(v) => v.head_dim.get(),
            None => 0,
        }
    }
    /// `VISION_Q_SIZE`.
    pub const fn vision_q_size(f: ModelFacts) -> usize {
        match f.0.vision {
            Some(v) => v.q_width(),
            None => 0,
        }
    }
    /// `VISION_IN_FEATURES`.
    pub const fn vision_in_features(f: ModelFacts) -> usize {
        match f.0.vision {
            Some(v) => v.in_features.get(),
            None => 0,
        }
    }
    /// `VISION_ROPE_INTERLEAVED`.
    pub const fn vision_rope_interleaved(f: ModelFacts) -> bool {
        matches!(
            f.0.vision,
            Some(Vision {
                pairing: VisionRopePairing::Interleaved,
                ..
            })
        )
    }
    /// `VISION_ATTN_SCALE`.
    pub const fn vision_attn_scale(f: ModelFacts) -> f32 {
        match f.0.vision {
            Some(v) => v.scale.get(),
            None => 0.0,
        }
    }
    /// `VISION_PATCH_GRID_SIDE`.
    pub const fn vision_patch_grid_side(f: ModelFacts) -> u32 {
        match f.0.vision {
            Some(Vision {
                patch_grid_side: Some(s),
                ..
            }) => s.get(),
            _ => 0,
        }
    }
    /// `VISION_POOL_KERNEL`.
    pub const fn vision_pool_kernel(f: ModelFacts) -> u32 {
        match f.0.vision {
            Some(Vision {
                pool_kernel: Some(k),
                ..
            }) => k.get(),
            _ => 0,
        }
    }
    /// `BLOCK_SIZE`: the base class's page, else the default.
    pub const fn block_size(f: ModelFacts) -> u32 {
        match f.0.attention {
            Some(a) => a.base.page.get(),
            None => DEFAULT_KV_PAGE.get(),
        }
    }
    /// `GLOBAL_BLOCK_SIZE`: the global class's page, else the default.
    pub const fn global_block_size(f: ModelFacts) -> u32 {
        match f.0.attention {
            Some(a) => a.global.page.get(),
            None => DEFAULT_KV_PAGE.get(),
        }
    }
    /// `GDN_NUM_K_HEADS`.
    pub const fn gdn_num_k_heads(f: ModelFacts) -> u32 {
        match f.0.gdn {
            Some(g) => g.k_heads.get(),
            None => 0,
        }
    }
    /// `GDN_NUM_V_HEADS`.
    pub const fn gdn_num_v_heads(f: ModelFacts) -> u32 {
        match f.0.gdn {
            Some(g) => g.v_heads.get(),
            None => 0,
        }
    }
    /// `GDN_HEAD_K_DIM`.
    pub const fn gdn_head_k_dim(f: ModelFacts) -> u32 {
        match f.0.gdn {
            Some(g) => g.k_dim.get(),
            None => 0,
        }
    }
    /// `GDN_HEAD_V_DIM`.
    pub const fn gdn_head_v_dim(f: ModelFacts) -> u32 {
        match f.0.gdn {
            Some(g) => g.v_dim.get(),
            None => 0,
        }
    }
    /// `GDN_CONV_KERNEL`.
    pub const fn gdn_conv_kernel(f: ModelFacts) -> u32 {
        match f.0.gdn {
            Some(g) => g.conv_kernel.get(),
            None => 0,
        }
    }
    /// `GDN_CONV_DIM`.
    pub const fn gdn_conv_dim(f: ModelFacts) -> usize {
        match f.0.gdn {
            Some(g) => g.conv_dim.get(),
            None => 0,
        }
    }
    /// `SPEC_DRAFTS`.
    pub const fn spec_drafts(f: ModelFacts) -> u32 {
        match f.0.spec_drafts {
            Some(d) => d.get(),
            None => 0,
        }
    }
    /// `DRAFT_CONV_GROUP`.
    pub const fn draft_conv_group(f: ModelFacts) -> u32 {
        match f.0.drafter {
            Some(d) => d.conv_group.get(),
            None => 0,
        }
    }
    /// `DRAFT_SELECTOR_RANK`.
    pub const fn draft_selector_rank(f: ModelFacts) -> u32 {
        match f.0.drafter {
            Some(d) => d.selector_rank.get(),
            None => 0,
        }
    }
    /// `DRAFT_MASK_TOKEN`.
    pub const fn draft_mask_token(f: ModelFacts) -> u32 {
        match f.0.drafter {
            Some(d) => d.mask_token.get(),
            None => 0,
        }
    }
    /// `TARGET_HIDDEN_WIDTH`: a head's target's, else the model's own hidden width.
    pub const fn target_hidden_width(f: ModelFacts) -> usize {
        match f.0.target {
            Some(t) => t.hidden.get(),
            None => f.0.hidden.get(),
        }
    }
    /// `TARGET_ROUTES_EXPERTS`.
    pub const fn target_routes_experts(f: ModelFacts) -> bool {
        matches!(
            f.0.target,
            Some(Target {
                routing: TargetRouting::Experts,
                ..
            })
        )
    }
    /// `DRAFT_CAPTURE_SLOTS`.
    pub const fn draft_capture_slots(f: ModelFacts) -> u32 {
        match f.0.capture_slots {
            Some(s) => s.get(),
            None => 0,
        }
    }
    /// `ROT_DIM`: the base class's, else `HEAD_DIM`.
    pub const fn rot_dim(f: ModelFacts) -> u32 {
        match f.0.attention {
            Some(a) => a.base.rot_dim.get(),
            None => head_dim(f),
        }
    }
    /// `GLOBAL_HEAD_DIM`: the global class's, else `HEAD_DIM`.
    pub const fn global_head_dim(f: ModelFacts) -> u32 {
        match f.0.attention {
            Some(a) => a.global.head_dim.get(),
            None => head_dim(f),
        }
    }
    /// `NUM_GLOBAL_KV_HEADS`: the global class's.
    pub const fn num_global_kv_heads(f: ModelFacts) -> u32 {
        match f.0.attention {
            Some(a) => a.global.kv_heads.get(),
            None => 0,
        }
    }
    /// `GLOBAL_ROT_DIM`: the global class's, else `GLOBAL_HEAD_DIM`.
    pub const fn global_rot_dim(f: ModelFacts) -> u32 {
        match f.0.attention {
            Some(a) => a.global.rot_dim.get(),
            None => global_head_dim(f),
        }
    }
    /// `GLOBAL_Q_SIZE`: the global class's query width, else `Q_SIZE`.
    pub const fn global_q_size(f: ModelFacts) -> usize {
        match f.0.attention {
            Some(a) => a.global.q_width(),
            None => q_size(f),
        }
    }
    /// `ROPE_PROPORTIONAL`.
    pub const fn rope_proportional(f: ModelFacts) -> bool {
        matches!(
            f.0.attention,
            Some(Attention {
                rope: Rope {
                    pairing: RopePairing::Proportional,
                    ..
                },
                ..
            })
        )
    }
    /// `ROPE_ON_READ`.
    pub const fn rope_on_read(f: ModelFacts) -> bool {
        matches!(
            f.0.attention,
            Some(Attention {
                rope: Rope {
                    kv_rope: KvRope::OnRead,
                    ..
                },
                ..
            })
        )
    }
    /// `HIDDEN_SIZE`.
    pub const fn hidden_size(f: ModelFacts) -> usize {
        f.0.hidden.get()
    }
    /// `VOCAB_SIZE`.
    pub const fn vocab_size(f: ModelFacts) -> usize {
        match f.0.vocab {
            Some(v) => v.get(),
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(q: u32, kv: u32, hd: u32, rot: u32) -> HeadGeometry {
        HeadGeometry::new(
            QueryHeads::new(q),
            KvHeads::new(kv),
            HeadDim::new(hd),
            RotDim::new(rot),
            BlockSize::new(16),
        )
    }

    /// qwen2.5-7b's 28 query heads over 4 kv heads are groups of 7.
    #[test]
    fn a_dividing_kv_head_count_has_its_group() {
        let g = geometry(28, 4, 128, 128);
        assert_eq!((g.gqa(), g.q_width()), (7, 28 * 128));
    }

    /// 29 query heads over 4 kv heads have no GQA grouping.
    #[test]
    #[should_panic(expected = "the kv heads divide the query heads")]
    fn a_kv_head_count_that_does_not_divide_is_refused() {
        geometry(29, 4, 128, 128);
    }

    /// A rot window wider than its head rotates lanes the head lacks.
    #[test]
    #[should_panic(expected = "the rot window fits its head")]
    fn a_rot_window_wider_than_its_head_is_refused() {
        geometry(32, 8, 64, 128);
    }

    #[test]
    #[should_panic(expected = "QueryHeads is a positive quantity")]
    fn zero_query_heads_are_refused() {
        QueryHeads::new(0);
    }

    #[test]
    #[should_panic(expected = "SoftCap is positive")]
    fn a_zero_soft_cap_is_no_cap() {
        SoftCap::new(0.0);
    }

    #[test]
    #[should_panic(expected = "AttnScale is finite")]
    fn a_nan_scale_is_refused() {
        AttnScale::new(f32::NAN);
    }

    /// Qwen3.5's section [11, 11, 10] is 32 pairs: a 64-lane rot window.
    #[test]
    fn an_mrope_section_fills_its_rot_window() {
        let a = Attention::new(
            geometry(16, 2, 256, 64),
            geometry(16, 2, 256, 64),
            AttnScale::new(0.0625),
            None,
            None,
            Rope::new(
                RopePairing::Standard,
                KvRope::AtWrite,
                Some(MropeSection::new([11, 11, 10])),
            ),
        );
        assert_eq!(a.mrope().map(MropeSection::pairs), Some(32));
    }

    #[test]
    #[should_panic(expected = "an MRoPE section's pairs are half each class's rot window")]
    fn an_mrope_section_that_misses_its_rot_window_is_refused() {
        Attention::new(
            geometry(16, 2, 256, 64),
            geometry(16, 2, 256, 64),
            AttnScale::new(0.0625),
            None,
            None,
            Rope::new(
                RopePairing::Standard,
                KvRope::AtWrite,
                Some(MropeSection::new([16, 24, 24])),
            ),
        );
    }

    /// Qwen3.5's GDN: 16 key heads of 128 and 32 value heads of 128 convolve 2·2048 + 4096.
    #[test]
    fn a_gdn_convolution_covers_its_streams() {
        let g = |conv_dim| {
            Gdn::new(
                GdnHeads::new(16),
                GdnHeads::new(32),
                GdnHeadDim::new(128),
                GdnHeadDim::new(128),
                ConvKernel::new(4),
                ConvDim::new(conv_dim),
            )
        };
        assert_eq!(g(8192).conv_dim().get(), 8192);
    }

    #[test]
    #[should_panic(expected = "the convolution covers the q, k and v streams side by side")]
    fn a_gdn_convolution_of_another_width_is_refused() {
        Gdn::new(
            GdnHeads::new(16),
            GdnHeads::new(32),
            GdnHeadDim::new(128),
            GdnHeadDim::new(128),
            ConvKernel::new(4),
            ConvDim::new(8000),
        );
    }

    /// A hybrid's attention layers, ascending, are a KV table.
    #[test]
    fn an_ascending_layer_table_is_a_kv_table() {
        const LAYERS: KvLayers = KvLayers::new(&[3, 7, 11]);
        assert_eq!(LAYERS.get(), [3, 7, 11]);
    }

    /// A descending table is refused by the one constructor both the bake's leaked table and the
    /// emitted const's literal go through.
    #[test]
    #[should_panic(expected = "KvLayers: the layers are strictly ascending")]
    fn a_descending_layer_table_is_refused() {
        let leaked: &'static [u32] = Vec::leak(vec![7, 3]);
        KvLayers::new(leaked);
    }

    /// Gemma 4's classes differ in kv heads and head dim, never in query heads.
    #[test]
    #[should_panic(expected = "both classes project the same query heads")]
    fn classes_that_disagree_on_query_heads_are_refused() {
        Attention::new(
            geometry(32, 8, 128, 128),
            geometry(16, 1, 512, 128),
            AttnScale::new(1.0),
            None,
            None,
            Rope::new(RopePairing::Standard, KvRope::AtWrite, None),
        );
    }

    fn parts(attention: Option<Attention>, kv_codec: KvCodec) -> FactsParts {
        FactsParts {
            metal_dtype: MetalDtype::Bf16,
            scale_dtype: ScaleDtype::F16,
            hidden: HiddenSize::new(2048),
            intermediate: Some(IntermediateSize::new(8192)),
            vocab: None,
            eps: RmsNormEps::new(1e-5),
            norm_offset: NormWeightOffset::new(0.0),
            final_softcap: None,
            kv_codec,
            kv_layers: None,
            attention,
            vision: None,
            gdn: None,
            drafter: None,
            spec_drafts: None,
            capture_slots: None,
            target: None,
        }
    }

    /// A TurboQuant KV codec on a body without attention codes nothing.
    #[test]
    #[should_panic(expected = "a TurboQuant KV codec is an attention's")]
    fn a_turboquant_codec_without_attention_is_refused() {
        ModelFacts::new(parts(None, KvCodec::TurboQuant(crate::TqBits::new(4))));
    }
}
