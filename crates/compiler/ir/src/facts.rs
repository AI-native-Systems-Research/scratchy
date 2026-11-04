// SPDX-License-Identifier: Apache-2.0
//! ⭐ A CANONICAL'S FACTS AS A TYPE.
//!
//! scratchy is a per-model compiler: every model quantity is a constant of the compilation that
//! bakes the model. [`ModelFacts`] states them as associated consts of one type per canonical, and
//! every lowering is generic over that type (`fn lower<M: ModelFacts>`), so the facts reach each
//! lowering as compile-time constants, never as a runtime value struct.
//!
//! Each quantity is its own newtype with a private field: a query-head count cannot be passed where a
//! kv-head count belongs, and an epsilon cannot be passed as a scale. A fact a model may lack is an
//! `Option` of its group (a model without a vision tower has no [`Vision`]), never a zero or `-1`
//! standing in for absence.
//!
//! Every law is checked in a `const fn` constructor. The per-canonical types are generated (see the
//! `scratchy-model-facts` crate), and each impl's consts are evaluated when that crate compiles, so a
//! config that breaks a law (a kv-head count that does not divide the query-head count, a rot window
//! wider than its head) is a build error naming the law, not a runtime refusal.

// Every type a generated impl names, reachable through this one module.
pub use crate::{KvCodec, TqBits};
pub use scratchy_tensors::{MetalDtype, ScaleDtype};

/// A token id. Zero is a token like any other, so this is not a positive quantity.
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

/// A positive count or extent. The constructor refuses zero, so holding one is holding the fact
/// that it is at least one; `get()` is the only exit.
macro_rules! positive {
    ($($(#[$m:meta])* $name:ident($t:ty)),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($t);
        impl $name {
            /// The quantity, refused at zero.
            pub const fn new(v: $t) -> Self {
                assert!(v > 0, concat!(stringify!($name), " is a positive quantity"));
                Self(v)
            }
            /// The quantity as a number, for the arithmetic and the kernel ABI that need one.
            pub const fn get(self) -> $t {
                self.0
            }
        }
    )*};
}

/// A finite `f32` scalar. `f32` is never a const parameter, so a scalar fact is an associated
/// const of this type; the constructor refuses NaN and infinity.
macro_rules! scalar {
    ($($(#[$m:meta])* $name:ident),* $(,)?) => {$(
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
        pub struct $name(f32);
        impl $name {
            /// The scalar, refused when not finite.
            pub const fn new(v: f32) -> Self {
                assert!(v.is_finite(), concat!(stringify!($name), " is finite"));
                Self(v)
            }
            /// The scalar as a number.
            pub const fn get(self) -> f32 {
                self.0
            }
        }
    )*};
}

positive! {
    /// How many query heads attend (`num_attention_heads`, sharded by the tensor-parallel width).
    QueryHeads(u32),
    /// How many key/value heads the cache holds (`num_key_value_heads`, sharded). Equal to
    /// [`QueryHeads`] only on a model without GQA, which is why it is a different type.
    KvHeads(u32),
    /// How wide one head is (`head_dim`). Not a hidden size and not a lane count.
    HeadDim(u32),
    /// The leading lanes of a head that rotary rotates (`partial_rotary_factor · head_dim`).
    RotDim(u32),
    /// Query heads per kv head. Never constructed by a reader: [`HeadGeometry::new`] divides it
    /// out once, after proving the division exact.
    Gqa(u32),
    /// Positions per paged-KV block.
    BlockSize(u32),
    /// The residual stream's width (`hidden_size`).
    HiddenSize(usize),
    /// The feed-forward's inner width.
    IntermediateSize(usize),
    /// The vocabulary: the lm_head's rows.
    VocabSize(usize),
    /// The positions a sliding-window layer attends to.
    AttnWindow(u32),
    /// The drafts each sequence of a speculative verify step carries.
    NumDrafts(u32),
    /// Blocks of a target's captured hidden rows that a drafter reads.
    CaptureSlots(u32),
    /// The channels each dynamic-convolution weight of a DFlash2 drafter covers.
    ConvGroup(u32),
    /// The rank of a DFlash2 drafter's selector codebooks.
    SelectorRank(u32),
    /// A Gated-DeltaNet head count (key or value side).
    GdnHeads(u32),
    /// A Gated-DeltaNet head's width (key or value side).
    GdnHeadDim(u32),
    /// The Gated-DeltaNet short convolution's kernel width.
    ConvKernel(u32),
    /// The Gated-DeltaNet convolution's channel count.
    ConvDim(usize),
    /// A vision tower's patch grid side (square).
    PatchGridSide(u32),
    /// A vision tower's post-encoder average pool kernel (= stride).
    PoolKernel(u32),
    /// The input width of a vision tower's patch embedding.
    InFeatures(usize),
    /// An MLA width (`kv_lora_rank`, the q/k nope and rope dims, the value head dim).
    MlaDim(usize),
}

scalar! {
    /// The attention softmax's scale.
    AttnScale,
    /// A tanh soft cap (`attn_logit_softcapping` / `final_logit_softcapping`).
    SoftCap,
    /// The RMSNorm epsilon.
    RmsNormEps,
    /// The offset an RMSNorm adds to its stored gain (`1.0` for a zero-centered gain).
    NormWeightOffset,
}

/// The MRoPE section split `[T, H, W]`: the rotary pairs the time, height and width axes own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MropeSection([u32; 3]);

impl MropeSection {
    /// The split as the config states it.
    pub const fn new(section: [u32; 3]) -> Self {
        Self(section)
    }
    /// The split.
    pub const fn get(self) -> [u32; 3] {
        self.0
    }
}

/// The layers that hold KV, ascending, when not every layer does (a recurrent hybrid's attention
/// layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvLayers(&'static [u32]);

impl KvLayers {
    /// The layers, refused when not strictly ascending.
    pub const fn new(layers: &'static [u32]) -> Self {
        let mut i = 1;
        while i < layers.len() {
            assert!(layers[i - 1] < layers[i], "KvLayers are strictly ascending");
            i += 1;
        }
        Self(layers)
    }
    /// The layers.
    pub const fn get(self) -> &'static [u32] {
        self.0
    }
}

/// Where a rotary pair's partner lane sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopePairing {
    /// Lane `i` pairs with `i + rot_dim / 2` inside the rot window (NeoX and HF partial rotary).
    Standard,
    /// Lane `i < rot_dim / 2` pairs with `i + head_dim / 2` (Gemma 4's proportional rope on its
    /// global class).
    Proportional,
}

/// How a cached key carries its rotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvRope {
    /// Rotated when written; attention reads it as stored.
    AtWrite,
    /// A relocatable (span) block stores K unrotated and attention re-ropes it to the reader's
    /// position, so one cached span serves any position.
    OnRead,
}

/// How a vision tower's 2D rope pairs its lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VisionRopePairing {
    /// GPT-NeoX rotate-half (every Qwen-VL tower).
    RotateHalf,
    /// Adjacent pairs, GPT-J style (MoonViT / LocateAnything).
    Interleaved,
}

/// Whether a head's target routes each row it verifies through experts of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetRouting {
    /// The target's feed-forward is dense.
    Dense,
    /// The target routes rows through experts.
    Experts,
}

/// One attention class's head geometry, with its GQA group divided out once, here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadGeometry {
    q_heads: QueryHeads,
    kv_heads: KvHeads,
    head_dim: HeadDim,
    rot_dim: RotDim,
    block_size: BlockSize,
    gqa: Gqa,
}

impl HeadGeometry {
    /// THE MINT, and the only evaluation of the GQA division. A kv-head count that does not divide
    /// the query-head count has no GQA grouping (its attention would read another head's keys), and
    /// a rot window wider than its head rotates lanes the head does not have: both are refused.
    pub const fn new(
        q_heads: QueryHeads,
        kv_heads: KvHeads,
        head_dim: HeadDim,
        rot_dim: RotDim,
        block_size: BlockSize,
    ) -> Self {
        assert!(
            q_heads.get().is_multiple_of(kv_heads.get()),
            "HeadGeometry: the kv-head count divides the query-head count"
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
            block_size,
            gqa: Gqa::new(q_heads.get() / kv_heads.get()),
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
    /// The head dim.
    pub const fn head_dim(self) -> HeadDim {
        self.head_dim
    }
    /// The rot window.
    pub const fn rot_dim(self) -> RotDim {
        self.rot_dim
    }
    /// The page size.
    pub const fn block_size(self) -> BlockSize {
        self.block_size
    }
    /// Query heads per kv head, exact by construction.
    pub const fn gqa(self) -> Gqa {
        self.gqa
    }
    /// `q_heads · head_dim`: the query projection's and the attention output's width.
    pub const fn q_width(self) -> usize {
        self.q_heads.get() as usize * self.head_dim.get() as usize
    }
    /// `kv_heads · head_dim`: one K or V projection's width.
    pub const fn kv_width(self) -> usize {
        self.kv_heads.get() as usize * self.head_dim.get() as usize
    }
}

/// Which of a model's two attention classes a layer belongs to. A hybrid arch (Gemma 4: sliding
/// layers beside every sixth global one) states two [`HeadGeometry`]s; a uniform one states the same
/// geometry twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttnClass {
    /// The windowed (sliding) layers': the model's base geometry.
    Sliding,
    /// The full causal layers': the model's global geometry.
    Full,
}

/// A decoder's attention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attention {
    sliding: HeadGeometry,
    full: HeadGeometry,
    scale: AttnScale,
    window: Option<AttnWindow>,
    full_pairing: RopePairing,
    kv_rope: KvRope,
}

impl Attention {
    /// The attention, refused when its two classes disagree about the query-head count (every
    /// layer projects the same queries).
    pub const fn new(
        sliding: HeadGeometry,
        full: HeadGeometry,
        scale: AttnScale,
        window: Option<AttnWindow>,
        full_pairing: RopePairing,
        kv_rope: KvRope,
    ) -> Self {
        assert!(
            sliding.q_heads().get() == full.q_heads().get(),
            "Attention: both classes attend with the same query heads"
        );
        Self {
            sliding,
            full,
            scale,
            window,
            full_pairing,
            kv_rope,
        }
    }
    /// A class's geometry.
    pub const fn class(self, class: AttnClass) -> HeadGeometry {
        match class {
            AttnClass::Sliding => self.sliding,
            AttnClass::Full => self.full,
        }
    }
    /// The softmax scale.
    pub const fn scale(self) -> AttnScale {
        self.scale
    }
    /// The sliding class's window; `None` on a model with no sliding layers.
    pub const fn window(self) -> Option<AttnWindow> {
        self.window
    }
    /// How the full class's rotary pairs its lanes (the sliding class's pairing is always
    /// [`RopePairing::Standard`]).
    pub const fn full_pairing(self) -> RopePairing {
        self.full_pairing
    }
    /// How a cached key carries its rotation.
    pub const fn kv_rope(self) -> KvRope {
        self.kv_rope
    }
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
    /// The encoder's facts.
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
    /// The encoder's heads (plain MHA: as many kv heads as query heads).
    pub const fn heads(self) -> QueryHeads {
        self.heads
    }
    /// The encoder's head dim.
    pub const fn head_dim(self) -> HeadDim {
        self.head_dim
    }
    /// `heads · head_dim`.
    pub const fn q_width(self) -> usize {
        self.heads.get() as usize * self.head_dim.get() as usize
    }
    /// The patch embedding's input width.
    pub const fn in_features(self) -> InFeatures {
        self.in_features
    }
    /// The 2D rope's pairing.
    pub const fn pairing(self) -> VisionRopePairing {
        self.pairing
    }
    /// The encoder's softmax scale.
    pub const fn scale(self) -> AttnScale {
        self.scale
    }
    /// The patch grid's side, for a tower that pools after its encoder.
    pub const fn patch_grid_side(self) -> Option<PatchGridSide> {
        self.patch_grid_side
    }
    /// The post-encoder pool's kernel.
    pub const fn pool_kernel(self) -> Option<PoolKernel> {
        self.pool_kernel
    }
}

/// A Gated-DeltaNet (linear attention) layer's geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Gdn {
    k_heads: GdnHeads,
    v_heads: GdnHeads,
    head_k_dim: GdnHeadDim,
    head_v_dim: GdnHeadDim,
    conv_kernel: ConvKernel,
    conv_dim: ConvDim,
}

impl Gdn {
    /// The geometry, refused when the convolution's channels are not the q, k and v streams side by
    /// side (`2 · k_heads · head_k_dim + v_heads · head_v_dim`).
    pub const fn new(
        k_heads: GdnHeads,
        v_heads: GdnHeads,
        head_k_dim: GdnHeadDim,
        head_v_dim: GdnHeadDim,
        conv_kernel: ConvKernel,
        conv_dim: ConvDim,
    ) -> Self {
        assert!(
            conv_dim.get()
                == 2 * k_heads.get() as usize * head_k_dim.get() as usize
                    + v_heads.get() as usize * head_v_dim.get() as usize,
            "Gdn: the convolution covers the q, k and v streams"
        );
        Self {
            k_heads,
            v_heads,
            head_k_dim,
            head_v_dim,
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
    pub const fn head_k_dim(self) -> GdnHeadDim {
        self.head_k_dim
    }
    /// A value head's width.
    pub const fn head_v_dim(self) -> GdnHeadDim {
        self.head_v_dim
    }
    /// The short convolution's kernel width.
    pub const fn conv_kernel(self) -> ConvKernel {
        self.conv_kernel
    }
    /// The convolution's channels.
    pub const fn conv_dim(self) -> ConvDim {
        self.conv_dim
    }
}

/// A DFlash2 drafter's own geometry (Splash's `DFlashDraftLayout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Drafter {
    conv_group: ConvGroup,
    selector_rank: SelectorRank,
    mask_token: TokenId,
}

impl Drafter {
    /// The drafter's facts.
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
    /// `dflash_config.conv_group_size`.
    pub const fn conv_group(self) -> ConvGroup {
        self.conv_group
    }
    /// `dflash_config.selector_rank`.
    pub const fn selector_rank(self) -> SelectorRank {
        self.selector_rank
    }
    /// `dflash_config.mask_token_id`: what its block's proposal rows embed.
    pub const fn mask_token(self) -> TokenId {
        self.mask_token
    }
}

/// ⭐ ONE CANONICAL'S FACTS. Implemented once per canonical by a generated type; every lowering is
/// generic over it and reads `M::FACT`.
pub trait ModelFacts {
    /// The decoder's attention; `None` for a body without one.
    const ATTENTION: Option<Attention>;
    /// The residual stream's width.
    const HIDDEN_SIZE: HiddenSize;
    /// The feed-forward's inner width; `None` for a body without one.
    const INTERMEDIATE_SIZE: Option<IntermediateSize>;
    /// The vocabulary; `None` for a body without an lm_head.
    const VOCAB_SIZE: Option<VocabSize>;
    /// The RMSNorm epsilon.
    const RMS_NORM_EPS: RmsNormEps;
    /// The offset every RMSNorm adds to its stored gain.
    const NORM_WEIGHT_OFFSET: NormWeightOffset;
    /// The tanh cap on the final logits.
    const FINAL_LOGIT_SOFTCAP: Option<SoftCap>;
    /// The MRoPE split, for a model whose rope reads per-axis positions.
    const MROPE_SECTION: Option<MropeSection>;
    /// How the KV cache is stored.
    const KV_CODEC: KvCodec;
    /// The layers holding KV, when not every layer does.
    const KV_CACHE_LAYERS: Option<KvLayers>;
    /// The element dtype the metal kernels run in.
    const METAL_DTYPE: MetalDtype;
    /// The dtype a quantized checkpoint stores its scales and biases in.
    const SCALE_DTYPE: ScaleDtype;
    /// The vision tower's encoder.
    const VISION: Option<Vision>;
    /// The Gated-DeltaNet layers' geometry.
    const GDN: Option<Gdn>;
    /// The drafts a verify step carries.
    const SPEC_DRAFTS: Option<NumDrafts>;
    /// A DFlash2 drafter's geometry.
    const DRAFTER: Option<Drafter>;
    /// The blocks of captured rows a drafter compiled beside this target reads.
    const CAPTURE_SLOTS: Option<CaptureSlots>;
    /// The width of a `target_hidden` row a head reads.
    const TARGET_HIDDEN_WIDTH: HiddenSize;
    /// Whether a head's target routes its verify rows through experts.
    const TARGET_ROUTING: TargetRouting;

    /// Every fact, evaluated. An associated const is evaluated only where it is read, so the
    /// generator names this beside each canonical (`const _: () = <T as ModelFacts>::LAWS;`): each
    /// constructor's law is then checked when stage 1 compiles, for every selected canonical,
    /// whatever stage 2 reads.
    const LAWS: () = {
        let _ = Self::ATTENTION;
        let _ = Self::HIDDEN_SIZE;
        let _ = Self::INTERMEDIATE_SIZE;
        let _ = Self::VOCAB_SIZE;
        let _ = Self::RMS_NORM_EPS;
        let _ = Self::NORM_WEIGHT_OFFSET;
        let _ = Self::FINAL_LOGIT_SOFTCAP;
        let _ = Self::MROPE_SECTION;
        let _ = Self::KV_CODEC;
        let _ = Self::KV_CACHE_LAYERS;
        let _ = Self::METAL_DTYPE;
        let _ = Self::SCALE_DTYPE;
        let _ = Self::VISION;
        let _ = Self::GDN;
        let _ = Self::SPEC_DRAFTS;
        let _ = Self::DRAFTER;
        let _ = Self::CAPTURE_SLOTS;
        let _ = Self::TARGET_HIDDEN_WIDTH;
        let _ = Self::TARGET_ROUTING;
    };
}

/// What runs once a canonical's facts are a type: the consumer of `scratchy-model-facts`'
/// generated `dispatch`, which names the canonical's type and calls [`visit`](Self::visit) at it.
/// A lowering behind a visitor is monomorphised per canonical, and only for the canonicals the
/// build selected.
pub trait FactsVisitor {
    /// What the visit produces.
    type Out;
    /// The visitor's body, at one canonical's facts.
    fn visit<M: ModelFacts>(self) -> Self::Out;
}
