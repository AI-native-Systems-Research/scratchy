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
use core::marker::PhantomData;
use scratchy_tensors::typed::typenum::U2;
use scratchy_tensors::typed::{Add, Dim, DimN, Divides, Equals, FitsIn, Mul};
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
    /// The leading lanes of a head that rotary rotates.
    RotDim(u32),
    /// Query heads per kv head — the GQA group size, minted once by
    /// [`HeadGeometry::new`](HeadGeometry::new) beside its divisibility law, so no reader
    /// re-derives the division.
    Gqa(u32),
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

/// One head's width. Deliberately outside `positive!`: its storage IS the non-zero its
/// law states, so [`HeadDim::nonzero`](HeadDim::nonzero) is a field read, not a
/// re-proven conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeadDim(std::num::NonZeroU32);
impl HeadDim {
    /// The width, refused at zero.
    pub const fn new(v: u32) -> Self {
        match std::num::NonZeroU32::new(v) {
            Some(nz) => Self(nz),
            None => panic!("HeadDim is a positive quantity"),
        }
    }
    /// The width as a number.
    pub const fn get(self) -> u32 {
        self.0.get()
    }
    /// The width as a non-zero — the storage already is one, so no caller re-proves
    /// the law this type states.
    pub const fn nonzero(self) -> std::num::NonZeroU32 {
        self.0
    }
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
    gqa: Gqa,
    q_width: u32,
    kv_width: u32,
}

impl HeadGeometry {
    /// The geometry of a model's numbers, as types: its laws are its bounds. The kv heads divide
    /// the query heads (`Kv: Divides<Q>`: an attention over a remainder would read another head's
    /// keys), and the rot window fits its head (`Rot: FitsIn<Hd>`). A model's emitted `FACTS`
    /// states each class through this, so rustc checks both laws for every model.
    ///
    /// ```
    /// # use scratchy_ir::model_facts::{BlockSize, HeadGeometry};
    /// # use scratchy_tensors::typed::typenum::{U4, U28, U128};
    /// const G: HeadGeometry = HeadGeometry::of::<U28, U4, U128, U128>(BlockSize::new(16));
    /// assert_eq!(G.gqa().get(), 7);
    /// ```
    ///
    /// 29 query heads over 4 kv heads have no GQA grouping (E0277):
    ///
    /// ```compile_fail,E0277
    /// # use scratchy_ir::model_facts::{BlockSize, HeadGeometry};
    /// # use scratchy_tensors::typed::typenum::{U4, U29, U128};
    /// const G: HeadGeometry = HeadGeometry::of::<U29, U4, U128, U128>(BlockSize::new(16));
    /// ```
    ///
    /// and a rot window wider than its head rotates lanes the head lacks (E0277):
    ///
    /// ```compile_fail,E0277
    /// # use scratchy_ir::model_facts::{BlockSize, HeadGeometry};
    /// # use scratchy_tensors::typed::typenum::{U8, U32, U64, U128};
    /// const G: HeadGeometry = HeadGeometry::of::<U32, U8, U64, U128>(BlockSize::new(16));
    /// ```
    pub const fn of<Q: Dim, Kv: Divides<Q>, Hd: Dim, Rot: FitsIn<Hd>>(page: BlockSize) -> Self {
        Self::new(
            QueryHeads::new(<Q as DimN>::N),
            KvHeads::new(<Kv as DimN>::N),
            HeadDim::new(<Hd as DimN>::N),
            RotDim::new(<Rot as DimN>::N),
            page,
        )
    }

    /// The geometry of values the fill and the bake read, refused when the kv heads do not divide
    /// the query heads or the rot window is wider than the head: [`HeadGeometry::of`]'s laws, a
    /// panic here (a build error under const evaluation) where the numbers are values.
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
        let q_width = q_heads
            .get()
            .checked_mul(head_dim.get())
            .expect("HeadGeometry: the query width fits u32");
        let kv_width = kv_heads
            .get()
            .checked_mul(head_dim.get())
            .expect("HeadGeometry: the kv width fits u32");
        Self {
            q_heads,
            kv_heads,
            head_dim,
            rot_dim,
            page,
            gqa: Gqa::new(q_heads.get() / kv_heads.get()),
            q_width,
            kv_width,
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
    /// Query heads per kv head — the group minted once at construction, exact by the
    /// constructor's law, so no reader re-derives the division.
    pub const fn gqa(self) -> Gqa {
        self.gqa
    }
    /// `q_heads · head_dim`, proven to fit at the constructor. A `u32` — the width every
    /// tape and kernel constant reads — widened to `usize` only by the `raw::` consts,
    /// where the cast is lossless.
    pub const fn q_width(self) -> u32 {
        self.q_width
    }
    /// `kv_heads · head_dim`, proven to fit at the constructor — one K or V projection's
    /// column count, narrower than [`q_width`](Self::q_width) by exactly the GQA group
    /// size.
    pub const fn kv_width(self) -> u32 {
        self.kv_width
    }
}

/// One attention class's numbers, as types: `Kv` kv heads of `Hd` lanes, the leading `Rot` of
/// them rotated, cached in pages of `Page` positions. [`ClassOf`] states its laws.
pub struct Class<Kv, Hd, Rot, Page>(PhantomData<fn() -> (Kv, Hd, Rot, Page)>);

/// An attention class over `Q` query heads, its laws its bounds: the kv heads divide the query
/// heads and the rot window fits its head ([`HeadGeometry::of`]'s).
pub trait ClassOf<Q: Dim> {
    /// The rot window.
    type Rot: Dim;
    /// The elements one KV page holds: `Page · Kv · Hd`.
    type PageElems: Dim;
    /// The class as the value its readers take.
    const GEOMETRY: HeadGeometry;
}

impl<Q: Dim, Kv: Divides<Q>, Hd: Dim, Rot: FitsIn<Hd>, Page: Dim> ClassOf<Q>
    for Class<Kv, Hd, Rot, Page>
where
    Mul<Mul<Page, Kv>, Hd>: Dim,
{
    type Rot = Rot;
    type PageElems = Mul<Mul<Page, Kv>, Hd>;
    const GEOMETRY: HeadGeometry =
        HeadGeometry::of::<Q, Kv, Hd, Rot>(BlockSize::new(<Page as DimN>::N));
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

/// A 1-D rope: no MRoPE section, which every rot window has.
pub struct NoMrope;

/// An MRoPE section as types: the pairs `T`, `H` and `W` its temporal, height and width bands take.
pub struct Mrope<T, H, W>(PhantomData<fn() -> (T, H, W)>);

/// A rope's MRoPE section over a class whose rot window is `Rot`, its law its bound: a section's
/// pairs are half the window (`2 · (T + H + W) = Rot`), so its bands rotate every lane the window
/// does and no other.
pub trait MropeOf<Rot: Dim> {
    /// The section, as the value its readers take.
    const SECTION: Option<MropeSection>;
}

impl<Rot: Dim> MropeOf<Rot> for NoMrope {
    const SECTION: Option<MropeSection> = None;
}

impl<T: Dim, H: Dim, W: Dim, Rot: Dim> MropeOf<Rot> for Mrope<T, H, W>
where
    Mul<Add<Add<T, H>, W>, U2>: Equals<Rot>,
{
    const SECTION: Option<MropeSection> = Some(MropeSection::new([
        <T as DimN>::N,
        <H as DimN>::N,
        <W as DimN>::N,
    ]));
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
    /// The attention of a model's numbers, as types: its laws are its bounds. Both classes project
    /// the one `Q` query heads ([`ClassOf<Q>`]); their pages hold the same elements
    /// (`G::PageElems: Equals<B::PageElems>`: the two classes' blocks share one physical pool,
    /// vLLM's page-size unification); and an MRoPE section's pairs are half each class's rot
    /// window ([`MropeOf`]). A model's emitted `FACTS` states its attention through this, so rustc
    /// checks every law for every model.
    ///
    /// ```
    /// # use scratchy_ir::model_facts::*;
    /// # use scratchy_tensors::typed::typenum::{U2, U10, U11, U16, U64, U256};
    /// const A: Attention = Attention::of::<
    ///     U16,
    ///     Class<U2, U256, U64, U16>,
    ///     Class<U2, U256, U64, U16>,
    ///     Mrope<U11, U11, U10>,
    /// >(AttnScale::new(0.0625), None, None, RopePairing::Standard, KvRope::AtWrite);
    /// assert_eq!(A.mrope().map(MropeSection::pairs), Some(32));
    /// ```
    ///
    /// Gemma 4's global class, 2 kv heads of 512 against its sliding class's 8 of 256, holds the
    /// same elements a page at twice the sliding page:
    ///
    /// ```
    /// # use scratchy_ir::model_facts::*;
    /// # use scratchy_tensors::typed::typenum::{U2, U8, U16, U32, U256, U512};
    /// const A: Attention = Attention::of::<
    ///     U32,
    ///     Class<U8, U256, U256, U16>,
    ///     Class<U2, U512, U512, U32>,
    ///     NoMrope,
    /// >(AttnScale::new(1.0), None, None, RopePairing::Proportional, KvRope::AtWrite);
    /// assert_eq!(A.global().page().get(), 32);
    /// ```
    ///
    /// A section whose pairs are not half the rot window rotates lanes the window does not (E0277):
    ///
    /// ```compile_fail,E0277
    /// # use scratchy_ir::model_facts::*;
    /// # use scratchy_tensors::typed::typenum::{U2, U16, U24, U64, U256};
    /// const A: Attention = Attention::of::<
    ///     U16,
    ///     Class<U2, U256, U64, U16>,
    ///     Class<U2, U256, U64, U16>,
    ///     Mrope<U16, U24, U24>,
    /// >(AttnScale::new(0.0625), None, None, RopePairing::Standard, KvRope::AtWrite);
    /// ```
    ///
    /// and a global page that holds other than the sliding page's elements shares no pool with it
    /// (E0277):
    ///
    /// ```compile_fail,E0277
    /// # use scratchy_ir::model_facts::*;
    /// # use scratchy_tensors::typed::typenum::{U2, U8, U16, U32, U256, U512};
    /// const A: Attention = Attention::of::<
    ///     U32,
    ///     Class<U8, U256, U256, U16>,
    ///     Class<U2, U512, U512, U16>,
    ///     NoMrope,
    /// >(AttnScale::new(1.0), None, None, RopePairing::Proportional, KvRope::AtWrite);
    /// ```
    pub const fn of<
        Q: Dim,
        B: ClassOf<Q>,
        G: ClassOf<Q, PageElems: Equals<B::PageElems>>,
        M: MropeOf<B::Rot> + MropeOf<G::Rot>,
    >(
        scale: AttnScale,
        softcap: Option<SoftCap>,
        window: Option<AttnWindow>,
        pairing: RopePairing,
        kv_rope: KvRope,
    ) -> Self {
        Self::new(
            B::GEOMETRY,
            G::GEOMETRY,
            scale,
            softcap,
            window,
            Rope::new(pairing, kv_rope, <M as MropeOf<B::Rot>>::SECTION),
        )
    }

    /// The attention of values the fill and the tests read, refused when its classes disagree on
    /// the query heads (every layer projects the same query heads; only the kv side differs by
    /// class). Its other laws are [`Attention::of`]'s bounds, which every model's `FACTS` states.
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
    /// The layers' geometry of a model's numbers, as types: `KH` key heads of `KD` lanes, `VH`
    /// value heads of `VD`, a short convolution `CK` wide over `CD` channels. Its law is its bound:
    /// the convolution covers exactly its streams, the q and k streams (`2 · KH · KD`) beside the v
    /// stream (`VH · VD`). Mamba-2's conv covers its `x` (`heads · head_dim`) beside its B and C
    /// (`2 · groups · state`): the same law under the reading above. A model's emitted `FACTS`
    /// states its layers through this, so rustc checks the law for every model.
    ///
    /// ```
    /// # use scratchy_ir::model_facts::Gdn;
    /// # use scratchy_tensors::typed::typenum::{U4, U16, U32, U128, U8192};
    /// const G: Gdn = Gdn::of::<U16, U32, U128, U128, U4, U8192>();
    /// assert_eq!(G.conv_dim().get(), 8192);
    /// ```
    ///
    /// A convolution of another width than its streams' (E0277):
    ///
    /// ```compile_fail,E0277
    /// # use scratchy_ir::model_facts::Gdn;
    /// # use scratchy_tensors::typed::typenum::{U4, U16, U32, U128, U4096};
    /// const G: Gdn = Gdn::of::<U16, U32, U128, U128, U4, U4096>();
    /// ```
    pub const fn of<KH: Dim, VH: Dim, KD: Dim, VD: Dim, CK: Dim, CD: Dim>() -> Self
    where
        Add<Mul<U2, Mul<KH, KD>>, Mul<VH, VD>>: Equals<CD>,
    {
        Self::new(
            GdnHeads::new(<KH as DimN>::N),
            GdnHeads::new(<VH as DimN>::N),
            GdnHeadDim::new(<KD as DimN>::N),
            GdnHeadDim::new(<VD as DimN>::N),
            ConvKernel::new(<CK as DimN>::N),
            ConvDim::new(<CD as DimN>::N as usize),
        )
    }

    /// The layers' geometry of values the fill and the tests read. Its law is [`Gdn::of`]'s bound,
    /// which every model's `FACTS` states.
    pub const fn new(
        k_heads: GdnHeads,
        v_heads: GdnHeads,
        k_dim: GdnHeadDim,
        v_dim: GdnHeadDim,
        conv_kernel: ConvKernel,
        conv_dim: ConvDim,
    ) -> Self {
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

/// How a head's width stands to its target's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetWidth {
    /// The head's own: it reads its target's rows through a projection of its own, so the two
    /// widths are unrelated (a DFlash2 drafter, an assistant).
    Own,
    /// Its target's: the head states its own rows at its target's width (the DSL's `as_dim`, an
    /// in-checkpoint head reading its target's lent lm_head), so [`ModelFacts::new`] holds the
    /// two equal.
    Lent,
}

/// What a head reads of the target it drafts for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    hidden: HiddenSize,
    routing: TargetRouting,
    width: TargetWidth,
}

impl Target {
    /// The target, as a head at its own width reads it.
    pub const fn new(hidden: HiddenSize, routing: TargetRouting) -> Self {
        Self {
            hidden,
            routing,
            width: TargetWidth::Own,
        }
    }
    /// The target, as a head lent its width reads it ([`TargetWidth::Lent`]).
    pub const fn lent(hidden: HiddenSize, routing: TargetRouting) -> Self {
        Self {
            hidden,
            routing,
            width: TargetWidth::Lent,
        }
    }
    /// The width of one of its hidden rows.
    pub const fn hidden(self) -> HiddenSize {
        self.hidden
    }
    /// How it routes the rows it verifies.
    pub const fn routing(self) -> TargetRouting {
        self.routing
    }
    /// How the head's width stands to this one.
    pub const fn width(self) -> TargetWidth {
        self.width
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
    /// attention's keys and values, so it needs an attention; a head lent its target's width
    /// ([`TargetWidth::Lent`]) is that width.
    pub const fn new(parts: FactsParts) -> Self {
        if let KvCodec::TurboQuant(_) = parts.kv_codec {
            assert!(
                parts.attention.is_some(),
                "ModelFacts: a TurboQuant KV codec is an attention's"
            );
        }
        if let Some(Target {
            hidden,
            width: TargetWidth::Lent,
            ..
        }) = parts.target
        {
            assert!(
                parts.hidden.get() == hidden.get(),
                "ModelFacts: a head lent its target's width is that width"
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
/// still take numbers (the runtime): `0` (or `-1`, `0.0`, `false`) where the model
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
            (Some(a), _) => a.base.q_width() as usize,
            (None, Some(v)) => v.q_width(),
            (None, None) => 0,
        }
    }
    /// `KV_SIZE`: one K or V projection's width, the base class's.
    pub const fn kv_size(f: ModelFacts) -> usize {
        match f.0.attention {
            Some(a) => a.base.kv_width() as usize,
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
            Some(a) => a.global.q_width() as usize,
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
        assert_eq!((g.gqa().get(), g.q_width()), (7, 28 * 128));
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

    fn head_of(target: Target) -> ModelFacts {
        ModelFacts::new(FactsParts {
            target: Some(target),
            ..parts(None, KvCodec::Dense)
        })
    }

    /// An in-checkpoint head states its rows at its target's width (`as_dim`): at the head's own
    /// width, it holds.
    #[test]
    fn a_head_lent_its_targets_width_at_that_width_holds() {
        let facts = head_of(Target::lent(HiddenSize::new(2048), TargetRouting::Experts));
        assert_eq!(facts.target().map(Target::width), Some(TargetWidth::Lent));
    }

    /// A head lent its target's width that is not its own is refused.
    #[test]
    #[should_panic(expected = "a head lent its target's width is that width")]
    fn a_head_lent_another_width_is_refused() {
        head_of(Target::lent(HiddenSize::new(4096), TargetRouting::Experts));
    }

    /// A drafter reads its target's rows through its own projection: the widths may differ.
    #[test]
    fn a_head_at_its_own_width_may_differ_from_its_target() {
        let facts = head_of(Target::new(HiddenSize::new(16384), TargetRouting::Experts));
        assert_eq!(facts.target().map(Target::width), Some(TargetWidth::Own));
    }
}
