// SPDX-License-Identifier: Apache-2.0
//! SHAPES AS RUST TYPES: the op signatures as generic Rust functions, and every carrier the
//! typed-Rust emit pass (`typed_emit.rs`, run by the build script) turned into a Rust function over them.
//!
//! A dim is a type: a config bound is a marker named for it (`HeadDim` for `head_dim`), a product
//! is [`Mul`] in the one canonical order [`canonical_mul`] writes. An activation is [`Act<T, D>`]:
//! `T` rows (the token axis) of `D` features. A weight is [`W1`]/[`W2`] at the shape its arch's
//! `weights.json` declares. So `gemm(x: Act<T, K>, w: W2<K, N>) -> Act<T, N>` is the law
//! `x`'s last dim equals `w`'s first, and rustc checks it where a carrier calls it: a shape error
//! in a carrier is a type error at the generated line, which names the carrier's file and line.
//!
//! The emitted functions also RUN: [`Trace`] records, per [`LocalId`], the shape each binding's
//! type states. That record is the carrier's shapes, symbolic in its config's bounds. Until phase
//! C of the shapes-as-Rust-types plan (issue #1086) makes it the source, the bake checks it against
//! `shape::infer`'s ([`check_parity`]).

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

use crate::classified::LocalId;
use crate::dim::{Dim, canonical_add, canonical_mul};

/// A dim as a type: the symbolic [`Dim`] it stands for.
pub trait DimType {
    /// The dim.
    fn dim() -> Dim;
}

/// A config bound as a type: the bound's name.
pub trait Bound {
    /// The config's name for it.
    const NAME: &'static str;
}

/// The marker types of config bounds, each `Name => "config_name"`.
macro_rules! bounds {
    ($($marker:ident => $name:literal),* $(,)?) => {$(
        #[doc = concat!("The `", $name, "` config bound, as a type.")]
        pub enum $marker {}
        impl Bound for $marker {
            const NAME: &'static str = $name;
        }
        impl DimType for $marker {
            fn dim() -> Dim {
                Dim::Bound($name.to_string())
            }
        }
    )*};
}

/// The product `A · B`. Products are written in [`canonical_mul`]'s order, so equal products are
/// equal types.
pub struct Mul<A, B>(PhantomData<(A, B)>);

impl<A: DimType, B: DimType> DimType for Mul<A, B> {
    fn dim() -> Dim {
        canonical_mul(vec![A::dim(), B::dim()])
    }
}

/// The quotient `A / B` (a vision merge's rows: the patches over the merge factor).
pub struct Div<A, B>(PhantomData<(A, B)>);

impl<A: DimType, B: DimType> DimType for Div<A, B> {
    fn dim() -> Dim {
        Dim::Div(Box::new(A::dim()), Box::new(B::dim()))
    }
}

/// The sum `A + B`: a packed projection's width, its parts side by side. Sums are written in
/// [`canonical_add`]'s order, so equal sums are equal types.
pub struct Add<A, B>(PhantomData<(A, B)>);

impl<A: DimType, B: DimType> DimType for Add<A, B> {
    fn dim() -> Dim {
        canonical_add(vec![A::dim(), B::dim()])
    }
}

/// A literal dim, `N`.
pub enum Lit<const N: u64> {}

impl<const N: u64> DimType for Lit<N> {
    fn dim() -> Dim {
        Dim::Lit(N)
    }
}

/// A scalar operand (a literal, `sqrt(<bound>)`, a config scalar): no shape.
#[derive(Clone, Copy)]
pub struct Scalar;

/// Phantom values are copies whatever their parameters: a shape handle holds no data.
macro_rules! phantom_copy {
    ($($t:ident<$($p:ident),*>),* $(,)?) => {$(
        impl<$($p),*> Clone for $t<$($p),*> {
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<$($p),*> Copy for $t<$($p),*> {}
    )*};
}

/// An activation: `T` rows of `D` features each.
pub struct Act<T, D>(PhantomData<(T, D)>);
/// One index per row: `[T]` (token ids, positions).
pub struct Tokens<T>(PhantomData<T>);
/// A rank-1 weight `[D]`.
pub struct W1<D>(PhantomData<D>);
/// A rank-2 weight `[K, N]`.
pub struct W2<K, N>(PhantomData<(K, N)>);

phantom_copy!(Act<T, D>, Tokens<T>, W1<D>, W2<K, N>);

/// A weight at the shape its type declares.
pub trait Declared {
    /// The weight.
    fn declared() -> Self;
}

impl<D> Declared for W1<D> {
    fn declared() -> Self {
        Self(PhantomData)
    }
}

impl<K, N> Declared for W2<K, N> {
    fn declared() -> Self {
        Self(PhantomData)
    }
}

/// An `if`'s merge: both arms produce the same type, which is the law that their bindings agree.
/// The trace runs both arms, so each arm's locals are recorded; which arm a layer takes changes no
/// shape.
pub fn merge<A>(then_arm: A, _else_arm: A) -> A {
    then_arm
}

/// The rotary tables: opaque to shapes.
#[derive(Clone, Copy)]
pub struct Rotary;
/// One layer's KV cache: opaque to shapes.
#[derive(Clone, Copy)]
pub struct KvCache;
/// The paged block table: opaque to shapes.
#[derive(Clone, Copy)]
pub struct BlockTable;
/// A vision forward's per-image sequence boundaries: opaque to shapes.
#[derive(Clone, Copy)]
pub struct CuSeqlens;
/// A vision forward's longest image sequence: opaque to shapes.
#[derive(Clone, Copy)]
pub struct MaxSeqlen;
/// A row permutation (a windowed vision tower's window order, or its inverse): opaque to shapes,
/// as its gather keeps the rows' shape.
#[derive(Clone, Copy)]
pub struct RowIndex;

/// A value whose type states a shape.
pub trait Traced {
    /// The shape its type states.
    fn shape() -> Vec<Dim>;
}

impl<T: DimType, D: DimType> Traced for Act<T, D> {
    fn shape() -> Vec<Dim> {
        vec![T::dim(), D::dim()]
    }
}

impl<T: DimType> Traced for Tokens<T> {
    fn shape() -> Vec<Dim> {
        vec![T::dim()]
    }
}

/// A tuple of traced values, bound to as many locals.
pub trait TracedTuple<const N: usize> {
    /// Each element's shape, in order.
    fn shapes() -> [Vec<Dim>; N];
}

impl<A: Traced, B: Traced> TracedTuple<2> for (A, B) {
    fn shapes() -> [Vec<Dim>; 2] {
        [A::shape(), B::shape()]
    }
}

impl<A: Traced, B: Traced, C: Traced> TracedTuple<3> for (A, B, C) {
    fn shapes() -> [Vec<Dim>; 3] {
        [A::shape(), B::shape(), C::shape()]
    }
}

/// The shapes a run of an emitted carrier bound, per local.
#[derive(Debug, Default)]
pub struct Trace {
    shapes: HashMap<LocalId, Vec<Dim>>,
}

impl Trace {
    /// Record `value`'s shape as local `id`'s, and pass it on.
    pub fn bind<A: Traced>(&mut self, id: u32, value: A) -> A {
        self.shapes.insert(LocalId(id), A::shape());
        value
    }

    /// [`Self::bind`], for a tuple bound to the locals `ids`.
    pub fn bind_tuple<const N: usize, A: TracedTuple<N>>(&mut self, ids: [u32; N], value: A) -> A {
        for (id, shape) in ids.into_iter().zip(A::shapes()) {
            self.shapes.insert(LocalId(id), shape);
        }
        value
    }

    /// A loop's body, run once: its locals' shapes are their types', the same every iteration.
    pub fn loop_once(&self) -> std::ops::Range<u32> {
        0..1
    }

    /// The shapes recorded, per local.
    pub fn shapes(&self) -> &HashMap<LocalId, Vec<Dim>> {
        &self.shapes
    }
}

/// A program's inputs at `T` rows, each the field its DSL name names (`ExternKind::name`). A
/// vision forward's patch width `Px` and rotary half width `Rh` are its carrier's own (an arch.json
/// may derive them), so each carrier names them in its own scope.
pub struct Inputs<T, Px, Rh, Bh, Ve> {
    pub input_ids: Tokens<T>,
    pub positions: Tokens<T>,
    pub rotary: Rotary,
    /// The sliding class's rotary tables.
    pub rotary_local: Rotary,
    /// `kv_cache[layer]`: every layer's is opaque alike.
    pub kv_cache: KvCache,
    pub block_table: BlockTable,
    /// A head's or drafter's target's final hidden rows, at the target's residual width.
    pub target_hidden: Act<T, Bh>,
    /// A vision forward's patch rows.
    pub pixels: Act<T, Px>,
    /// A vision forward's rotary cos / sin rows.
    pub cos: Act<T, Rh>,
    pub sin: Act<T, Rh>,
    pub cu_seqlens: CuSeqlens,
    pub max_seqlen: MaxSeqlen,
    /// A windowed vision tower's (Qwen2.5-VL) full-attention and window-attention varlen tables.
    pub cu_seqlens_full: CuSeqlens,
    pub cu_seqlens_window: CuSeqlens,
    pub max_seqlen_full: MaxSeqlen,
    pub max_seqlen_window: MaxSeqlen,
    /// Its window permutation and that permutation's inverse, one index per grouped row.
    pub window_index: RowIndex,
    pub reverse_indices: RowIndex,
    /// A SigLIP tower's positional-embedding index per row.
    pub position_ids: Tokens<T>,
    /// A host-interpolated positional embedding (Qwen3.5-VL), one row per patch at the residual
    /// width.
    pub pos_embeds: Act<T, Ve>,
}

impl<T, Px, Rh, Bh, Ve> Inputs<T, Px, Rh, Bh, Ve> {
    /// The inputs, at their shapes.
    pub fn declared() -> Self {
        Self {
            input_ids: Tokens(PhantomData),
            positions: Tokens(PhantomData),
            rotary: Rotary,
            rotary_local: Rotary,
            kv_cache: KvCache,
            block_table: BlockTable,
            target_hidden: Act(PhantomData),
            pixels: Act(PhantomData),
            cos: Act(PhantomData),
            sin: Act(PhantomData),
            cu_seqlens: CuSeqlens,
            max_seqlen: MaxSeqlen,
            cu_seqlens_full: CuSeqlens,
            cu_seqlens_window: CuSeqlens,
            max_seqlen_full: MaxSeqlen,
            max_seqlen_window: MaxSeqlen,
            window_index: RowIndex,
            reverse_indices: RowIndex,
            position_ids: Tokens(PhantomData),
            pos_embeds: Act(PhantomData),
        }
    }
}

/// The op signatures, each named as its `OpKind` is in a carrier.
pub mod ops {
    use super::{Act, Add, BlockTable, CuSeqlens, Div, KvCache, Lit, MaxSeqlen, Mul, PhantomData};
    use super::{
        HiddenSize, KvLoraRank, NumAttentionHeads, QkNopeHeadDim, QkRopeHeadDim, VHeadDim,
    };
    use super::{Rotary, RowIndex, Scalar, Tokens, W1, W2};

    /// `embed(ids: [T], table: [V, D]) -> [T, D]`.
    pub fn embed<T, V, D>(_ids: Tokens<T>, _table: W2<V, D>) -> Act<T, D> {
        Act(PhantomData)
    }

    /// `rmsnorm(x: [T, D], w: [D]) -> [T, D]`.
    pub fn rmsnorm<T, D>(x: Act<T, D>, _w: W1<D>) -> Act<T, D> {
        x
    }

    /// `gemm(x: [T, K], w: [K, N]) -> [T, N]`.
    pub fn gemm<T, K, N>(_x: Act<T, K>, _w: W2<K, N>) -> Act<T, N> {
        Act(PhantomData)
    }

    /// `rope_append(q, k, v, positions, rotary, kv_cache)`: q, k and v, each its own shape.
    pub fn rope_append<T, Q, K, V>(
        q: Act<T, Q>,
        k: Act<T, K>,
        v: Act<T, V>,
        _positions: Tokens<T>,
        _rotary: Rotary,
        _kv_cache: KvCache,
    ) -> (Act<T, Q>, Act<T, K>, Act<T, V>) {
        (q, k, v)
    }

    /// `attention(q: [T, Hd · H], k, v: [T, Hd · Kv], kv_cache, block_table) -> [T, Hd · H]`: one
    /// attention class, whichever its dims: q, k and v share its head width `Hd`, q is `H` heads
    /// of it and k and v `Kv`. rustc infers each class's from its projections (a uniform model's
    /// `HeadDim`, Gemma 4's global class's `GlobalHeadDim`). A product puts the head width first:
    /// [`canonical_mul`]'s order, which the tests pin for every head-width and head-count bound.
    pub fn attention<T, Hd, H, Kv>(
        q: Act<T, Mul<Hd, H>>,
        _k: Act<T, Mul<Hd, Kv>>,
        _v: Act<T, Mul<Hd, Kv>>,
        _kv_cache: KvCache,
        _block_table: BlockTable,
    ) -> Act<T, Mul<Hd, H>> {
        q
    }

    /// `attention(q, k, v)`: an encoder's (bidirectional, no KV cache or block table), with
    /// [`attention`]'s q/k/v law.
    pub fn encoder_attention<T, Hd, H, Kv>(
        q: Act<T, Mul<Hd, H>>,
        _k: Act<T, Mul<Hd, Kv>>,
        _v: Act<T, Mul<Hd, Kv>>,
    ) -> Act<T, Mul<Hd, H>> {
        q
    }

    /// `sliding_attention(..)`: [`attention`]'s signature, for the sliding class.
    pub fn sliding_attention<T, Hd, H, Kv>(
        q: Act<T, Mul<Hd, H>>,
        k: Act<T, Mul<Hd, Kv>>,
        v: Act<T, Mul<Hd, Kv>>,
        kv_cache: KvCache,
        block_table: BlockTable,
    ) -> Act<T, Mul<Hd, H>> {
        attention(q, k, v, kv_cache, block_table)
    }

    /// `rmsnorm_unit(x: [T, D]) -> [T, D]`: an RMS norm with no gain.
    pub fn rmsnorm_unit<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `gelu(x: [T, D]) -> [T, D]`.
    pub fn gelu<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `tanh_softcap(x: [T, D]) -> [T, D]`.
    pub fn tanh_softcap<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `scalar_weight_mul(x: [T, D], w: [1]) -> [T, D]`: `x` times a one-element weight.
    pub fn scalar_weight_mul<T, D>(x: Act<T, D>, _w: W1<Lit<1>>) -> Act<T, D> {
        x
    }

    /// `[T, Hd · H] -> [H · T, Hd]`: a product axis's `H` factor moved into the rows, the per-head
    /// view a per-head norm reads. Only factors move, so the element count is the source's by
    /// construction.
    pub fn split_heads<T, Hd, H>(_x: Act<T, Mul<Hd, H>>) -> Act<Mul<H, T>, Hd> {
        Act(PhantomData)
    }

    /// `[H · T, Hd] -> [T, Hd · H]`: [`split_heads`]'s inverse.
    pub fn merge_heads<T, Hd, H>(_x: Act<Mul<H, T>, Hd>) -> Act<T, Mul<Hd, H>> {
        Act(PhantomData)
    }

    /// `gemma_moe(router_in: [T, D], expert_in: [T, D], router, experts) -> [T, D]`: the sparse
    /// block routes off one row set and runs the experts on the other, both the residual width,
    /// its two weight bundles declared at it.
    pub fn gemma_moe<T, D>(
        _router_in: Act<T, D>,
        expert_in: Act<T, D>,
        _router: W1<D>,
        _experts: W1<D>,
    ) -> Act<T, D> {
        expert_in
    }

    /// `concat(a: [T, D], b: [T, D]) -> [T, D · 2]`: two row sets of one width, side by side.
    pub fn concat<T, D>(_a: Act<T, D>, _b: Act<T, D>) -> Act<T, Mul<D, Lit<2>>> {
        Act(PhantomData)
    }

    /// `rope_query(q, positions, rotary) -> q`: rope on the query alone.
    pub fn rope_query<T, Q>(q: Act<T, Q>, _positions: Tokens<T>, _rotary: Rotary) -> Act<T, Q> {
        q
    }

    /// `shared_attention(q: [T, Hd · H], kv_cache, block_table) -> [T, Hd · H]`: [`attention`]'s
    /// query law, over another model's cache (no k or v of its own).
    pub fn shared_attention<T, Hd, H>(
        q: Act<T, Mul<Hd, H>>,
        _kv_cache: KvCache,
        _block_table: BlockTable,
    ) -> Act<T, Mul<Hd, H>> {
        q
    }

    /// `shared_sliding_attention(..)`: [`shared_attention`]'s signature, for the sliding class.
    pub fn shared_sliding_attention<T, Hd, H>(
        q: Act<T, Mul<Hd, H>>,
        kv_cache: KvCache,
        block_table: BlockTable,
    ) -> Act<T, Mul<Hd, H>> {
        shared_attention(q, kv_cache, block_table)
    }

    /// `export_hidden(x) -> x`: rows a caller reads back.
    pub fn export_hidden<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `gate_split(qg: [T, Q · 2]) -> ([T, Q], [T, Q])`: a gated q projection, query then gate.
    pub fn gate_split<T, Q>(_qg: Act<T, Mul<Q, Lit<2>>>) -> (Act<T, Q>, Act<T, Q>) {
        (Act(PhantomData), Act(PhantomData))
    }

    /// `gate_apply(attn: [T, D], gate: [T, D]) -> [T, D]`: the attention output, gated.
    pub fn gate_apply<T, D>(attn: Act<T, D>, _gate: Act<T, D>) -> Act<T, D> {
        attn
    }

    /// `gate_scale(routed: [T, D], shared: [T, D], gate: [T, 1]) -> [T, D]`: the routed experts'
    /// rows plus the shared expert's, scaled by its one-column gate.
    pub fn gate_scale<T, D>(
        routed: Act<T, D>,
        _shared: Act<T, D>,
        _gate: Act<T, Lit<1>>,
    ) -> Act<T, D> {
        routed
    }

    /// `moe_block(x: [T, D], experts) -> [T, D]`: the routed experts over the residual width, their
    /// bundle declared at it.
    pub fn moe_block<T, D>(x: Act<T, D>, _experts: W1<D>) -> Act<T, D> {
        x
    }

    /// `moe_block(router_in: [T, H], expert_in: [T, L], moe) -> [T, L]`: a latent MoE (Nemotron-3
    /// Super). The router reads the block input, at the bundle's declared width; the experts read
    /// their latent projection, and their weighted sum stays latent.
    pub fn latent_moe_block<T, H, L>(
        _router_in: Act<T, H>,
        expert_in: Act<T, L>,
        _experts: W1<H>,
    ) -> Act<T, L> {
        expert_in
    }

    /// `[T, E] -> [T / M, E · M]`: each `M` consecutive rows become one row (a vision merge). Only
    /// factors move, so the element count is the source's by construction. Its one law beside the
    /// types: `M` divides the row count, a property of the INPUT rows (vision preprocessing builds
    /// whole merge groups), which the types do not check.
    pub fn group_rows<T, M, E>(_x: Act<T, E>) -> Act<Div<T, M>, Mul<E, M>> {
        Act(PhantomData)
    }

    /// `[T / M, E · M] -> [T, E]`: [`group_rows`]' inverse, each grouped row back to its `M` rows
    /// (a vision window permute's return). Only factors move, so the element count is the
    /// source's by construction.
    pub fn ungroup_rows<T, M, E>(_x: Act<Div<T, M>, Mul<E, M>>) -> Act<T, E> {
        Act(PhantomData)
    }

    /// `embedding_gather(x: [R, D], index) -> [R, D]`: `x`'s rows, permuted.
    pub fn embedding_gather<R, D>(x: Act<R, D>, _index: RowIndex) -> Act<R, D> {
        x
    }

    /// `rope_append_interleaved(..)`: [`rope_append`]'s signature, for interleaved rotary pairs.
    pub fn rope_append_interleaved<T, Q, K, V>(
        q: Act<T, Q>,
        k: Act<T, K>,
        v: Act<T, V>,
        _positions: Tokens<T>,
        _rotary: Rotary,
        _kv_cache: KvCache,
    ) -> (Act<T, Q>, Act<T, K>, Act<T, V>) {
        (q, k, v)
    }

    /// `kv_append(k, v, kv_cache) -> (k, v)`: an un-roped layer's step written to its cache.
    pub fn kv_append<T, K, V>(
        k: Act<T, K>,
        v: Act<T, V>,
        _kv_cache: KvCache,
    ) -> (Act<T, K>, Act<T, V>) {
        (k, v)
    }

    /// `relu2(x) -> x`: squared ReLU.
    pub fn relu2<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `sink_attention(q, k, v, sinks: [H], kv_cache, block_table)`: [`attention`]'s law, with one
    /// learned sink logit per query head.
    pub fn sink_attention<T, Hd, H, Kv>(
        q: Act<T, Mul<Hd, H>>,
        _k: Act<T, Mul<Hd, Kv>>,
        _v: Act<T, Mul<Hd, Kv>>,
        _sinks: W1<H>,
        _kv_cache: KvCache,
        _block_table: BlockTable,
    ) -> Act<T, Mul<Hd, H>> {
        q
    }

    /// `sink_sliding_attention(..)`: [`sink_attention`]'s signature, for the sliding class.
    pub fn sink_sliding_attention<T, Hd, H, Kv>(
        q: Act<T, Mul<Hd, H>>,
        _k: Act<T, Mul<Hd, Kv>>,
        _v: Act<T, Mul<Hd, Kv>>,
        _sinks: W1<H>,
        _kv_cache: KvCache,
        _block_table: BlockTable,
    ) -> Act<T, Mul<Hd, H>> {
        q
    }

    /// `gptoss_moe(x: [T, D], mlp) -> [T, D]`: a fused routed MoE with per-expert biases, its bundle at the
    /// router's width.
    pub fn gptoss_moe<T, D>(x: Act<T, D>, _mlp: W1<D>) -> Act<T, D> {
        x
    }

    /// A Gated-DeltaNet layer's packed `in_proj_qkv` row, the kernel's split: q and k at key width
    /// (`2 · Kd · Hk`), then v at value width (`Hv · Vd`), in [`canonical_add`]'s order (the value
    /// width first).
    pub type GdnQkv<Kd, Hk, Hv, Vd> = Add<Mul<Hv, Vd>, Mul<Kd, Mul<Hk, Lit<2>>>>;

    /// The drafter's dynamic-convolution base kernel: two stages of `K` taps over `D` channels.
    pub type DraftConvBase<K, D> = Mul<K, Mul<D, Lit<2>>>;

    /// The drafter's dynamic weights: one per `G` channels of the base kernel.
    pub type DraftConvDynamic<K, D, G> = Div<DraftConvBase<K, D>, G>;

    /// A drafter context row's keys then values: `2 · Kv · Hd`.
    pub type DraftKv<Hd, Kv> = Mul<Hd, Mul<Kv, Lit<2>>>;

    /// Splash's packed drafter q|k|v row: `H` query heads, then [`DraftKv`].
    pub type DraftQkv<Hd, H, Kv> = Add<Mul<Hd, H>, DraftKv<Hd, Kv>>;

    /// `gated_delta_net(qkv, z, a, b, linear_attn) -> [T, Hv · Vd]`: a Gated-DeltaNet layer. `qkv`
    /// is `in_proj_qkv`'s packed row, the kernel's split: q and k at key width (`2 · Kd · Hk`),
    /// then v at value width (`Hv · Vd`); `z` is one value row, `a` and `b` one scalar per value
    /// head. A sum is written in [`canonical_add`]'s order (the value width first).
    pub fn gated_delta_net<T, Kd, Hk, Hv, Vd, D>(
        _qkv: Act<T, GdnQkv<Kd, Hk, Hv, Vd>>,
        z: Act<T, Mul<Hv, Vd>>,
        _a: Act<T, Hv>,
        _b: Act<T, Hv>,
        _linear_attn: W1<D>,
    ) -> Act<T, Mul<Hv, Vd>> {
        z
    }

    /// `mamba2(zxbcdt, mamba) -> [T, Di]`: a Mamba-2 mixer. `zxbcdt` is `in_proj`'s packed row,
    /// the kernel's split: the conv channels `C` (`xBC`), the gate `z` at `Di` (`mamba_d_inner`)
    /// and one `dt` per head (`Nh`), in [`canonical_add`]'s order.
    pub fn mamba2<T, C, Di, Nh, D>(
        _zxbcdt: Act<T, Add<C, Add<Di, Nh>>>,
        _mamba: W1<D>,
    ) -> Act<T, Di> {
        Act(PhantomData)
    }

    /// `mla_split(kv_a: [T, kv_lora_rank + qk_rope_head_dim]) -> (kv_latent, k_pe)`: the
    /// compressed KV, then the one shared rope key.
    pub fn mla_split<T>(
        _kv_a: Act<T, Add<KvLoraRank, QkRopeHeadDim>>,
    ) -> (Act<T, KvLoraRank>, Act<T, QkRopeHeadDim>) {
        (Act(PhantomData), Act(PhantomData))
    }

    /// `mla_attention(q, kv_b, k_pe, positions, rotary, kv_cache, block_table) -> [T, heads ·
    /// v_head_dim]`: multi-head latent attention. Per head, `q` is a query's no-rope and rope
    /// parts, `kv_b` a key's no-rope part and a value; `k_pe` is the shared rope key.
    pub fn mla_attention<T>(
        _q: Act<T, Mul<NumAttentionHeads, Add<QkNopeHeadDim, QkRopeHeadDim>>>,
        _kv_b: Act<T, Mul<NumAttentionHeads, Add<QkNopeHeadDim, VHeadDim>>>,
        _k_pe: Act<T, QkRopeHeadDim>,
        _positions: Tokens<T>,
        _rotary: Rotary,
        _kv_cache: KvCache,
        _block_table: BlockTable,
    ) -> Act<T, Mul<NumAttentionHeads, VHeadDim>> {
        Act(PhantomData)
    }

    /// `pos_embed(position_ids: [T], table: [P, E]) -> [T, E]`: a vision tower's learned
    /// positional rows.
    pub fn pos_embed<T, P, E>(_ids: Tokens<T>, _table: W2<P, E>) -> Act<T, E> {
        Act(PhantomData)
    }

    /// `avg_pool_2d(x: [T, E]) -> [T / P, E]`: each `P` (`vision_pool_factor`) patch rows pooled
    /// into one. `P` is the carrier's, passed by the emit pass: the input does not name it.
    pub fn avg_pool_2d<P, T, E>(_x: Act<T, E>) -> Act<Div<T, P>, E> {
        Act(PhantomData)
    }

    /// `draft_window_store` / `draft_window_load(context: [T, hidden_size], positions) ->
    /// context`: the drafter's context window.
    pub fn draft_window_store<T>(
        context: Act<T, HiddenSize>,
        _positions: Tokens<T>,
    ) -> Act<T, HiddenSize> {
        context
    }

    /// See [`draft_window_store`].
    pub fn draft_window_load<T>(
        context: Act<T, HiddenSize>,
        _positions: Tokens<T>,
    ) -> Act<T, HiddenSize> {
        context
    }

    /// `draft_context_kv(kv: [T, 2 · Kv · Hd], k_norm: [Hd], positions, rotary) -> kv`: a context
    /// row's keys then values.
    pub fn draft_context_kv<T, Hd, Kv>(
        kv: Act<T, DraftKv<Hd, Kv>>,
        _k_norm: W1<Hd>,
        _positions: Tokens<T>,
        _rotary: Rotary,
    ) -> Act<T, DraftKv<Hd, Kv>> {
        kv
    }

    /// `draft_conv(x: [T, D], dynamic: [T, 2 · K · D / G], base: [2 · K · D]) -> [T, D]`: the
    /// drafter's dynamic convolution, `K` taps (`conv_kernel_size`), one dynamic weight per `G`
    /// channels (`conv_group_size`), two stages.
    pub fn draft_conv<T, D, K, G>(
        x: Act<T, D>,
        _dynamic: Act<T, DraftConvDynamic<K, D, G>>,
        _base: W1<DraftConvBase<K, D>>,
    ) -> Act<T, D> {
        x
    }

    /// `draft_conv(x, dynamic, base, residual: [T, D])`: [`draft_conv`]'s finishing stage, onto
    /// a residual like `x`.
    pub fn draft_conv_residual<T, D, K, G>(
        x: Act<T, D>,
        _dynamic: Act<T, DraftConvDynamic<K, D, G>>,
        _base: W1<DraftConvBase<K, D>>,
        _residual: Act<T, D>,
    ) -> Act<T, D> {
        x
    }

    /// `draft_attention(qkv: [T, Hd · H + 2 · Hd · Kv], q_norm, k_norm: [Hd], positions, rotary)
    /// -> [T, Hd · H]`: Splash's packed q|k|v row in, an attention row out.
    pub fn draft_attention<T, Hd, H, Kv>(
        _qkv: Act<T, DraftQkv<Hd, H, Kv>>,
        _q_norm: W1<Hd>,
        _k_norm: W1<Hd>,
        _positions: Tokens<T>,
        _rotary: Rotary,
    ) -> Act<T, Mul<Hd, H>> {
        Act(PhantomData)
    }

    /// `draft_select(logits: [T, V], selector: [T, R], predecessor, successor: [R · V]) ->
    /// logits`: the drafter's candidate selector, its codebooks one row of `V` per rank.
    pub fn draft_select<T, V, R>(
        logits: Act<T, V>,
        _selector: Act<T, R>,
        _predecessor: W1<Mul<R, V>>,
        _successor: W1<Mul<R, V>>,
    ) -> Act<T, V> {
        logits
    }

    /// `mean(x: [T, D]) -> [T, D]`: each row's mean, broadcast over it.
    pub fn mean<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `sub(a: [T, D], b: [T, D]) -> [T, D]`.
    pub fn sub<T, D>(a: Act<T, D>, _b: Act<T, D>) -> Act<T, D> {
        a
    }

    /// `bias_add(x: [T, D], b: [D]) -> [T, D]`.
    pub fn bias_add<T, D>(x: Act<T, D>, _b: W1<D>) -> Act<T, D> {
        x
    }

    /// `gelu_erf(x: [T, D]) -> [T, D]`.
    pub fn gelu_erf<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `quick_gelu(x: [T, D]) -> [T, D]`.
    pub fn quick_gelu<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `vision_rope(q, k, cos, sin) -> (q, k)`: the 2-D rope, each its own shape.
    pub fn vision_rope<T, Q, K, C>(
        q: Act<T, Q>,
        k: Act<T, K>,
        _cos: Act<T, C>,
        _sin: Act<T, C>,
    ) -> (Act<T, Q>, Act<T, K>) {
        (q, k)
    }

    /// `varlen_attention(q, k, v: [T, D], cu_seqlens, max_seqlen) -> [T, D]`: a vision tower's
    /// per-image attention, q, k and v at one width.
    pub fn varlen_attention<T, D>(
        q: Act<T, D>,
        _k: Act<T, D>,
        _v: Act<T, D>,
        _cu_seqlens: CuSeqlens,
        _max_seqlen: MaxSeqlen,
    ) -> Act<T, D> {
        q
    }

    /// `as_dim(x: [T, From]) -> [T, To]`: `x`'s width stated as another bound. A model compiles
    /// through it only where the two are equal: the cast (`Program::dim_casts`) makes the model's
    /// facts a lent target (`Target::lent`), whose law `ModelFacts::new` holds when its `FACTS`
    /// const evaluates. Here it is the type the program reads the rows at.
    pub fn as_dim<T, From, To>(_x: Act<T, From>) -> Act<T, To> {
        Act(PhantomData)
    }

    /// `x * s` for a scalar `s`: `x`'s shape.
    pub fn scale<T, D>(x: Act<T, D>, _s: Scalar) -> Act<T, D> {
        x
    }

    /// `x + s` for a scalar `s` (Gemma's `(1 + w)` norm weight): `x`'s shape, an activation's or
    /// a weight's.
    pub fn offset<X>(x: X, _s: Scalar) -> X {
        x
    }

    /// `add(a: [T, D], b: [T, D]) -> [T, D]`.
    pub fn add<T, D>(a: Act<T, D>, _b: Act<T, D>) -> Act<T, D> {
        a
    }

    /// `a * b`, elementwise: `[T, D] · [T, D] -> [T, D]`.
    pub fn mul<T, D>(a: Act<T, D>, _b: Act<T, D>) -> Act<T, D> {
        a
    }

    /// `silu(x: [T, D]) -> [T, D]`.
    pub fn silu<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }

    /// `capture_hidden(x: [T, D]) -> [T, D]`: the rows a drafter reads.
    pub fn capture_hidden<T, D>(x: Act<T, D>) -> Act<T, D> {
        x
    }
}

// The emit pass's output: the marker of every bound the signatures above and the carriers' weights
// name (`typed_emit::SIGNATURE_BOUNDS` and each `weights.json`), one module per carrier, and
// `typed_program`, which names each carrier's traced function by its arch.
include!(concat!(env!("OUT_DIR"), "/typed_carriers.rs"));

/// A traced shape and `shape::infer`'s for the same local, under one model's bounds.
#[derive(Debug)]
pub struct ParityMismatch {
    pub local: LocalId,
    pub name: String,
    pub line: Option<u32>,
    pub traced: Option<Vec<Dim>>,
    pub inferred: Option<Vec<Dim>>,
}

/// TRANSITIONAL — deleted with `shape.rs`'s solver in phase C, when the trace becomes the source.
///
/// The carrier `arch`'s traced shapes against `inferred` (what `shape::infer` closed for one model),
/// local by local, each evaluated under `bounds` with `num_tokens` set. A local `inferred` shapes
/// that `classified` never bound is one reshape recovery added, which the trace (over the
/// classifier's program) cannot name: those are returned apart, as `recovered`.
///
/// `None` when `arch` has no typed program.
pub fn check_parity(
    arch: &str,
    classified: &crate::classified::Program,
    inferred: &HashMap<LocalId, Vec<Dim>>,
    bounds: &BTreeMap<String, u64>,
) -> Option<Parity> {
    let forward = typed_program(arch)?;
    let mut trace = Trace::default();
    forward(&mut trace);
    // A row count no bound equals, so a dim that confused rows with features shows.
    const NUM_TOKENS: u64 = 7919;
    let mut bounds = bounds.clone();
    bounds.insert(NumTokens::NAME.to_string(), NUM_TOKENS);
    let eval = |shape: &Vec<Dim>| -> Option<Vec<u64>> {
        shape
            .iter()
            .map(|d| crate::shape::eval_closed_dim(d, &bounds))
            .collect()
    };
    let mut mismatches = Vec::new();
    let mut compared = 0;
    for id in classified.locals.ids() {
        let traced = trace.shapes().get(&id);
        let inferred = inferred.get(&id);
        let agree = match (traced, inferred) {
            (Some(t), Some(i)) => {
                compared += 1;
                eval(t).is_some() && eval(t) == eval(i)
            }
            (None, None) => true,
            _ => false,
        };
        if !agree {
            mismatches.push(ParityMismatch {
                local: id,
                name: classified.locals.name(id).to_string(),
                line: classified.locals.line(id),
                traced: traced.cloned(),
                inferred: inferred.cloned(),
            });
        }
    }
    let bound: std::collections::HashSet<LocalId> = classified.locals.ids().collect();
    let mut recovered: Vec<LocalId> = (inferred.keys())
        .filter(|id| !bound.contains(id))
        .copied()
        .collect();
    recovered.sort_by_key(|id| id.0);
    Some(Parity {
        compared,
        mismatches,
        recovered,
    })
}

/// [`check_parity`]'s verdict.
#[derive(Debug)]
pub struct Parity {
    /// How many locals both sides shaped, and so were compared.
    pub compared: usize,
    /// Every local whose shapes differ, or that one side shaped and the other did not.
    pub mismatches: Vec<ParityMismatch>,
    /// The locals reshape recovery added: none of the classifier's, so outside the trace.
    pub recovered: Vec<LocalId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generic attention signature matches `Mul<Hd, H>`: a product of a head width and a
    /// head count written in [`canonical_mul`]'s order puts the head width first. Pinned for every
    /// head-width and head-count bound the signatures name, so a `weights.json` product of them is
    /// the type the signature expects.
    #[test]
    fn a_head_width_comes_first_in_its_products() {
        for (width, count) in [
            (HeadDim::dim(), NumAttentionHeads::dim()),
            (HeadDim::dim(), NumKeyValueHeads::dim()),
            (GlobalHeadDim::dim(), NumAttentionHeads::dim()),
            (GlobalHeadDim::dim(), NumGlobalKeyValueHeads::dim()),
        ] {
            assert_eq!(
                canonical_mul(vec![count.clone(), width.clone()]),
                Dim::Mul(vec![width, count])
            );
        }
    }
}
