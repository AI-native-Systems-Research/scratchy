// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE MODEL'S HEAD GEOMETRY, FROM `config.json` TO THE TYPE SYSTEM.
//!
//! scratchy is a per-model compiler. `arch-<name>`/`<stem>` (scratchy-models)
//! and `<preset>` (scratchy-quantizations) Cargo features pin the model set
//! at cargo-build time, the
//! `#[forward]` proc-macro parses that model's `config.json`, and the SuperDSC lowering runs INSIDE
//! the macro's expansion — so a bundle's query-head count, kv-head count and head dim are as
//! constant as its op list. The emitter wants them as const generics, because a head dim
//! parameterises the device layout (slabs = head_dim/lanes; the head-major collapse is valid only
//! at head_dim == lanes) and a branch on a const is reviewable where a branch on a value is not.
//!
//! A const generic must be monomorphised when THIS crate is compiled, one compilation before any
//! expansion — so the instantiations cannot be emitted by the macro that knows the geometry. They
//! are emitted by [`build.rs`](../build.rs) instead, which reads the same `config.json` files the
//! macro will read and writes their geometries out as literal tokens. This module turns those
//! literals into the two value→const doors the lowering dispatches through.
//!
//! Nothing here is written by hand. Dropping a `config.json` into
//! `crates/models/arch/<arch>/configs/` re-runs the build script and adds its geometry; there is no
//! list of models, no list of geometry triples and no list of head dims in any source file.
//!
//! THE TWO HALVES:
//! * [`ModelAttnGeometry`] is the VALUE side — the geometry as the tape carries it, minted once
//!   from the config numbers with the GQA divisibility proven at the mint, so no consumer
//!   downstream re-derives or re-checks a group size.
//! * [`with_config_attn_geometry`] / [`with_config_head_dim`] are the DOORS — the only places a
//!   geometry value becomes a const, with the arms the build script generated. Same shape as
//!   `sdsc_abstract::with_baked_rung`, which is the decode ladder's door.

// The build script's output: `GEOMETRY_SOURCE_STEMS` plus the two callback macros
// `for_each_config_attn_geometry!` / `for_each_config_head_dim!`, whose bodies are the literals it
// parsed out of the model configs. Included before the doors below, which invoke them.
include!(concat!(env!("OUT_DIR"), "/config_geometry.rs"));

/// A COUNT along one head axis of a model. Distinct types, so a query-head count cannot be handed
/// to something that wanted a kv-head count — under GQA they are different numbers, and passing one
/// where the other belongs addresses another head's keys, which is fluent wrong output rather than
/// a fault. The same discipline as `addr::shape`'s `Rows`/`PadRows`/`Slabs`, and this is where
/// [`Gqa`] — the group size those two counts determine — lives too.
macro_rules! head_counts {
    ($($t:ident => $doc:literal),* $(,)?) => { $(
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        pub struct $t(u32);
        impl $t {
            /// The count as the config declares it.
            pub const fn new(v: u32) -> Self { Self(v) }
            /// The count as a value, for the derivations that need one.
            pub const fn get(self) -> u32 { self.0 }
        }
    )* };
}

head_counts! {
    QueryHeads => "How many QUERY heads the model has — `num_attention_heads`.",
    KvHeads    => "How many KEY/VALUE heads the model has — `num_key_value_heads`. Equal to \
                   [`QueryHeads`] only for a non-GQA model, which is exactly why it is a different \
                   type.",
    HeadDim    => "How wide ONE head is — `head_dim`. Not a hidden size, not a stick width, not a \
                   lane count: the three quantities a head dim is most often confused with, all of \
                   which are also small powers of two.",
    Gqa        => "Query heads per kv head. Never written by a call site: it is \
                   [`ModelAttnGeometry`]'s own division, performed once at the mint where the \
                   divisibility is proven.",
}

/// ⭐⭐⭐⭐⭐ ONE MODEL'S ATTENTION GEOMETRY AS A VALUE, WITH ITS GQA PROOF SPENT AT THE MINT.
///
/// The tape carries this — not three adjacent `u32` fields — so the head counts cannot be
/// transposed in a struct literal, a head dim from another op cannot be paired with them, and the
/// group size cannot be recomputed anywhere downstream.
///
/// [`mint`](Self::mint) is the ONE place divisibility is decided. `None` there is a config whose
/// kv-head count does not divide its query-head count: such a model has no GQA grouping at all, and
/// the arithmetic that used to paper over it — `(nqh / nkvh.max(1)).max(1)` — yields a plausible
/// group size whose attention reads another head's keys. Holding a `ModelAttnGeometry` IS holding
/// the fact that its [`gqa`](Self::gqa) is exact, so no consumer checks again.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ModelAttnGeometry {
    nqh: QueryHeads,
    nkvh: KvHeads,
    hd: HeadDim,
    gqa: Gqa,
}

impl ModelAttnGeometry {
    /// THE MINT — and the one evaluation of the GQA division in the whole pipeline.
    ///
    /// `None` for a zero count, a zero head dim, or a kv-head count that does not divide the
    /// query-head count. The caller is the parse boundary from the model config (the macro's
    /// wavefront bridge), so a `None` there is a build error naming the model, not a runtime arm.
    pub const fn mint(nqh: QueryHeads, nkvh: KvHeads, hd: HeadDim) -> Option<ModelAttnGeometry> {
        if nqh.get() == 0 || nkvh.get() == 0 || hd.get() == 0 {
            return None;
        }
        if !nqh.get().is_multiple_of(nkvh.get()) {
            return None;
        }
        Some(ModelAttnGeometry {
            nqh,
            nkvh,
            hd,
            gqa: Gqa::new(nqh.get() / nkvh.get()),
        })
    }

    /// The query-head count.
    pub const fn nqh(self) -> QueryHeads {
        self.nqh
    }

    /// The kv-head count.
    pub const fn nkvh(self) -> KvHeads {
        self.nkvh
    }

    /// The head dim.
    pub const fn hd(self) -> HeadDim {
        self.hd
    }

    /// The GQA group size, divided out at the mint. Exact by construction — there is no rounding
    /// and no `max(1)` here, because a non-dividing pair never became a value of this type.
    pub const fn gqa(self) -> Gqa {
        self.gqa
    }

    /// The `[rows, nqh*hd]` token-stream width — the query projection's and the attention output's
    /// column count. The ONE place this product is written.
    pub const fn q_width(self) -> u32 {
        self.nqh.get() * self.hd.get()
    }

    /// The `[rows, nkvh*hd]` kv-stream width — one K or V projection's column count, narrower than
    /// [`q_width`](Self::q_width) by exactly the GQA group size.
    pub const fn kv_width(self) -> u32 {
        self.nkvh.get() * self.hd.get()
    }
}

impl std::fmt::Display for ModelAttnGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nqh={} nkvh={} head_dim={}",
            self.nqh.get(),
            self.nkvh.get(),
            self.hd.get()
        )
    }
}

/// What the SDSC lowering runs once its geometry is a const — the consumer side of
/// [`with_config_attn_geometry`]. The method is generic over the three consts because that is the
/// whole point of the door: the body receives the geometry AS CONSTS, with the
/// [`AttnGeometry`](crate::sdsc_abstract::AttnGeometry) witness that spends the divisibility proof,
/// so every layout term it derives is the compiler's arithmetic.
#[cfg(feature = "superdsc")]
pub trait OnAttnGeometry {
    /// What the dispatch produces.
    type Out;
    /// The arm's body, at the geometry the configs declared.
    fn on_geometry<const NQH: u32, const NKVH: u32, const HD: u32>(
        self,
        geom: crate::sdsc_abstract::AttnGeometry<NQH, NKVH, HD>,
    ) -> Self::Out;
}

/// ⭐ THE VALUE→CONST DOOR FOR A MODEL'S ATTENTION GEOMETRY. The arms are the geometries the build
/// script parsed out of `crates/models/arch/*/configs/*.json`, one per distinct triple.
///
/// `None` is a geometry no config in this workspace declares — the caller refuses the bake, loudly,
/// naming the geometry and the configs that were read. The fix is a `config.json`, never an edit
/// here: there is no list in this file to grow.
#[cfg(feature = "superdsc")]
pub fn with_config_attn_geometry<C: OnAttnGeometry>(
    geometry: ModelAttnGeometry,
    consumer: C,
) -> Option<C::Out> {
    macro_rules! geometry_arms {
        ($(($nqh:literal, $nkvh:literal, $hd:literal)),* $(,)?) => {
            match (geometry.nqh().get(), geometry.nkvh().get(), geometry.hd().get()) {
                $(($nqh, $nkvh, $hd) => Some(consumer.on_geometry(
                    crate::sdsc_abstract::AttnGeometry::<$nqh, $nkvh, $hd>::minted(),
                )),)*
                _ => None,
            }
        };
    }
    for_each_config_attn_geometry!(geometry_arms)
}

/// What a WHOLE-MODEL lowering runs once every one of its seven numbers is a const — the consumer
/// side of [`with_config_model`].
///
/// ⭐⭐ SEVEN, NOT THREE, BECAUSE A LOWERING SPECIALISES ON MORE THAN ATTENTION. Whether a row fits
/// the scratchpad is `hidden * rows * 2` against the LX capacity; whether it needs a lane mask is
/// `hidden % elements-per-stick`. Both decide which OPS are emitted, so both have to be constants —
/// a `const fn` reading a runtime hidden size folds to nothing.
#[cfg(feature = "superdsc")]
pub trait OnModel {
    /// What the dispatch produces.
    type Out;
    /// The arm's body, at the numbers one config declared.
    fn on_model<
        const NQH: u32,
        const NKVH: u32,
        const HD: u32,
        const HIDDEN: u32,
        const LAYERS: u32,
        const FFN: u32,
        const VOCAB: u32,
    >(
        self,
    ) -> Self::Out;
}

/// ⭐ THE VALUE→CONST DOOR FOR A WHOLE MODEL. The arms are what the build script parsed out of
/// `crates/models/arch/*/configs/*.json`, one per distinct seven-tuple.
///
/// `None` is a model no config in this workspace declares, and the caller must refuse the bake
/// naming it. The fix is a `config.json`, never an edit here — there is no list in this file to
/// grow, which is the same discipline [`with_config_attn_geometry`] follows.
///
/// ⛔ A MODEL MISSING ANY ONE OF THE SEVEN YIELDS NO ARM. The build script requires all of them
/// present and non-zero, so a config without an `intermediate_size` is refused BY NAME here rather
/// than instantiated against a default nobody wrote down.
#[cfg(feature = "superdsc")]
pub fn with_config_model<C: OnModel>(
    nqh: u32,
    nkvh: u32,
    hd: u32,
    hidden: u32,
    layers: u32,
    ffn: u32,
    vocab: u32,
    consumer: C,
) -> Option<C::Out> {
    macro_rules! model_arms {
        ($(($nqh:literal, $nkvh:literal, $hd:literal, $hidden:literal, $layers:literal, $ffn:literal, $vocab:literal)),* $(,)?) => {
            match (nqh, nkvh, hd, hidden, layers, ffn, vocab) {
                $(($nqh, $nkvh, $hd, $hidden, $layers, $ffn, $vocab) => Some(
                    consumer.on_model::<$nqh, $nkvh, $hd, $hidden, $layers, $ffn, $vocab>(),
                ),)*
                _ => None,
            }
        };
    }
    for_each_config_model!(model_arms)
}

/// What a head-dim-parameterised lowering runs once its head dim is a const — the consumer side of
/// [`with_config_head_dim`], and the same discipline as [`OnAttnGeometry`].
pub trait OnHeadDim {
    /// What the dispatch produces.
    type Out;
    /// The arm's body, at the head dim the configs declared.
    fn on_head_dim<const HD: u32>(self) -> Self::Out;
}

/// ⭐ THE VALUE→CONST DOOR FOR A HEAD DIM. The arms are every head dim the build script found in the
/// workspace's model configs — the decoder's `head_dim`, a per-layer-class model's
/// `global_head_dim`, and MLA's `qk_rope_head_dim`.
///
/// `None` is a head dim no config declares; the caller refuses, loudly. A model whose rotary runs
/// at a width this workspace has never seen arrives as a `config.json`, not as an edit here.
pub fn with_config_head_dim<C: OnHeadDim>(head_dim: HeadDim, consumer: C) -> Option<C::Out> {
    macro_rules! head_dim_arms {
        ($($hd:literal),* $(,)?) => {
            match head_dim.get() {
                $($hd => Some(consumer.on_head_dim::<$hd>()),)*
                _ => None,
            }
        };
    }
    for_each_config_head_dim!(head_dim_arms)
}

/// The model stems the build script read, for a refusal message: a bake that names a geometry no
/// arm covers should say WHICH configs were in scope — the scan itself is unfiltered (every config
/// under `crates/models/arch/configs/`), so an empty list means the configs directory itself is
/// missing or empty, not that a model was scoped out.
pub fn geometry_sources() -> String {
    if GEOMETRY_SOURCE_STEMS.is_empty() {
        "no model configs found under crates/models/arch/configs/".to_string()
    } else {
        GEOMETRY_SOURCE_STEMS.join(", ")
    }
}
