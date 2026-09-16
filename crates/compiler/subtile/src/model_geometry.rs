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

// ⭐ THE FOUR HEAD COUNTS NOW LIVE IN `ktir-superdsc`, RE-EXPORTED HERE SO EVERY PATH STILL RESOLVES.
// The extraction moved `sdsc_abstract` + `addr` (the typed device facts)
// into that leaf crate, and they cannot go without `Gqa`: `addr::shape::Shape::gqa` calls
// `Gqa::new` and `sdsc_abstract`'s `KvHead::of_query`/`group_first_query` take one. It was the ONLY
// unresolved path when those two moved.
//
// ⛔ ONLY THE COUNTS AND THE VALUE SIDE WENT. Everything in this file that knows about MODELS
// stayed: the `include!` of the build script's arms, both value→const doors, and
// `geometry_sources()`. That is the line the extraction must not cross — a door's arm set is the
// CONSUMER's model inventory, so no model inventory reached the leaf crate. `ModelAttnGeometry` and
// its mint (where the GQA divisibility is still the one place it is decided) followed the counts,
// because the lowering's own request type names it.
pub use ktir_superdsc::head_counts::{Gqa, HeadDim, KvHeads, QueryHeads};
// The VALUE side, named directly rather than re-exported: this module's two doors take one as a
// parameter, and every other call site in the tree names `ktir_superdsc::head_counts` itself.
use ktir_superdsc::head_counts::ModelAttnGeometry;

/// What the SDSC lowering runs once its geometry is a const — the consumer side of
/// [`with_config_attn_geometry`]. The method is generic over the three consts because that is the
/// whole point of the door: the body receives the geometry AS CONSTS, with the
/// [`AttnGeometry`](crate::sdsc_abstract::AttnGeometry) witness that spends the divisibility proof,
/// so every layout term it derives is the compiler's arithmetic.
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
