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
//! literals into the value→const door the lowering dispatches through.
//!
//! Nothing here is written by hand. Dropping a `config.json` into
//! `crates/models/arch/<arch>/configs/` re-runs the build script and adds its geometry; there is no
//! list of models, no list of geometry triples and no list of head dims in any source file.
//!
//! THE TWO HALVES:
//! * the VALUE side is `scratchy_ir::model_facts::Attention` — the geometry as one typed value,
//!   built by the fill with the GQA divisibility proven in `HeadGeometry::new`, so no consumer
//!   downstream re-derives or re-checks a group size. The tape's attention ops carry their class
//!   and that one value, minted at the bridge; every consumer reads the class's geometry off the
//!   op itself.
//! * [`with_config_head_dim`] is the DOOR — the only place a head dim becomes a const, with the
//!   arms the build script generated. Same shape as `sdsc_abstract::with_baked_rung`, which is the
//!   decode ladder's door. The door and the value side compile on every target.

// The build script's output: `GEOMETRY_SOURCE_STEMS` plus the callback macro
// `for_each_config_head_dim!`, whose body is the head dims it parsed out of the model configs.
// Included before the door below, which invokes it.
include!(concat!(env!("OUT_DIR"), "/config_geometry.rs"));

// ⭐ ONE HEAD-DIM TYPE. The head counts live in `scratchy_ir::model_facts` (#1083), where
// their laws are stated at the constructors; this module once carried a second, lawless copy,
// and a value crossing between the two was one fact stated twice. The door below is typed on
// the facts' own `HeadDim`.
use scratchy_ir::model_facts::HeadDim;

/// What a head-dim-parameterised lowering runs once its head dim is a const — the consumer side of
/// [`with_config_head_dim`].
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
/// at a width this workspace has never seen arrives as a `config.json`, not as an edit here. Each
/// arm's const passes through [`scratchy_ir::model_facts::HeadDim::new`], so rustc checks the
/// positive-head-dim law at every instantiation.
pub fn with_config_head_dim<C: OnHeadDim>(head_dim: HeadDim, consumer: C) -> Option<C::Out> {
    macro_rules! head_dim_arms {
        ($($hd:literal),* $(,)?) => {
            match head_dim.get() {
                $($hd => Some(consumer.on_head_dim::<{
                    scratchy_ir::model_facts::HeadDim::new($hd).get()
                }>()),)*
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
