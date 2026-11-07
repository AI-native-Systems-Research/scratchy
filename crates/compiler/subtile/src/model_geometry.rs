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
//! macro will read and writes ONE MARKER TYPE PER CONFIG ([`config_models`]): its rope classes'
//! head dims as typenum types. This module turns those types into the rope's lowering.
//!
//! Nothing here is written by hand. Dropping a `config.json` into
//! `crates/models/arch/<arch>/configs/` re-runs the build script and adds its type; there is no
//! list of models, no list of geometry triples and no list of head dims in any source file.
//!
//! THE TWO HALVES:
//! * the VALUE side is `scratchy_ir::model_facts::Attention` — the geometry as one typed value,
//!   built by the fill with the GQA divisibility proven in `HeadGeometry::new`, so no consumer
//!   downstream re-derives or re-checks a group size. The tape's attention ops carry their class
//!   and that one value, minted at the bridge; every consumer reads the class's geometry off the
//!   op itself.
//! * the TYPE side is the config's marker type ([`ConfigRopes`]). A head dim never becomes a const
//!   by a match on its value: [`rope_bridge::RopeHd`] names each head dim's `rope_at` instantiation
//!   by its typenum type, and the bake crosses ONE door, on model identity
//!   ([`with_config_model`]: its config's stem → that config's marker type).

// The build script's output: `GEOMETRY_SOURCE_STEMS`, the marker types (`config_models`), and the
// callback macros `for_each_config_model!` and (under `spyre`) `for_each_config_head_dim_type!`.
include!(concat!(env!("OUT_DIR"), "/config_geometry.rs"));

/// The model stems the build script read, for a refusal message: a bake whose stem no config
/// states should say WHICH configs were in scope — the scan itself is unfiltered (every config
/// under `crates/models/arch/configs/`), so an empty list means the configs directory itself is
/// missing or empty, not that a model was scoped out.
pub fn geometry_sources() -> String {
    if GEOMETRY_SOURCE_STEMS.is_empty() {
        "no model configs found under crates/models/arch/configs/".to_string()
    } else {
        GEOMETRY_SOURCE_STEMS.join(", ")
    }
}

// ══════════════════════════════════════════════════════════════════════════════════════════════
//  EACH CONFIG'S ROPE CLASSES AS A TYPE, AND ONE DOOR ON MODEL IDENTITY.
// ══════════════════════════════════════════════════════════════════════════════════════════════

/// A model config's rope classes, as types: the build script writes one marker type per config
/// ([`config_models`]), whose head dims are typenum unsigneds read out of that `config.json`.
pub trait ConfigRopes {
    /// The config's stem.
    const STEM: &'static str;
    /// The base class's (`rotary_local`) head dim.
    type Base: scratchy_tensors::typed::typenum::Unsigned;
    /// The global class's (plain `rotary`) head dim.
    type Global: scratchy_tensors::typed::typenum::Unsigned;
    /// The rope adapter of the config's classes in the tensor-parallel world `T`, read off the
    /// two types above.
    #[cfg(feature = "spyre")]
    fn ropes<T: ktir_superdsc::collective::Tp>() -> rope_bridge::ModelRopes<T>;
}

/// What runs once the bake's model is a type — the consumer side of [`with_config_model`].
pub trait OnConfigModel {
    /// What the dispatch produces.
    type Out;
    /// The body, at the model's own marker type.
    fn on_model<M: ConfigRopes>(self) -> Self::Out;
}

/// ⛔ THE DOOR ON MODEL IDENTITY: the bake's config stem → that config's marker type. The arms are
/// every config the build script read; `None` is a stem no config states. A shape value never
/// crosses here: what crosses is which model the bake was handed.
pub fn with_config_model<C: OnConfigModel>(stem: &str, consumer: C) -> Option<C::Out> {
    macro_rules! model_arms {
        ($(($stem:literal, $ty:ident)),* $(,)?) => {
            match stem {
                $($stem => Some(consumer.on_model::<config_models::$ty>()),)*
                _ => None,
            }
        };
    }
    for_each_config_model!(model_arms)
}

/// The rope bridge, type → const: each head dim a config states, as a typenum, names its own
/// `rope_at::<HD>` instantiation. No value is matched: the impl is chosen by the type.
#[cfg(feature = "spyre")]
pub mod rope_bridge {
    use ktir_superdsc::collective::Tp;
    use ktir_superdsc::emit::EmittedOp;
    use ktir_superdsc::emit::lower_ktir_to_superdsc::{Error, RopeAt, rope_at};
    use scratchy_tensors::typed::typenum::{self, Unsigned};

    /// One rope class's lowering, in the tensor-parallel world `T`.
    pub type RopeFn<T> = for<'a> fn(RopeAt<'a>) -> Result<Vec<EmittedOp<T>>, Error>;

    /// A head dim, as a type, with its rope instantiation.
    pub trait RopeHd: Unsigned {
        /// `rope_at::<T, Self::U32>`.
        fn rope<T: Tp>(a: RopeAt<'_>) -> Result<Vec<EmittedOp<T>>, Error>;
    }

    macro_rules! rope_impls {
        ($(($hd:literal, $u:ident)),* $(,)?) => {
            $(impl RopeHd for typenum::$u {
                fn rope<T: Tp>(a: RopeAt<'_>) -> Result<Vec<EmittedOp<T>>, Error> {
                    rope_at::<T, $hd>(a)
                }
            })*
        };
    }
    for_each_config_head_dim_type!(rope_impls);

    /// The rope adapter the bake hands down: each class's head dim and lowering, read off the
    /// model's marker type, so only `rope_at` is instantiated per head dim (and world).
    pub struct ModelRopes<T: Tp> {
        /// The base class: its head dim and its lowering.
        pub base: (u32, RopeFn<T>),
        /// The global class: its head dim and its lowering.
        pub global: (u32, RopeFn<T>),
    }

    // By hand: a derive would ask `T: Clone`, and the fields are fn pointers whatever `T` is.
    impl<T: Tp> Clone for ModelRopes<T> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<T: Tp> Copy for ModelRopes<T> {}

    impl<T: Tp> ModelRopes<T> {
        /// The adapter of the model `M`.
        pub const fn of<M: super::ConfigRopes>() -> Self
        where
            M::Base: RopeHd,
            M::Global: RopeHd,
        {
            Self {
                base: (M::Base::U32, <M::Base as RopeHd>::rope::<T>),
                global: (M::Global::U32, <M::Global as RopeHd>::rope::<T>),
            }
        }
    }
}
