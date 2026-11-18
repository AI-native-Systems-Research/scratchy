// SPDX-License-Identifier: Apache-2.0
//! A decoder state as the compiler takes it: one handle ([`DecoderState`]) holding everything the
//! decoder state contributes from its own crate, and the typing ([`Typing`]) the build that compiles the
//! models hands the forward compiler, which lists those handles. A decoder state crate states each
//! DSL op once, through [`state_op!`], each subtile op its lowerings push once, through
//! [`subtile_op!`], and lists its ops, their lowerings ([`lower`]), its inputs and its subtile ops
//! once, through [`decoder_state!`].

use scratchy_forward_compiler_frontend::classify::ClassifyError;
pub use scratchy_forward_compiler_frontend::dim::{DeriveStage, DerivedDim, DerivedRule};
pub use scratchy_forward_compiler_frontend::state::StateName;
use scratchy_forward_compiler_frontend::state::{DimTable, OpName, StateVocabulary};
use scratchy_subtile::state_op::StateOpFacts;
use scratchy_typed_shapes::Shapes;

pub mod lower;

use lower::LowerFn;

/// One decoder state, as the build lists it: its DSL vocabulary, the lowering of each of its DSL
/// ops, and the facts of each subtile op those lowerings push.
#[derive(Clone, Copy)]
pub struct DecoderState {
    vocabulary: &'static dyn StateVocabulary,
    lowering: fn(OpName) -> Option<LowerFn>,
    subtile: fn(OpName) -> Option<&'static StateOpFacts>,
}

impl DecoderState {
    pub const fn new(
        vocabulary: &'static dyn StateVocabulary,
        lowering: fn(OpName) -> Option<LowerFn>,
        subtile: fn(OpName) -> Option<&'static StateOpFacts>,
    ) -> Self {
        Self {
            vocabulary,
            lowering,
            subtile,
        }
    }

    /// The ops and state inputs the decoder state adds to the core vocabulary.
    pub const fn vocabulary(self) -> &'static dyn StateVocabulary {
        self.vocabulary
    }

    /// The lowering of the decoder state's DSL op `op`, `None` for a name it declares no DSL op by.
    pub fn lowering(self, op: OpName) -> Option<LowerFn> {
        (self.lowering)(op)
    }

    /// The facts of the decoder state's subtile op `op`, `None` for a name it declares no subtile
    /// op by.
    pub fn subtile(self, op: OpName) -> Option<&'static StateOpFacts> {
        (self.subtile)(op)
    }
}

/// How the carriers are typed: what the build that compiles the models hands the forward compiler.
/// `scratchy_models_typed::typing()` builds it for every build.
pub struct Typing<'a> {
    /// The shapes of the carrier whose arch is the argument: its typed program, run (`None` when
    /// no carrier of that arch is typed).
    pub traced: fn(&str) -> Option<Shapes>,
    /// The decoder states whose ops and inputs a carrier may name after the core ones: the
    /// same list the typing classified with.
    pub decoder_states: &'a [DecoderState],
}

impl<'a> Typing<'a> {
    /// The vocabulary of each decoder state, in list order: what `classify::carrier_program` classifies a
    /// carrier under.
    pub fn vocabularies(&self) -> impl Iterator<Item = &'a dyn StateVocabulary> {
        vocabularies(self.decoder_states)
    }

    /// The derived dims a config derives by under this typing: the core rows, then each decoder
    /// state's, assembled as a carrier classified under [`Typing::vocabularies`] holds them.
    pub fn dims(&self) -> Result<DimTable, ClassifyError> {
        DimTable::assemble(&self.vocabularies().collect::<Vec<_>>())
    }
}

/// The vocabulary of each decoder state in `decoder_states`, in list order: [`Typing::vocabularies`], for the
/// build script that classifies the carriers before any typed program exists.
pub fn vocabularies(decoder_states: &[DecoderState]) -> impl Iterator<Item = &dyn StateVocabulary> {
    decoder_states
        .iter()
        .map(|state| state.vocabulary() as &dyn StateVocabulary)
}

/// The items [`state_op!`], [`subtile_op!`] and [`decoder_state!`] expand to, named from the
/// decoder state crate without its depending on the front end or the subtile IR.
#[doc(hidden)]
pub mod __private {
    pub use scratchy_forward_compiler_frontend::dim::DerivedDim;
    pub use scratchy_forward_compiler_frontend::state::{
        Arity, InputName, OpName, SignaturePath, StateInputDecl, StateName, StateOpDecl, StateOpId,
        StateVocabulary,
    };
    pub use scratchy_subtile::state_op::{StateOpFacts, StateOutputs, StateRows};
    pub use scratchy_subtile::tensor_parallel::TpRule;

    pub use crate::lower::LowerFn;
}

/// The number of token trees it is given, as a `u8`: an op's arity, counted from its parameter
/// names. The recursion limit (128 by default) stops the expansion well before a count could
/// pass `u8::MAX`, so an op with that many parameters fails to compile.
#[doc(hidden)]
#[macro_export]
macro_rules! __state_count {
    () => {
        0u8
    };
    ($one:tt) => {
        1u8
    };
    ($head:tt $($tail:tt)+) => {
        1u8 + $crate::__state_count!($($tail)+)
    };
}

/// One DSL op, stated once: its typed signature, which a carrier's call is type-checked against.
/// Beside the `fn` it expands to a module of the same name (the two live in different
/// namespaces) holding `DECL`, the op's DSL declaration: its name, its arity counted from the
/// parameters and its output count counted from the return type (a 2- or 3-tuple returns two or
/// three values, anything else one).
///
/// ```
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         /// Two operands in, two values out.
///         pub fn probe_split(a: u32, b: u32) -> (u32, u32) {
///             (a, b)
///         }
///     }
/// }
/// assert_eq!(ops::probe_split::DECL.arity.get(), 2);
/// assert_eq!(ops::probe_split::DECL.outputs.get().get(), 2);
/// ```
///
/// An op returning more values than a state op may declare does not build:
///
/// ```compile_fail
/// scratchy_decoder_state::state_op! {
///     pub fn probe(a: u32) -> (u32, u32, u32, u32) {
///         (a, a, a, a)
///     }
/// }
/// ```
///
/// Nor one returning no value:
///
/// ```compile_fail
/// scratchy_decoder_state::state_op! {
///     pub fn probe(_a: u32) -> () {}
/// }
/// ```
#[macro_export]
macro_rules! state_op {
    (@op $outputs:ident [$(#[$meta:meta])*] $vis:vis $name:ident [$($g:ident),*]
        [$($p:ident : $pt:ty),*] [$ret:ty] $body:block) => {
        $(#[$meta])*
        $vis fn $name<$($g),*>($($p: $pt),*) -> $ret $body

        #[doc = ::core::concat!(
            "The DSL declaration of the state op `",
            ::core::stringify!($name),
            "`, counted from its signature."
        )]
        $vis mod $name {
            /// The op as the decoder state's DSL vocabulary declares it.
            pub const DECL: $crate::__private::StateOpDecl = $crate::__private::StateOpDecl {
                name: $crate::__private::OpName::new(::core::stringify!($name)),
                arity: $crate::__private::Arity::new($crate::__state_count!($($p)*)),
                outputs: $crate::__private::StateOutputs::$outputs.count(),
            };
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($g:ident),* $(,)?>)? ($($p:ident : $pt:ty),* $(,)?)
            -> () $body:block
    ) => {
        ::core::compile_error!(::core::concat!(
            "state op `",
            ::core::stringify!($name),
            "` returns no value: a state op returns one, two or three"
        ));
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($g:ident),* $(,)?>)? ($($p:ident : $pt:ty),* $(,)?)
            -> ($r0:ty, $r1:ty $(,)?) $body:block
    ) => {
        $crate::state_op!(@op Two [$(#[$meta])*] $vis $name [$($($g),*)?] [$($p: $pt),*]
            [($r0, $r1)] $body);
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($g:ident),* $(,)?>)? ($($p:ident : $pt:ty),* $(,)?)
            -> ($r0:ty, $r1:ty, $r2:ty $(,)?) $body:block
    ) => {
        $crate::state_op!(@op Three [$(#[$meta])*] $vis $name [$($($g),*)?] [$($p: $pt),*]
            [($r0, $r1, $r2)] $body);
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($g:ident),* $(,)?>)? ($($p:ident : $pt:ty),* $(,)?)
            -> ($r0:ty, $r1:ty, $r2:ty, $($more:ty),+ $(,)?) $body:block
    ) => {
        ::core::compile_error!(::core::concat!(
            "state op `",
            ::core::stringify!($name),
            "` returns more than three values: a state op returns one, two or three"
        ));
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($g:ident),* $(,)?>)? ($($p:ident : $pt:ty),* $(,)?)
            -> $ret:ty $body:block
    ) => {
        $crate::state_op!(@op One [$(#[$meta])*] $vis $name [$($($g),*)?] [$($p: $pt),*]
            [$ret] $body);
    };
}

/// One or more subtile ops, each stated once as a `static`
/// [`StateOpFacts`](scratchy_subtile::state_op::StateOpFacts): the op's name, the decoder state
/// it belongs to (the crate's `const` [`StateName`], which [`decoder_state!`] checks against its
/// own name), its operand count, its output count, its row law, its sharding rule and the operand
/// that is its weight (`none`, or the operand's index). A decoder state's lowerings push these
/// ops; the DSL op a carrier calls is [`state_op!`]'s, and one DSL op may lower to several.
///
/// `outputs` is a [`StateOutputs`](scratchy_subtile::state_op::StateOutputs) variant, `rows` a
/// [`StateRows`](scratchy_subtile::state_op::StateRows) variant and `tp` a
/// [`TpRule`](scratchy_subtile::tensor_parallel::TpRule) variant.
///
/// ```
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// scratchy_decoder_state::subtile_op! {
///     /// One operand in, one value out.
///     pub static PROBE_COPY: name = probe_copy, state = STATE, arity = 1, outputs = One,
///         rows = In0, tp = Whole, weight = none;
///     /// An activation and a weight in, one row per token out.
///     pub static PROBE_MIX: name = probe_mix, state = STATE, arity = 2, outputs = One,
///         rows = M, tp = Whole, weight = 1;
/// }
///
/// fn main() {
///     assert_eq!(PROBE_COPY.id().name().get(), "probe_copy");
///     assert_eq!(PROBE_COPY.weight(), None);
///     assert_eq!(PROBE_MIX.arity().get(), 2);
///     assert_eq!(PROBE_MIX.weight(), Some(1));
/// }
/// ```
///
/// An op with no operands whose rows are operand 0's does not build:
///
/// ```compile_fail,E0080
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE: name = probe, state = STATE, arity = 0, outputs = One,
///         rows = In0, tp = Whole, weight = none;
/// }
/// fn main() {}
/// ```
///
/// Nor one stating the rule only the sharding pass inserts:
///
/// ```compile_fail,E0080
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE: name = probe, state = STATE, arity = 1, outputs = One,
///         rows = In0, tp = Collective, weight = none;
/// }
/// fn main() {}
/// ```
///
/// Nor one whose weight is past its operands:
///
/// ```compile_fail,E0080
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE: name = probe, state = STATE, arity = 2, outputs = One,
///         rows = M, tp = Whole, weight = 2;
/// }
/// fn main() {}
/// ```
#[macro_export]
macro_rules! subtile_op {
    (@weight none) => {
        ::core::option::Option::None
    };
    (@weight $operand:literal) => {
        ::core::option::Option::Some($operand)
    };
    ($(
        $(#[$meta:meta])*
        $vis:vis static $facts:ident: name = $name:ident, state = $state:path,
            arity = $arity:literal, outputs = $outputs:ident, rows = $rows:ident,
            tp = $tp:ident, weight = $weight:tt $(,)?;
    )+) => {$(
        $(#[$meta])*
        $vis static $facts: $crate::__private::StateOpFacts = {
            const ARITY: $crate::__private::Arity = $crate::__private::Arity::new($arity);
            const ROWS: $crate::__private::StateRows = $crate::__private::StateRows::$rows;
            const TP: $crate::__private::TpRule = $crate::__private::TpRule::$tp;

            const _: () = ::core::assert!(
                !(ARITY.get() == 0
                    && ::core::matches!(ROWS, $crate::__private::StateRows::In0)),
                ::core::concat!(
                    "subtile op `",
                    ::core::stringify!($name),
                    "` takes no operand, so its rows cannot be operand 0's"
                )
            );
            const _: () = ::core::assert!(
                !::core::matches!(TP, $crate::__private::TpRule::Collective),
                ::core::concat!(
                    "subtile op `",
                    ::core::stringify!($name),
                    "` states `Collective`, which only the sharding pass inserts"
                )
            );

            $crate::__private::StateOpFacts::new(
                $crate::__private::StateOpId::new(
                    $state,
                    $crate::__private::OpName::new(::core::stringify!($name)),
                ),
                ARITY,
                $crate::__private::StateOutputs::$outputs,
                ROWS,
                TP,
                $crate::subtile_op!(@weight $weight),
            )
        };
    )+};
}

/// One decoder state, listed once:
///
/// - `name`: the crate's `const` [`StateName`];
/// - `signatures`: the absolute path of the module holding its DSL ops' signatures and its
///   inputs (`::<crate>::<module>`, as the typed carriers call it from another crate);
/// - `ops`: each DSL op [`state_op!`] declared there, paired with the path of its
///   [`LowerFn`](lower::LowerFn), so a DSL op listed without a lowering does not parse;
/// - `inputs`: each input, whose name must resolve in that module (`typed_emit` calls it as a
///   nullary fn, so a non-fn fails in the generated carrier);
/// - `subtile`: each subtile op [`subtile_op!`] declared, which the lowerings push; a listed row
///   stated under another decoder state's name does not build;
/// - then, optionally, `dims = [...]`: the path of each `const` [`DerivedDim`] the decoder state
///   adds after the core ones, in the order they derive (omitted, it adds none).
///
/// Expands to a `static` [`DecoderState`] whose vocabulary declares those ops, inputs and dims,
/// whose `lowering` maps each DSL op's name to its lowering, and whose `subtile` maps each
/// subtile op's name to its facts. A decoder state crate reaches its own signatures through the
/// absolute path by naming itself: `extern crate self as <crate>;`.
///
/// ```
/// extern crate self as probe;
///
/// use scratchy_decoder_state::StateName;
/// use scratchy_forward_compiler_frontend::state::OpName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         /// One operand in, one value out.
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///     }
///
///     /// The probe's layered state input.
///     pub fn probe_state() -> u32 {
///         0
///     }
/// }
///
/// scratchy_decoder_state::subtile_op! {
///     /// The one subtile op `probe_scale` lowers to.
///     pub static PROBE_SCALE_TILE: name = probe_scale_tile, state = STATE, arity = 1,
///         outputs = One, rows = In0, tp = Whole, weight = none;
/// }
///
/// pub mod lower {
///     use scratchy_decoder_state::lower::{
///         ArgIndex, LowerError, OutIndex, StateLowering, StateSubOp,
///     };
///
///     /// `probe_scale(a)`: one subtile op over `a`, as wide as the output.
///     pub fn scale(l: &mut dyn StateLowering) -> Result<(), LowerError> {
///         let a = l.input(ArgIndex::new(0))?;
///         let cols = l.out_cols(OutIndex::new(0))?;
///         let scaled = l.push(StateSubOp::new(&crate::PROBE_SCALE_TILE, |_| cols), vec![a]);
///         l.produce(OutIndex::new(0), scaled);
///         Ok(())
///     }
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     /// The probe decoder state.
///     pub static PROBE: name = STATE, signatures = ::probe::ops,
///         ops = [probe_scale => lower::scale], inputs = [probe_state],
///         subtile = [PROBE_SCALE_TILE];
/// }
///
/// fn main() {
///     assert_eq!(PROBE.vocabulary().name(), STATE);
///     assert_eq!(PROBE.vocabulary().ops().len(), 1);
///     assert_eq!(PROBE.vocabulary().inputs().len(), 1);
///     assert!(PROBE.vocabulary().derived_dims().is_empty());
///     assert!(PROBE.lowering(ops::probe_scale::DECL.name).is_some());
///     assert!(PROBE.lowering(OpName::new("probe_scale_tile")).is_none());
///     assert!(PROBE.subtile(OpName::new("probe_scale_tile")).is_some());
///     assert!(PROBE.subtile(ops::probe_scale::DECL.name).is_none());
/// }
/// ```
///
/// A decoder state adding derived dims lists each, after its subtile ops:
///
/// ```
/// extern crate self as probe;
///
/// use scratchy_decoder_state::{DeriveStage, DerivedDim, DerivedRule, StateName};
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///     }
///
///     pub fn probe_state() -> u32 {
///         0
///     }
/// }
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE_SCALE_TILE: name = probe_scale_tile, state = STATE, arity = 1,
///         outputs = One, rows = In0, tp = Whole, weight = none;
/// }
///
/// fn unlowered(
///     _: &mut dyn scratchy_decoder_state::lower::StateLowering,
/// ) -> Result<(), scratchy_decoder_state::lower::LowerError> {
///     Ok(())
/// }
///
/// /// Each head's probe lanes.
/// pub const PROBE_WIDTH: DerivedDim = DerivedDim {
///     name: "probe_width",
///     stage: DeriveStage::Implicit,
///     rule: DerivedRule::Mul(&[
///         DerivedRule::Bound("num_attention_heads"),
///         DerivedRule::Bound("probe_lanes"),
///     ]),
/// };
///
/// /// The probe width, twice.
/// pub const PROBE_PAIR: DerivedDim = DerivedDim {
///     name: "probe_pair",
///     stage: DeriveStage::Implicit,
///     rule: DerivedRule::Mul(&[DerivedRule::Lit(2), DerivedRule::Bound("probe_width")]),
/// };
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = STATE, signatures = ::probe::ops,
///         ops = [probe_scale => unlowered], inputs = [probe_state],
///         subtile = [PROBE_SCALE_TILE], dims = [PROBE_WIDTH, crate::PROBE_PAIR];
/// }
///
/// fn main() {
///     let names: Vec<_> = PROBE.vocabulary().derived_dims().iter().map(|d| d.name).collect();
///     assert_eq!(names, ["probe_width", "probe_pair"]);
/// }
/// ```
///
/// A DSL op listed without its lowering does not parse:
///
/// ```compile_fail
/// extern crate self as probe;
///
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///     }
/// }
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE_SCALE_TILE: name = probe_scale_tile, state = STATE, arity = 1,
///         outputs = One, rows = In0, tp = Whole, weight = none;
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = STATE, signatures = ::probe::ops,
///         ops = [probe_scale], inputs = [], subtile = [PROBE_SCALE_TILE];
/// }
/// fn main() {}
/// ```
///
/// Nor does a decoder state build when a subtile row it lists is stated under another decoder
/// state's name:
///
/// ```compile_fail,E0080
/// extern crate self as probe;
///
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
/// pub const OTHER: StateName = StateName::new("other");
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///     }
/// }
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE_SCALE_TILE: name = probe_scale_tile, state = OTHER, arity = 1,
///         outputs = One, rows = In0, tp = Whole, weight = none;
/// }
///
/// fn unlowered(
///     _: &mut dyn scratchy_decoder_state::lower::StateLowering,
/// ) -> Result<(), scratchy_decoder_state::lower::LowerError> {
///     Ok(())
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = STATE, signatures = ::probe::ops,
///         ops = [probe_scale => unlowered], inputs = [], subtile = [PROBE_SCALE_TILE];
/// }
/// fn main() {}
/// ```
///
/// A dim that is not a [`DerivedDim`] does not build:
///
/// ```compile_fail,E0308
/// extern crate self as probe;
///
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///     }
/// }
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE_SCALE_TILE: name = probe_scale_tile, state = STATE, arity = 1,
///         outputs = One, rows = In0, tp = Whole, weight = none;
/// }
///
/// fn unlowered(
///     _: &mut dyn scratchy_decoder_state::lower::StateLowering,
/// ) -> Result<(), scratchy_decoder_state::lower::LowerError> {
///     Ok(())
/// }
///
/// pub const NOT_A_DIM: u32 = 0;
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = STATE, signatures = ::probe::ops,
///         ops = [probe_scale => unlowered], inputs = [], subtile = [PROBE_SCALE_TILE],
///         dims = [NOT_A_DIM];
/// }
/// fn main() {}
/// ```
///
/// Nor a decoder state listing an op no [`state_op!`] declared:
///
/// ```compile_fail,E0433
/// extern crate self as probe;
///
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// pub mod ops {}
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE_SCALE_TILE: name = probe_scale_tile, state = STATE, arity = 1,
///         outputs = One, rows = In0, tp = Whole, weight = none;
/// }
///
/// fn unlowered(
///     _: &mut dyn scratchy_decoder_state::lower::StateLowering,
/// ) -> Result<(), scratchy_decoder_state::lower::LowerError> {
///     Ok(())
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = STATE, signatures = ::probe::ops,
///         ops = [nope => unlowered], inputs = [], subtile = [PROBE_SCALE_TILE];
/// }
/// fn main() {}
/// ```
///
/// Nor one listing a plain `fn` as an op:
///
/// ```compile_fail,E0433
/// extern crate self as probe;
///
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// pub mod ops {
///     pub fn plain(a: u32) -> u32 {
///         a
///     }
/// }
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE_SCALE_TILE: name = probe_scale_tile, state = STATE, arity = 1,
///         outputs = One, rows = In0, tp = Whole, weight = none;
/// }
///
/// fn unlowered(
///     _: &mut dyn scratchy_decoder_state::lower::StateLowering,
/// ) -> Result<(), scratchy_decoder_state::lower::LowerError> {
///     Ok(())
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = STATE, signatures = ::probe::ops,
///         ops = [plain => unlowered], inputs = [], subtile = [PROBE_SCALE_TILE];
/// }
/// fn main() {}
/// ```
///
/// Nor one listing an input whose name does not resolve in the signatures module:
///
/// ```compile_fail,E0432
/// extern crate self as probe;
///
/// use scratchy_decoder_state::StateName;
///
/// pub const STATE: StateName = StateName::new("probe");
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///     }
/// }
///
/// scratchy_decoder_state::subtile_op! {
///     pub static PROBE_SCALE_TILE: name = probe_scale_tile, state = STATE, arity = 1,
///         outputs = One, rows = In0, tp = Whole, weight = none;
/// }
///
/// fn unlowered(
///     _: &mut dyn scratchy_decoder_state::lower::StateLowering,
/// ) -> Result<(), scratchy_decoder_state::lower::LowerError> {
///     Ok(())
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = STATE, signatures = ::probe::ops,
///         ops = [probe_scale => unlowered], inputs = [nope], subtile = [PROBE_SCALE_TILE];
/// }
/// fn main() {}
/// ```
#[macro_export]
macro_rules! decoder_state {
    (
        $(#[$meta:meta])*
        $vis:vis static $state_static:ident: name = $name:path,
            signatures = $(:: $seg:ident)+,
            ops = [$($op:ident => $lower:path),+ $(,)?],
            inputs = [$($input:ident),* $(,)?],
            subtile = [$($row:path),+ $(,)?]
            $(, dims = [$($dim:path),* $(,)?])? $(,)?;
    ) => {
        $(#[$meta])*
        $vis static $state_static: $crate::DecoderState = {
            use $(:: $seg)+ as signatures;

            const NAME: $crate::__private::StateName = $name;

            $(
                const _: () = ::core::assert!(
                    NAME.same_as($row.id().state()),
                    ::core::concat!(
                        "subtile op `",
                        ::core::stringify!($row),
                        "` is stated under another decoder state's name"
                    )
                );
            )+

            struct Vocabulary;

            impl $crate::__private::StateVocabulary for Vocabulary {
                fn name(&self) -> $crate::__private::StateName {
                    NAME
                }

                fn ops(&self) -> &'static [$crate::__private::StateOpDecl] {
                    const OPS: &[$crate::__private::StateOpDecl] =
                        &[$(signatures::$op::DECL),+];
                    OPS
                }

                fn inputs(&self) -> &'static [$crate::__private::StateInputDecl] {
                    // Each input names its constructor, which a carrier calls. Its type
                    // parameters are the carrier's to infer, so only that the name resolves is checked.
                    $(use signatures::$input as _;)*
                    const INPUTS: &[$crate::__private::StateInputDecl] = &[$(
                        $crate::__private::StateInputDecl {
                            name: $crate::__private::InputName::new(::core::stringify!($input)),
                        }
                    ),*];
                    INPUTS
                }

                fn signatures(&self) -> $crate::__private::SignaturePath {
                    $crate::__private::SignaturePath::new(::core::concat!(
                        $("::", ::core::stringify!($seg)),+
                    ))
                }

                fn derived_dims(&self) -> &'static [$crate::__private::DerivedDim] {
                    const DIMS: &[$crate::__private::DerivedDim] = &[$($($dim),*)?];
                    DIMS
                }
            }

            fn lowering(
                op: $crate::__private::OpName,
            ) -> ::core::option::Option<$crate::__private::LowerFn> {
                $(
                    if op == signatures::$op::DECL.name {
                        return ::core::option::Option::Some($lower);
                    }
                )+
                ::core::option::Option::None
            }

            fn subtile(
                op: $crate::__private::OpName,
            ) -> ::core::option::Option<&'static $crate::__private::StateOpFacts> {
                $(
                    if op == $row.id().name() {
                        return ::core::option::Option::Some(&$row);
                    }
                )+
                ::core::option::Option::None
            }

            $crate::DecoderState::new(&Vocabulary, lowering, subtile)
        };
    };
}
