// SPDX-License-Identifier: Apache-2.0
//! A decoder state as the compiler takes it: one handle ([`DecoderState`]) holding everything the
//! decoder state contributes from its own crate, and the typing ([`Typing`]) the build that compiles the
//! models hands the forward compiler, which lists those handles. A decoder state crate states each op
//! once, through [`state_op!`], and lists its ops and inputs once, through [`decoder_state!`].

use scratchy_forward_compiler_frontend::classify::ClassifyError;
pub use scratchy_forward_compiler_frontend::dim::{DeriveStage, DerivedDim, DerivedRule};
use scratchy_forward_compiler_frontend::state::{DimTable, OpName, StateVocabulary};
use scratchy_subtile::state_op::StateOpFacts;
use scratchy_typed_shapes::Shapes;

/// One decoder state, as the build lists it: its DSL vocabulary, and the facts its ops carry
/// into the subtile IR.
#[derive(Clone, Copy)]
pub struct DecoderState {
    vocabulary: &'static dyn StateVocabulary,
    subtile: fn(OpName) -> Option<&'static StateOpFacts>,
}

impl DecoderState {
    pub const fn new(
        vocabulary: &'static dyn StateVocabulary,
        subtile: fn(OpName) -> Option<&'static StateOpFacts>,
    ) -> Self {
        Self {
            vocabulary,
            subtile,
        }
    }

    /// The ops and state inputs the decoder state adds to the core vocabulary.
    pub const fn vocabulary(self) -> &'static dyn StateVocabulary {
        self.vocabulary
    }

    /// The subtile facts of the decoder state's op named by its argument, `None` for a name the decoder state
    /// states no facts for.
    pub const fn subtile(self) -> fn(OpName) -> Option<&'static StateOpFacts> {
        self.subtile
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

/// The items [`state_op!`] and [`decoder_state!`] expand to, named from the decoder state crate without its
/// depending on the front end or the subtile IR.
#[doc(hidden)]
pub mod __private {
    pub use scratchy_forward_compiler_frontend::dim::DerivedDim;
    pub use scratchy_forward_compiler_frontend::state::{
        Arity, InputName, OpName, SignaturePath, StateInputDecl, StateName, StateOpDecl, StateOpId,
        StateVocabulary,
    };
    pub use scratchy_subtile::state_op::{StateOpFacts, StateOutputs, StateRows};
    pub use scratchy_subtile::tensor_parallel::TpRule;
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

/// One state op, stated once: its typed signature, which a carrier's call is type-checked
/// against, then its row law and its sharding rule. Beside the `fn` it expands to a module of the
/// same name (the two live in different namespaces) holding:
///
/// - `DECL`, the op's DSL declaration: its name, its arity counted from the parameters and its
///   output count counted from the return type (a 2- or 3-tuple returns two or three values,
///   anything else one);
/// - `facts(state)`, the op's subtile facts under the decoder state [`decoder_state!`] names, which reads the
///   same counts, so the declaration and the facts cannot disagree.
///
/// For now an op's subtile operands are its DSL arguments, one source each and in order, and its
/// subtile outputs are its DSL outputs, so the one arity counted from the parameters serves both;
/// a decoder state that lowers differently will state its subtile operand count separately.
///
/// `rows` is a [`StateRows`](scratchy_subtile::state_op::StateRows) variant and `tp` a
/// [`TpRule`](scratchy_subtile::tensor_parallel::TpRule) variant.
///
/// ```
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         /// Two operands in, two values out.
///         pub fn probe_split(a: u32, b: u32) -> (u32, u32) {
///             (a, b)
///         }
///         rows = In0, tp = Whole;
///     }
/// }
/// assert_eq!(ops::probe_split::DECL.arity.get(), 2);
/// assert_eq!(ops::probe_split::DECL.outputs.get().get(), 2);
/// ```
///
/// An op without its row law does not build:
///
/// ```compile_fail
/// scratchy_decoder_state::state_op! {
///     pub fn probe(a: u32) -> u32 {
///         a
///     }
///     tp = Whole;
/// }
/// ```
///
/// Nor does an op returning more values than a state op may declare:
///
/// ```compile_fail
/// scratchy_decoder_state::state_op! {
///     pub fn probe(a: u32) -> (u32, u32, u32, u32) {
///         (a, a, a, a)
///     }
///     rows = In0, tp = Whole;
/// }
/// ```
///
/// Nor one returning no value:
///
/// ```compile_fail
/// scratchy_decoder_state::state_op! {
///     pub fn probe(_a: u32) -> () {}
///     rows = In0, tp = Whole;
/// }
/// ```
///
/// Nor an op with no operands whose rows are operand 0's:
///
/// ```compile_fail,E0080
/// scratchy_decoder_state::state_op! {
///     pub fn probe() -> u32 {
///         0
///     }
///     rows = In0, tp = Whole;
/// }
/// fn main() {}
/// ```
///
/// Nor one stating the rule only the sharding pass inserts:
///
/// ```compile_fail,E0080
/// scratchy_decoder_state::state_op! {
///     pub fn probe(a: u32) -> u32 {
///         a
///     }
///     rows = In0, tp = Collective;
/// }
/// fn main() {}
/// ```
#[macro_export]
macro_rules! state_op {
    (@op $outputs:ident [$(#[$meta:meta])*] $vis:vis $name:ident [$($g:ident),*]
        [$($p:ident : $pt:ty),*] [$ret:ty] $body:block $rows:ident $tp:ident) => {
        $(#[$meta])*
        $vis fn $name<$($g),*>($($p: $pt),*) -> $ret $body

        #[doc = ::core::concat!(
            "The declaration and subtile facts of the state op `",
            ::core::stringify!($name),
            "`, counted from its signature."
        )]
        $vis mod $name {
            const ARITY: $crate::__private::Arity =
                $crate::__private::Arity::new($crate::__state_count!($($p)*));
            const OUTPUTS: $crate::__private::StateOutputs =
                $crate::__private::StateOutputs::$outputs;
            const ROWS: $crate::__private::StateRows = $crate::__private::StateRows::$rows;
            const TP: $crate::__private::TpRule = $crate::__private::TpRule::$tp;

            const _: () = ::core::assert!(
                !(ARITY.get() == 0
                    && ::core::matches!(ROWS, $crate::__private::StateRows::In0)),
                ::core::concat!(
                    "state op `",
                    ::core::stringify!($name),
                    "` takes no operand, so its rows cannot be operand 0's"
                )
            );
            const _: () = ::core::assert!(
                !::core::matches!(TP, $crate::__private::TpRule::Collective),
                ::core::concat!(
                    "state op `",
                    ::core::stringify!($name),
                    "` states `Collective`, which only the sharding pass inserts"
                )
            );

            /// The op as the decoder state's DSL vocabulary declares it.
            pub const DECL: $crate::__private::StateOpDecl = $crate::__private::StateOpDecl {
                name: $crate::__private::OpName::new(::core::stringify!($name)),
                arity: ARITY,
                outputs: OUTPUTS.count(),
            };

            /// The op's subtile facts, as the decoder state named `state` declares it.
            pub const fn facts(
                state: $crate::__private::StateName,
            ) -> $crate::__private::StateOpFacts {
                $crate::__private::StateOpFacts::new(
                    $crate::__private::StateOpId::new(state, DECL.name),
                    ARITY,
                    OUTPUTS,
                    ROWS,
                    TP,
                    ::core::option::Option::None,
                )
            }
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($g:ident),* $(,)?>)? ($($p:ident : $pt:ty),* $(,)?)
            -> () $body:block
        rows = $rows:ident, tp = $tp:ident $(,)?;
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
        rows = $rows:ident, tp = $tp:ident $(,)?;
    ) => {
        $crate::state_op!(@op Two [$(#[$meta])*] $vis $name [$($($g),*)?] [$($p: $pt),*]
            [($r0, $r1)] $body $rows $tp);
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($g:ident),* $(,)?>)? ($($p:ident : $pt:ty),* $(,)?)
            -> ($r0:ty, $r1:ty, $r2:ty $(,)?) $body:block
        rows = $rows:ident, tp = $tp:ident $(,)?;
    ) => {
        $crate::state_op!(@op Three [$(#[$meta])*] $vis $name [$($($g),*)?] [$($p: $pt),*]
            [($r0, $r1, $r2)] $body $rows $tp);
    };
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident $(<$($g:ident),* $(,)?>)? ($($p:ident : $pt:ty),* $(,)?)
            -> ($r0:ty, $r1:ty, $r2:ty, $($more:ty),+ $(,)?) $body:block
        rows = $rows:ident, tp = $tp:ident $(,)?;
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
        rows = $rows:ident, tp = $tp:ident $(,)?;
    ) => {
        $crate::state_op!(@op One [$(#[$meta])*] $vis $name [$($($g),*)?] [$($p: $pt),*]
            [$ret] $body $rows $tp);
    };
}

/// One decoder state, listed once: its name, the absolute path of the module holding its ops'
/// signatures and its inputs (`::<crate>::<module>`, as the typed carriers call it from another
/// crate), each op [`state_op!`] declared there and each input, whose name must resolve in that
/// module (`typed_emit` calls it as a nullary fn, so a non-fn fails in the generated carrier).
/// Then, optionally, `dims = [...]`: the path of each `const` [`DerivedDim`] the decoder state adds after
/// the core ones, in the order they derive (omitted, it adds none).
/// Expands to a `static` [`DecoderState`] whose vocabulary declares those ops, inputs and dims and
/// whose `subtile` maps each op's name to its facts. A decoder state crate reaches its own signatures
/// through the absolute path by naming itself: `extern crate self as <crate>;`.
///
/// ```
/// extern crate self as probe;
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         /// One operand in, one value out.
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///         rows = In0, tp = Whole;
///     }
///
///     /// The probe's layered state input.
///     pub fn probe_state() -> u32 {
///         0
///     }
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     /// The probe decoder state.
///     pub static PROBE: name = probe, signatures = ::probe::ops,
///         ops = [probe_scale], inputs = [probe_state];
/// }
///
/// fn main() {
///     assert_eq!(PROBE.vocabulary().ops().len(), 1);
///     assert_eq!(PROBE.vocabulary().inputs().len(), 1);
///     assert!(PROBE.vocabulary().derived_dims().is_empty());
///     assert!(PROBE.subtile()(ops::probe_scale::DECL.name).is_some());
/// }
/// ```
///
/// A decoder state adding derived dims lists each, after its inputs:
///
/// ```
/// extern crate self as probe;
///
/// use scratchy_decoder_state::{DeriveStage, DerivedDim, DerivedRule};
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///         rows = In0, tp = Whole;
///     }
///
///     pub fn probe_state() -> u32 {
///         0
///     }
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
///     pub static PROBE: name = probe, signatures = ::probe::ops,
///         ops = [probe_scale], inputs = [probe_state], dims = [PROBE_WIDTH, crate::PROBE_PAIR];
/// }
///
/// fn main() {
///     let names: Vec<_> = PROBE.vocabulary().derived_dims().iter().map(|d| d.name).collect();
///     assert_eq!(names, ["probe_width", "probe_pair"]);
/// }
/// ```
///
/// A dim that is not a [`DerivedDim`] does not build:
///
/// ```compile_fail,E0308
/// extern crate self as probe;
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///         rows = In0, tp = Whole;
///     }
/// }
///
/// pub const NOT_A_DIM: u32 = 0;
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = probe, signatures = ::probe::ops,
///         ops = [probe_scale], inputs = [], dims = [NOT_A_DIM];
/// }
/// fn main() {}
/// ```
///
/// A decoder state listing an op no [`state_op!`] declared does not build:
///
/// ```compile_fail,E0433
/// extern crate self as probe;
///
/// pub mod ops {}
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = probe, signatures = ::probe::ops, ops = [nope], inputs = [];
/// }
/// fn main() {}
/// ```
///
/// Nor one listing a plain `fn` as an op:
///
/// ```compile_fail,E0433
/// extern crate self as probe;
///
/// pub mod ops {
///     pub fn plain(a: u32) -> u32 {
///         a
///     }
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = probe, signatures = ::probe::ops, ops = [plain], inputs = [];
/// }
/// fn main() {}
/// ```
///
/// Nor one listing an input whose name does not resolve in the signatures module:
///
/// ```compile_fail,E0432
/// extern crate self as probe;
///
/// pub mod ops {
///     scratchy_decoder_state::state_op! {
///         pub fn probe_scale(a: u32) -> u32 {
///             a
///         }
///         rows = In0, tp = Whole;
///     }
/// }
///
/// scratchy_decoder_state::decoder_state! {
///     pub static PROBE: name = probe, signatures = ::probe::ops,
///         ops = [probe_scale], inputs = [nope];
/// }
/// fn main() {}
/// ```
#[macro_export]
macro_rules! decoder_state {
    (
        $(#[$meta:meta])*
        $vis:vis static $state_static:ident: name = $state:ident,
            signatures = $(:: $seg:ident)+,
            ops = [$($op:ident),+ $(,)?],
            inputs = [$($input:ident),* $(,)?]
            $(, dims = [$($dim:path),* $(,)?])? $(,)?;
    ) => {
        $(#[$meta])*
        $vis static $state_static: $crate::DecoderState = {
            use $(:: $seg)+ as signatures;

            const NAME: $crate::__private::StateName =
                $crate::__private::StateName::new(::core::stringify!($state));

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

            fn subtile(
                op: $crate::__private::OpName,
            ) -> ::core::option::Option<&'static $crate::__private::StateOpFacts> {
                $(
                    if op == signatures::$op::DECL.name {
                        static FACTS: $crate::__private::StateOpFacts =
                            signatures::$op::facts(NAME);
                        return ::core::option::Option::Some(&FACTS);
                    }
                )+
                ::core::option::Option::None
            }

            $crate::DecoderState::new(&Vocabulary, subtile)
        };
    };
}
