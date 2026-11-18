// SPDX-License-Identifier: Apache-2.0
//! A decoder state as the compiler takes it: one handle ([`DecoderState`]) holding everything the
//! decoder state contributes from its own crate, and the typing ([`Typing`]) the build that compiles the
//! models hands the forward compiler, which lists those handles.

use scratchy_forward_compiler_frontend::state::StateVocabulary;
use scratchy_typed_shapes::Shapes;

/// One decoder state, as the build lists it: its DSL vocabulary.
#[derive(Clone, Copy)]
pub struct DecoderState {
    vocabulary: &'static dyn StateVocabulary,
}

impl DecoderState {
    pub const fn new(vocabulary: &'static dyn StateVocabulary) -> Self {
        Self { vocabulary }
    }

    /// The ops and state inputs the decoder state adds to the core vocabulary.
    pub const fn vocabulary(self) -> &'static dyn StateVocabulary {
        self.vocabulary
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
}

/// The vocabulary of each decoder state in `decoder_states`, in list order: [`Typing::vocabularies`], for the
/// build script that classifies the carriers before any typed program exists.
pub fn vocabularies(decoder_states: &[DecoderState]) -> impl Iterator<Item = &dyn StateVocabulary> {
    decoder_states
        .iter()
        .map(|state| state.vocabulary() as &dyn StateVocabulary)
}
