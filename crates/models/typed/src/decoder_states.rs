// SPDX-License-Identifier: Apache-2.0
//! The decoder states every build compiles. The one list: the crate compiles this file as
//! `decoder_states`, and the build script compiles the same file to classify the carriers it types.

use scratchy_forward_compiler_frontend::state::StateVocabulary;

/// The decoder states every build compiles: the ops and state inputs a carrier may name after
/// the core vocabulary's. The forward compiler classifies each carrier with this list, as the
/// build script typed it.
pub fn decoder_states() -> &'static [&'static dyn StateVocabulary] {
    &[&scratchy_state_decoder_mla::Mla]
}
