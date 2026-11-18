// SPDX-License-Identifier: Apache-2.0
//! The decoder states every build compiles. The one list: the crate compiles this file as
//! `decoder_states`, and the build script compiles the same file to classify the carriers it types.

use scratchy_decoder_state::DecoderState;

/// The decoder states every build compiles: the ops and state inputs a carrier may name after
/// the core vocabulary's. The forward compiler classifies each carrier with this list, as the
/// build script typed it.
pub fn decoder_states() -> &'static [DecoderState] {
    static DECODER_STATES: [DecoderState; 1] = [scratchy_state_decoder_mla::MLA];
    &DECODER_STATES
}
