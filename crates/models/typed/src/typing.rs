// SPDX-License-Identifier: Apache-2.0
//! The typing every build hands the forward compiler, built in this one place.

use scratchy_decoder_state::Typing;

/// How the carriers are typed: their typed programs ([`crate::traced`]) under the decoder states the
/// build compiles ([`crate::decoder_states`]), the list the build script classified them with.
pub fn typing() -> Typing<'static> {
    Typing {
        traced: crate::traced,
        decoder_states: crate::decoder_states(),
    }
}
