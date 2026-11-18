// SPDX-License-Identifier: Apache-2.0
//! The typing every build hands the forward compiler lists each decoder state the build compiles once,
//! and its vocabulary list is that list, decoder state for decoder state.

use scratchy_forward_compiler_frontend::state::StateName;
use scratchy_models_typed::typing;

#[test]
fn typing_lists_every_state_once() {
    let typing = typing();
    let names: Vec<StateName> = typing
        .decoder_states
        .iter()
        .map(|r| r.vocabulary().name())
        .collect();
    assert_eq!(names, [StateName::new("mla")]);
    assert_eq!(typing.vocabularies().count(), typing.decoder_states.len());
}
