// SPDX-License-Identifier: Apache-2.0
//! Multi-head latent attention (MLA): the keys and values are decompressed per head from one
//! compressed latent, beside one rope key every head shares. This crate is the decoder state ([`MLA`])
//! and the typed signatures of its ops and state input ([`ops`]), each op stated once there;
//! nothing shared names it, so deleting the crate and its one list entry deletes the decoder state.

// The signatures path is absolute, as the typed carriers call it from another crate; the decoder state
// reaches its own ops through the same path.
extern crate self as scratchy_state_decoder_mla;

pub mod ops;

scratchy_decoder_state::decoder_state! {
    /// The multi-head latent attention decoder state: its three ops, and one layer's latent cache as its
    /// state input.
    pub static MLA: name = mla, signatures = ::scratchy_state_decoder_mla::ops,
        ops = [mla_split, mla_append, mla_attention], inputs = [mla_cache];
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU8;

    use scratchy_forward_compiler_frontend::state::{
        Arity, InputName, OpName, OutputCount, SignaturePath, StateName,
    };
    use scratchy_subtile::state_op::StateRows;
    use scratchy_subtile::tensor_parallel::TpRule;

    use super::MLA;

    /// Each op's arity and output count, counted from its signature, are the ones a carrier's
    /// call binds, and its facts read the same counts.
    #[test]
    fn mla_states_each_op_once() {
        let count = |n: u8| OutputCount::new(NonZeroU8::MIN.saturating_add(n - 1));
        let vocabulary = MLA.vocabulary();
        assert_eq!(vocabulary.name(), StateName::new("mla"));
        assert_eq!(
            vocabulary.signatures(),
            SignaturePath::new("::scratchy_state_decoder_mla::ops")
        );
        let expected = [
            ("mla_split", 1, 2, StateRows::In0),
            ("mla_append", 6, 3, StateRows::In0),
            ("mla_attention", 6, 1, StateRows::M),
        ];
        assert_eq!(vocabulary.ops().len(), expected.len());
        for (decl, (name, arity, outputs, rows)) in vocabulary.ops().iter().zip(expected) {
            assert_eq!(decl.name, OpName::new(name));
            assert_eq!(decl.arity, Arity::new(arity));
            assert_eq!(decl.outputs, count(outputs));
            let facts = (MLA.subtile())(decl.name).expect("every MLA op states its facts");
            assert_eq!(facts.id().state(), StateName::new("mla"));
            assert_eq!(facts.id().name(), decl.name);
            assert_eq!(facts.arity(), decl.arity);
            assert_eq!(facts.outputs(), decl.outputs);
            assert_eq!(facts.rows(), rows);
            assert_eq!(facts.tp(), TpRule::Whole);
        }
        let [input] = vocabulary.inputs() else {
            panic!("MLA declares one state input");
        };
        assert_eq!(input.name, InputName::new("mla_cache"));
        assert!((MLA.subtile())(OpName::new("mla_cache")).is_none());
    }
}
