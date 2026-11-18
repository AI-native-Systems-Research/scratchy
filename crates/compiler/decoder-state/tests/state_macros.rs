// SPDX-License-Identifier: Apache-2.0
//! A decoder state stated through `state_op!` and `decoder_state!`: the build counts each op's arity and
//! outputs from its signature, and the vocabulary and the subtile facts read the same count.

// The decoder state's signatures path is absolute, as a carrier in another crate calls it.
extern crate self as state_macros;

use std::num::NonZeroU8;

use scratchy_forward_compiler_frontend::state::{
    Arity, InputName, OpName, OutputCount, SignaturePath, StateName,
};
use scratchy_subtile::state_op::StateRows;
use scratchy_subtile::tensor_parallel::TpRule;

pub mod ops {
    scratchy_decoder_state::state_op! {
        /// Three operands in, two values out.
        pub fn probe_mix(a: u32, b: u32, c: u32) -> (u32, u32) {
            (a, b + c)
        }
        rows = M, tp = Whole;
    }

    /// The probe's layered state input.
    pub fn probe_state() -> u32 {
        0
    }
}

scratchy_decoder_state::decoder_state! {
    /// The test decoder state.
    pub static PROBE: name = probe, signatures = ::state_macros::ops,
        ops = [probe_mix], inputs = [probe_state];
}

#[test]
fn state_counts_arity_and_outputs() {
    let two = OutputCount::new(NonZeroU8::MIN.saturating_add(1));
    let vocabulary = PROBE.vocabulary();
    assert_eq!(vocabulary.name(), StateName::new("probe"));
    assert_eq!(
        vocabulary.signatures(),
        SignaturePath::new("::state_macros::ops")
    );
    let [decl] = vocabulary.ops() else {
        panic!("the probe declares one op");
    };
    assert_eq!(decl.name, OpName::new("probe_mix"));
    assert_eq!(decl.arity, Arity::new(3));
    assert_eq!(decl.outputs, two);
    let [input] = vocabulary.inputs() else {
        panic!("the probe declares one input");
    };
    assert_eq!(input.name, InputName::new("probe_state"));

    let facts = (PROBE.subtile())(OpName::new("probe_mix")).expect("probe_mix states facts");
    assert_eq!(facts.id().state(), StateName::new("probe"));
    assert_eq!(facts.id().name(), OpName::new("probe_mix"));
    assert_eq!(facts.arity(), Arity::new(3));
    assert_eq!(facts.outputs(), two);
    assert_eq!(facts.rows(), StateRows::M);
    assert_eq!(facts.tp(), TpRule::Whole);
    assert!((PROBE.subtile())(OpName::new("probe_state")).is_none());

    assert_eq!(ops::probe_mix(1, 2, 3), (1, 5));
    assert_eq!(ops::probe_state(), 0);
}
