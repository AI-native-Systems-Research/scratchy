// SPDX-License-Identifier: Apache-2.0
//! A decoder state stated through `state_op!`, `subtile_op!` and `decoder_state!`: the build
//! counts each DSL op's arity and outputs from its signature, pairs it with its lowering, and maps
//! each subtile op's name to the facts its row states.

// The decoder state's signatures path is absolute, as a carrier in another crate calls it.
extern crate self as state_macros;

use std::num::NonZeroU8;

use scratchy_decoder_state::lower::{ArgIndex, LowerError, OutIndex, StateLowering, StateSubOp};
use scratchy_forward_compiler_frontend::state::{
    Arity, InputName, OpName, OutputCount, SignaturePath, StateName,
};
use scratchy_subtile::state_op::StateRows;
use scratchy_subtile::tensor_parallel::TpRule;

/// The test decoder state's name.
pub const STATE: StateName = StateName::new("probe");

pub mod ops {
    scratchy_decoder_state::state_op! {
        /// Three operands in, two values out.
        pub fn probe_mix(a: u32, b: u32, c: u32) -> (u32, u32) {
            (a, b + c)
        }
    }

    /// The probe's layered state input.
    pub fn probe_state() -> u32 {
        0
    }
}

scratchy_decoder_state::subtile_op! {
    /// The first of `probe_mix`'s two subtile ops: its three operands in, one row per token out.
    pub static PROBE_MIX_A: name = probe_mix_a, state = STATE, arity = 3, outputs = One,
        rows = M, tp = Whole, weight = none;
    /// The second: `probe_mix_a`'s output and a weight in, one value out.
    pub static PROBE_MIX_B: name = probe_mix_b, state = STATE, arity = 2, outputs = One,
        rows = In0, tp = Whole, weight = 1;
}

/// `probe_mix(a, b, c)`: output 0 is `probe_mix_a` over the three, output 1 is argument 0.
fn lower_mix(l: &mut dyn StateLowering) -> Result<(), LowerError> {
    let args = [
        l.input(ArgIndex::new(0))?,
        l.input(ArgIndex::new(1))?,
        l.input(ArgIndex::new(2))?,
    ];
    let cols = l.out_cols(OutIndex::new(0))?;
    let mixed = l.push(StateSubOp::new(&PROBE_MIX_A, |_| cols), args.to_vec());
    l.produce(OutIndex::new(0), mixed);
    l.produce(OutIndex::new(1), args[0]);
    Ok(())
}

scratchy_decoder_state::decoder_state! {
    /// The test decoder state.
    pub static PROBE: name = STATE, signatures = ::state_macros::ops,
        ops = [probe_mix => lower_mix], inputs = [probe_state],
        subtile = [PROBE_MIX_A, PROBE_MIX_B];
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

    assert!(PROBE.lowering(decl.name).is_some());
    assert!(PROBE.subtile(decl.name).is_none());
    for (facts, name, arity, rows, weight) in [
        (&PROBE_MIX_A, "probe_mix_a", 3, StateRows::M, None),
        (&PROBE_MIX_B, "probe_mix_b", 2, StateRows::In0, Some(1)),
    ] {
        let found = PROBE
            .subtile(OpName::new(name))
            .unwrap_or_else(|| panic!("{name} states facts"));
        assert!(std::ptr::eq(found, facts));
        assert_eq!(found.id().state(), StateName::new("probe"));
        assert_eq!(found.id().name(), OpName::new(name));
        assert_eq!(found.arity(), Arity::new(arity));
        assert_eq!(found.outputs(), OutputCount::ONE);
        assert_eq!(found.rows(), rows);
        assert_eq!(found.tp(), TpRule::Whole);
        assert_eq!(found.weight(), weight);
        assert!(PROBE.lowering(OpName::new(name)).is_none());
    }
    assert!(PROBE.subtile(OpName::new("probe_state")).is_none());

    assert_eq!(ops::probe_mix(1, 2, 3), (1, 5));
    assert_eq!(ops::probe_state(), 0);
}
