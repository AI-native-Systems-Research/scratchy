// SPDX-License-Identifier: Apache-2.0
//! How a decoder state lowers its own DSL ops into the subtile IR: each op's [`LowerFn`] builds
//! through a [`StateLowering`], a state-neutral builder the bridge implements over its own
//! helpers. The builder hands out the op's inputs, output widths, config bounds, cos/sin pairs,
//! weights and layers, and takes only the [`StateSubOp`]s of the decoder state's own listed
//! subtile rows: a push of any other row is refused ([`LowerError::NotOwnRow`]), so a decoder
//! state can push its own subtile ops and nothing else.

pub use scratchy_subtile::handoff::UnrollIndex;
pub use scratchy_subtile::lower::{GemmWeight, InputRef};
pub use scratchy_subtile::state_op::StateSubOp;

use scratchy_forward_compiler_frontend::state::StateOpId;

/// A DSL argument position of the op being lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgIndex(u8);

impl ArgIndex {
    pub const fn new(position: u8) -> Self {
        Self(position)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// A DSL output position of the op being lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutIndex(u8);

impl OutIndex {
    pub const fn new(position: u8) -> Self {
        Self(position)
    }

    pub const fn get(self) -> u8 {
        self.0
    }
}

/// A weight argument as a subtile operand: its source, and the storage scheme the op that reads
/// it is built [`weighted`](StateSubOp::weighted) with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateWeight {
    pub source: InputRef,
    pub scheme: GemmWeight,
}

/// Why a lowering could not build its op, raised at expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowerError {
    /// The op has no argument at `arg`.
    MissingArg { arg: ArgIndex },
    /// The argument at `arg` is not a value the builder resolves: no op produced it, it is not a
    /// tile or a weight, or its shape does not close under the model's bounds.
    Unresolved { arg: ArgIndex },
    /// The argument at `arg` is not a weight.
    NotAWeight { arg: ArgIndex },
    /// The argument at `arg` is not a layered state input.
    NotLayered { arg: ArgIndex },
    /// The width of output `out` does not resolve under the model's bounds.
    UnresolvedWidth { out: OutIndex },
    /// The model's config states no bound `name`.
    MissingBound { name: &'static str },
    /// A width, `what` (a config bound or one a lowering computes), does not fit a `u32`.
    WidthOverflow { what: &'static str },
    /// The weight's storage has no scheme a state op can read it under.
    Weight { detail: &'static str },
    /// A push of the subtile op `op` whose facts are not a row the lowering's decoder state lists.
    NotOwnRow { op: StateOpId },
}

/// The builder a decoder state's lowering of one DSL op writes through.
pub trait StateLowering {
    /// The subtile value of DSL argument `arg`.
    fn input(&mut self, arg: ArgIndex) -> Result<InputRef, LowerError>;
    /// The column width of DSL output `out`.
    fn out_cols(&self, out: OutIndex) -> Result<u32, LowerError>;
    /// The model's config bound `name`, as the `u32` widths are counted in.
    fn bound(&self, name: &'static str) -> Result<u32, LowerError>;
    /// The cos and sin sources at `cols` columns, one pair per width.
    fn cos_sin(&mut self, cols: u32) -> (InputRef, InputRef);
    /// DSL argument `arg` as a weight: its source and its storage scheme.
    fn weight(&mut self, arg: ArgIndex) -> Result<StateWeight, LowerError>;
    /// The layer DSL argument `arg`, a layered state input, names.
    fn layer(&self, arg: ArgIndex) -> Result<UnrollIndex, LowerError>;
    /// Appends `op` over `inputs`, returning its output; an op whose facts are not one of the
    /// decoder state's own listed rows is refused.
    fn push(&mut self, op: StateSubOp, inputs: Vec<InputRef>) -> Result<InputRef, LowerError>;
    /// Records `value` as DSL output `out`.
    fn produce(&mut self, out: OutIndex, value: InputRef);
}

/// A decoder state's lowering of one of its DSL ops.
pub type LowerFn = fn(&mut dyn StateLowering) -> Result<(), LowerError>;

#[cfg(feature = "test-support")]
pub use recorder::Recorder;

#[cfg(feature = "test-support")]
mod recorder {
    use std::collections::BTreeMap;

    use super::{
        ArgIndex, InputRef, LowerError, OutIndex, StateLowering, StateSubOp, StateWeight,
        UnrollIndex,
    };
    use crate::DecoderState;

    /// A [`StateLowering`] over given inputs (one per argument), output widths (one per output),
    /// bounds, weights and layers, which records every op pushed, every output produced and
    /// every cos/sin pair asked for. Each cos/sin pair is a fresh pair of external sources past
    /// every source the inputs and weights name; each pushed op's output is
    /// [`InputRef::Op`] of its push order. A push of a row `state` does not list is refused, as
    /// the bridge refuses it.
    pub struct Recorder {
        state: DecoderState,
        inputs: Vec<InputRef>,
        widths: Vec<u32>,
        bounds: BTreeMap<&'static str, u32>,
        weights: BTreeMap<u8, StateWeight>,
        layers: BTreeMap<u8, UnrollIndex>,
        next_ext: usize,
        pushes: Vec<(StateSubOp, Vec<InputRef>)>,
        produces: Vec<(OutIndex, InputRef)>,
        cos_sins: Vec<(u32, InputRef, InputRef)>,
    }

    fn past(r: InputRef) -> usize {
        match r {
            InputRef::Ext(e) => e + 1,
            InputRef::Op(_) => 0,
        }
    }

    impl Recorder {
        /// A recorder for a lowering of `state`, whose argument `i` is `inputs[i]` and output `o`
        /// is `widths[o]` wide.
        pub fn new(state: DecoderState, inputs: Vec<InputRef>, widths: Vec<u32>) -> Self {
            let next_ext = inputs.iter().copied().map(past).max().unwrap_or(0);
            Self {
                state,
                inputs,
                widths,
                bounds: BTreeMap::new(),
                weights: BTreeMap::new(),
                layers: BTreeMap::new(),
                next_ext,
                pushes: Vec::new(),
                produces: Vec::new(),
                cos_sins: Vec::new(),
            }
        }

        /// This recorder, with the config bound `name` at `value`.
        pub fn with_bound(mut self, name: &'static str, value: u32) -> Self {
            self.bounds.insert(name, value);
            self
        }

        /// This recorder, with argument `arg` the weight `weight`.
        pub fn with_weight(mut self, arg: ArgIndex, weight: StateWeight) -> Self {
            self.next_ext = self.next_ext.max(past(weight.source));
            self.weights.insert(arg.get(), weight);
            self
        }

        /// This recorder, with argument `arg` a layered state input at `layer`.
        pub fn with_layer(mut self, arg: ArgIndex, layer: UnrollIndex) -> Self {
            self.layers.insert(arg.get(), layer);
            self
        }

        /// Every op pushed, with its inputs, in push order.
        pub fn pushes(&self) -> &[(StateSubOp, Vec<InputRef>)] {
            &self.pushes
        }

        /// Every output produced, with its value, in produce order.
        pub fn produces(&self) -> &[(OutIndex, InputRef)] {
            &self.produces
        }

        /// Every cos/sin pair asked for: its width, then the pair.
        pub fn cos_sins(&self) -> &[(u32, InputRef, InputRef)] {
            &self.cos_sins
        }
    }

    impl StateLowering for Recorder {
        fn input(&mut self, arg: ArgIndex) -> Result<InputRef, LowerError> {
            self.inputs
                .get(usize::from(arg.get()))
                .copied()
                .ok_or(LowerError::MissingArg { arg })
        }

        fn out_cols(&self, out: OutIndex) -> Result<u32, LowerError> {
            self.widths
                .get(usize::from(out.get()))
                .copied()
                .ok_or(LowerError::UnresolvedWidth { out })
        }

        fn bound(&self, name: &'static str) -> Result<u32, LowerError> {
            self.bounds
                .get(name)
                .copied()
                .ok_or(LowerError::MissingBound { name })
        }

        fn cos_sin(&mut self, cols: u32) -> (InputRef, InputRef) {
            let (cos, sin) = (
                InputRef::Ext(self.next_ext),
                InputRef::Ext(self.next_ext + 1),
            );
            self.next_ext += 2;
            self.cos_sins.push((cols, cos, sin));
            (cos, sin)
        }

        fn weight(&mut self, arg: ArgIndex) -> Result<StateWeight, LowerError> {
            self.weights
                .get(&arg.get())
                .copied()
                .ok_or(LowerError::NotAWeight { arg })
        }

        fn layer(&self, arg: ArgIndex) -> Result<UnrollIndex, LowerError> {
            self.layers
                .get(&arg.get())
                .copied()
                .ok_or(LowerError::NotLayered { arg })
        }

        fn push(&mut self, op: StateSubOp, inputs: Vec<InputRef>) -> Result<InputRef, LowerError> {
            if !self.state.owns(op.facts()) {
                return Err(LowerError::NotOwnRow { op: op.id() });
            }
            self.pushes.push((op, inputs));
            Ok(InputRef::Op(self.pushes.len() - 1))
        }

        fn produce(&mut self, out: OutIndex, value: InputRef) {
            self.produces.push((out, value));
        }
    }
}
