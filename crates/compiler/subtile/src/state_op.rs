// SPDX-License-Identifier: Apache-2.0
//! A state op as the subtile IR carries it: ONE opaque [`SubOp::State`] variant whose facts
//! travel with it, so the shared op vocabulary never grows a variant per decoder state. The facts a
//! registry column reads (arity, output count, row law, sharding rule) are the decoder state's own,
//! stated once in its [`StateOpFacts`] and pointed at by every instance; the instance adds only
//! what one call site fixes, its output widths ([`OutCols`]).
//!
//! [`SubOp::State`]: crate::subtile_ir::SubOp::State
//! [`StateOpFacts`]: crate::state_op::StateOpFacts
//! [`OutCols`]: crate::state_op::OutCols

use std::num::NonZeroU8;

use scratchy_tensors::compiler_ids::{Arity, OutputCount, StateOpId};

use crate::tensor_parallel::TpRule;

/// How many values a state op produces: one variant per count a state op may declare, so a
/// declaration of more names no variant and does not build.
///
/// ```compile_fail,E0599
/// let _ = scratchy_subtile::state_op::StateOutputs::Four;
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StateOutputs {
    One = 1,
    Two = 2,
    Three = 3,
}

impl StateOutputs {
    /// The most outputs a state op may declare.
    pub const MAX: Self = Self::Three;

    /// The count, as the front end states an op's outputs.
    pub const fn count(self) -> OutputCount {
        OutputCount::new(NonZeroU8::MIN.saturating_add(self as u8 - 1))
    }
}

/// The capacity of [`OutCols`]: the most outputs a state op may declare ([`StateOutputs::MAX`]).
pub const STATE_OUTPUT_SLOTS: usize = StateOutputs::MAX as usize;

/// The op row law a state op obeys: the registry's `rows` classes a state op may name
/// (`[in0]` and `[m]` in [`crate::ops`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StateRows {
    /// Output rows are operand 0's registered rows.
    In0,
    /// Output rows are the token count.
    M,
}

/// Everything the shared passes know about one state op, declared once in the decoder state's own
/// crate: its identity, its operand arity, its output count, its row law and its sharding rule.
#[derive(Debug)]
pub struct StateOpFacts {
    id: StateOpId,
    arity: Arity,
    outputs: StateOutputs,
    rows: StateRows,
    tp: TpRule,
}

impl StateOpFacts {
    /// The facts of the op `id`.
    pub const fn new(
        id: StateOpId,
        arity: Arity,
        outputs: StateOutputs,
        rows: StateRows,
        tp: TpRule,
    ) -> Self {
        Self {
            id,
            arity,
            outputs,
            rows,
            tp,
        }
    }

    pub const fn id(&self) -> StateOpId {
        self.id
    }

    pub const fn arity(&self) -> Arity {
        self.arity
    }

    pub const fn outputs(&self) -> OutputCount {
        self.outputs.count()
    }

    pub const fn rows(&self) -> StateRows {
        self.rows
    }

    pub const fn tp(&self) -> TpRule {
        self.tp
    }

    /// Each output slot the op declares, in order.
    pub fn slots(&self) -> impl Iterator<Item = OutputSlot> {
        (0..self.outputs as u8).map(OutputSlot)
    }
}

/// One output slot of a state op, minted only by [`StateOpFacts::slots`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputSlot(u8);

impl OutputSlot {
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// One state op instance's output widths: output 0's, which every op has, and one for each
/// further output its facts declare. An output the op does not declare holds no width, so two
/// instances compare by their declared widths alone and no reader can take one for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutCols {
    first: u32,
    rest: [Option<u32>; STATE_OUTPUT_SLOTS - 1],
}

impl OutCols {
    /// The width of output 0, which every op has: the registry's `out_cols`.
    pub fn first(self) -> u32 {
        self.first
    }

    /// The width of output `slot`, `None` past the outputs this instance holds.
    fn get(self, slot: OutputSlot) -> Option<u32> {
        match slot.0.checked_sub(1) {
            None => Some(self.first),
            Some(k) => self.rest.get(usize::from(k)).copied().flatten(),
        }
    }
}

/// A state op as one node performs it: its decoder state's facts and this instance's output widths.
/// The facts pointer and the id it carries travel together, so one op's facts are never read for
/// another's.
#[derive(Clone, Copy, Debug)]
pub struct StateSubOp {
    facts: &'static StateOpFacts,
    out_cols: OutCols,
}

impl StateSubOp {
    /// An instance of the op `facts` describes, whose output `slot` is `width(slot)` wide: asked
    /// once for each slot the facts declare, so the instance has exactly that many widths.
    pub fn new(facts: &'static StateOpFacts, mut width: impl FnMut(OutputSlot) -> u32) -> Self {
        let first = width(OutputSlot(0));
        let mut rest = [None; STATE_OUTPUT_SLOTS - 1];
        for (at, slot) in rest.iter_mut().zip(facts.slots().skip(1)) {
            *at = Some(width(slot));
        }
        Self {
            facts,
            out_cols: OutCols { first, rest },
        }
    }

    /// The width of output `slot`: `None` for a slot this op does not declare.
    pub fn width(self, slot: OutputSlot) -> Option<u32> {
        self.out_cols.get(slot)
    }

    /// Each output this op declares, with its width, in order.
    pub fn widths(self) -> impl Iterator<Item = (OutputSlot, u32)> {
        let cols = self.out_cols;
        self.facts
            .slots()
            .filter_map(move |slot| cols.get(slot).map(|w| (slot, w)))
    }

    pub fn facts(self) -> &'static StateOpFacts {
        self.facts
    }

    pub fn id(self) -> StateOpId {
        self.facts.id
    }

    pub fn out_cols(self) -> OutCols {
        self.out_cols
    }

    /// The registry's `name`: the op's own name.
    pub fn name(self) -> &'static str {
        self.facts.id.name().get()
    }

    /// The registry's `arity_ok`: whether `n_operands` is the op's declared arity.
    pub fn arity_ok(self, n_operands: usize) -> bool {
        n_operands == usize::from(self.facts.arity.get())
    }

    /// The registry's `out_rows`, under the op's own row law.
    pub fn out_rows(self, m: u32, operand_rows: impl Fn(usize) -> u32) -> u32 {
        match self.facts.rows {
            StateRows::In0 => operand_rows(0),
            StateRows::M => m,
        }
    }

    /// The registry's `tp_rule`: the op's declared sharding rule.
    pub fn tp_rule(self) -> TpRule {
        self.facts.tp
    }
}

/// Equal when they are the same op with the same output widths: the id, not the facts' address.
impl PartialEq for StateSubOp {
    fn eq(&self, other: &Self) -> bool {
        (self.facts.id, self.out_cols) == (other.facts.id, other.out_cols)
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hasher};
    use std::num::NonZeroU32;

    use scratchy_tensors::compiler_ids::{OpName, StateName};

    use super::*;
    use crate::lower::{InputRef, LoweringInput, OpDesc};
    use crate::subtile_ir::{SourceShape, SubOp, lower_region};

    const PROBE: StateName = StateName::new("probe");

    static PROBE_A: StateOpFacts = StateOpFacts::new(
        StateOpId::new(PROBE, OpName::new("probe_a")),
        Arity::new(2),
        StateOutputs::One,
        StateRows::In0,
        TpRule::Whole,
    );

    static PROBE_B: StateOpFacts = StateOpFacts::new(
        StateOpId::new(PROBE, OpName::new("probe_b")),
        Arity::new(1),
        StateOutputs::Two,
        StateRows::M,
        TpRule::Heads,
    );

    fn op(facts: &'static StateOpFacts, cols: u32) -> SubOp {
        SubOp::State(StateSubOp::new(facts, |slot| cols + u32::from(slot.get())))
    }

    fn class_key(op: &SubOp) -> u64 {
        let mut h = DefaultHasher::new();
        op.reroll_class_key(&mut h);
        h.finish()
    }

    #[test]
    fn a_state_op_reads_its_own_facts() {
        let a = op(&PROBE_A, 64);
        assert!(a.arity_ok(2));
        assert!(!a.arity_ok(1));
        assert_eq!(a.out_rows(1, |_| 7), 7);
        assert_eq!(a.out_cols(|_| 9), 64);
        assert_eq!(a.tp_rule(), TpRule::Whole);
        assert_eq!(a.name(), "probe_a");
        assert_eq!(a.gemm_weight_input(), None);
        assert_eq!(a.additive_offset(), None);

        let b = op(&PROBE_B, 32);
        assert!(b.arity_ok(1));
        assert_eq!(b.out_rows(1, |_| 7), 1);
        assert_eq!(b.tp_rule(), TpRule::Heads);
        assert_eq!(b.name(), "probe_b");
        let SubOp::State(b) = b else {
            panic!("a state op is SubOp::State");
        };
        let widths: Vec<(u8, u32)> = b.widths().map(|(s, w)| (s.get(), w)).collect();
        assert_eq!(widths, [(0, 32), (1, 33)]);
        assert_eq!(PROBE_B.outputs().get().get(), 2);
        assert_eq!(STATE_OUTPUT_SLOTS, 3);
    }

    #[test]
    fn a_width_is_read_only_for_a_declared_output() {
        let SubOp::State(a) = op(&PROBE_A, 64) else {
            panic!("a state op is SubOp::State");
        };
        let b_slots: Vec<OutputSlot> = PROBE_B.slots().collect();
        assert_eq!(a.width(b_slots[0]), Some(64));
        assert_eq!(a.width(b_slots[1]), None);
        let a_widths: Vec<(u8, u32)> = a.widths().map(|(s, w)| (s.get(), w)).collect();
        assert_eq!(a_widths, [(0, 64)]);
    }

    #[test]
    fn two_state_ops_are_two_classes() {
        assert_ne!(class_key(&op(&PROBE_A, 64)), class_key(&op(&PROBE_B, 64)));
        assert_eq!(class_key(&op(&PROBE_A, 64)), class_key(&op(&PROBE_A, 64)));
        assert_ne!(class_key(&op(&PROBE_A, 64)), class_key(&op(&PROBE_A, 32)));
        assert_ne!(op(&PROBE_A, 64), op(&PROBE_B, 64));
        assert_eq!(op(&PROBE_A, 64), op(&PROBE_A, 64));
    }

    #[test]
    fn a_state_op_lowers_its_region_unchanged() {
        let state = op(&PROBE_A, 48);
        let SubOp::State(arch) = state else {
            panic!("a state op is SubOp::State");
        };
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: 16 },
                SourceShape { rows: 1, cols: 8 },
            ],
            ops: vec![OpDesc {
                op: SubOp::State(arch),
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            }],
            result: 0,
            exported: None,
        };
        let graph = lower_region(&input, NonZeroU32::new(64).unwrap());
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].op, state);
        assert_eq!(graph.nodes[0].inputs.len(), 2);
        let out = graph.nodes[0].output.region;
        assert_eq!((out.rows.len, out.cols.len), (1, 48));
    }
}
