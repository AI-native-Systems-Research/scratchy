// SPDX-License-Identifier: Apache-2.0
//! THE UNROLLED TAPE, READ AS STEPS — each `Compute` with the source op it computes and its
//! positional operands.
//!
//! Every pass that keys a target decision on the tape's steps reads the tape the same way: a step
//! writes one [`SlotId`], performs its node's [`SubOp`] (the op is where a target's facts are
//! keyed) for one source op of the `LoweringInput` the graph was lowered from, and reads each
//! operand either from an earlier step or from a graph source. This is that reading, once, for the
//! colourer and the folder.

use std::collections::HashMap;

use crate::subtile_ir::{SubOp, SubtileIR, TensorId};
use crate::subtile_tape::{ComputeInput, Instr, SlotId, SubtileTape};

/// A step's position on the tape, counting `Compute`s only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StepPos(pub(crate) u32);

impl std::fmt::Display for StepPos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "step {}", self.0)
    }
}

/// A step's positional operand index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OperandIx(pub u8);

impl OperandIx {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl std::fmt::Display for OperandIx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Why the tape could not be read as steps.
#[derive(Clone, Debug)]
pub enum TapeReadError {
    /// The tape holds a loop; read the unrolled tape.
    Rolled,
    /// A step writes a tensor no source op produces, so no fact is keyed on it.
    NoSourceOp { step: StepPos },
    /// A step reads a column-split producer; one operand has no single producing step.
    SplitOperand {
        step: StepPos,
        op: &'static str,
        operand: OperandIx,
    },
}

impl std::fmt::Display for TapeReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rolled => write!(f, "the tape holds a loop; the pass reads the unrolled tape"),
            Self::NoSourceOp { step } => {
                write!(f, "{step} writes a tensor no source op produces")
            }
            Self::SplitOperand { step, op, operand } => write!(
                f,
                "{step} ({op}) operand {operand} has several writers; one buffer per operand \
                 cannot hold it"
            ),
        }
    }
}

/// A step's operand: an earlier step's output (by tape position), or a graph source.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Operand {
    Step(usize),
    Source(TensorId),
}

/// One `Compute` of the unrolled tape.
pub(crate) struct Step<'a> {
    pub(crate) writes: SlotId,
    /// The `LoweringInput` op this step computes.
    pub(crate) source_op: usize,
    /// The op the step's node performs.
    pub(crate) op: &'a SubOp,
    pub(crate) operands: Vec<Operand>,
    pub(crate) output: TensorId,
}

/// The unrolled `tape` of `graph`, in tape order.
pub(crate) fn read_steps<'a>(
    graph: &'a SubtileIR,
    tape: &SubtileTape,
) -> Result<Vec<Step<'a>>, TapeReadError> {
    let source_op: HashMap<TensorId, usize> = graph
        .op_output
        .iter()
        .enumerate()
        .map(|(j, t)| (*t, j))
        .collect();
    let mut pos_of: HashMap<SlotId, usize> = HashMap::new();
    let mut steps: Vec<Step<'a>> = Vec::new();
    for instr in tape.instrs() {
        let Instr::Compute {
            node,
            writes,
            inputs,
            ..
        } = instr
        else {
            if matches!(instr, Instr::OpenLoop { .. } | Instr::CloseLoop { .. }) {
                return Err(TapeReadError::Rolled);
            }
            continue;
        };
        let p = steps.len();
        let node = &graph.nodes[node.index()];
        let (output, op) = (node.output.tensor, &node.op);
        let j = *source_op.get(&output).ok_or(TapeReadError::NoSourceOp {
            step: StepPos(p as u32),
        })?;
        let operands = inputs
            .iter()
            .enumerate()
            .map(|(k, input)| match input {
                ComputeInput::Computed(slots) => match slots.as_slice() {
                    [one] => Ok(Operand::Step(
                        *pos_of
                            .get(one)
                            .expect("the tape writes every slot before reading it"),
                    )),
                    _ => Err(TapeReadError::SplitOperand {
                        step: StepPos(p as u32),
                        op: op.name(),
                        operand: OperandIx(k as u8),
                    }),
                },
                ComputeInput::External { tensor, .. } => Ok(Operand::Source(*tensor)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        pos_of.insert(*writes, p);
        steps.push(Step {
            writes: *writes,
            source_op: j,
            op,
            operands,
            output,
        });
    }
    Ok(steps)
}
