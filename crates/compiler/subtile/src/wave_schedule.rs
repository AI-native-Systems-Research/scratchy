// SPDX-License-Identifier: Apache-2.0
//! Tape-owned wave scheduling: dependence levels over a
//! [`LoweringInput`]'s op graph.
//!
//! The tape's `ops` are already in topological order; this pass
//! assigns each op its longest-path DEPTH from the graph's sources
//! (level 0 = no op inputs). Two ops share a level exactly when
//! neither transitively depends on the other along data edges, so a
//! level is a wave: its members may dispatch concurrently between
//! barriers. Hazards that data edges don't carry (KV cache
//! write→read, colored-slot reuse) are the emitter's barrier walk's
//! concern — levels only promise data-dependence safety, which is
//! the same contract the solver's waves gave the emitters.
//!
//! This is target-neutral tape substrate: metal keys its emission
//! groups off these levels; the spyre reroll can consume the same
//! function when its launch-group construction moves tape-side.

use crate::lower::{InputRef, LoweringInput};

/// Longest-path dependence level per op, parallel to `input.ops`.
///
/// `levels[i] = 1 + max(levels[j])` over `InputRef::Op(j)` inputs
/// (0 when the op reads only external sources). Panics if the tape
/// violates its own topological-order invariant (`j < i`).
pub fn wave_levels(input: &LoweringInput) -> Vec<usize> {
    let mut levels = Vec::with_capacity(input.ops.len());
    for (i, od) in input.ops.iter().enumerate() {
        let lvl = od
            .inputs
            .iter()
            .filter_map(|r| match r {
                InputRef::Op(j) => {
                    assert!(*j < i, "tape op {i} references op {j} (not topological)");
                    Some(levels[*j] + 1)
                }
                InputRef::Ext(_) => None,
            })
            .max()
            .unwrap_or(0);
        levels.push(lvl);
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{LoweredOp, OpDesc};

    fn op(inputs: Vec<InputRef>) -> OpDesc {
        OpDesc {
            op: LoweredOp::Add,
            m: 1,
            inputs,
        }
    }

    #[test]
    fn diamond_levels() {
        // 0     (src)
        // ├─ 1  (reads 0)
        // ├─ 2  (reads 0)
        // └─ 3  (reads 1 and 2)
        let input = LoweringInput {
            sources: vec![],
            ops: vec![
                op(vec![InputRef::Ext(0)]),
                op(vec![InputRef::Op(0)]),
                op(vec![InputRef::Op(0), InputRef::Ext(0)]),
                op(vec![InputRef::Op(1), InputRef::Op(2)]),
            ],
            result: 3,
        };
        assert_eq!(wave_levels(&input), vec![0, 1, 1, 2]);
    }

    #[test]
    fn chain_is_sequential() {
        let input = LoweringInput {
            sources: vec![],
            ops: vec![
                op(vec![InputRef::Ext(0)]),
                op(vec![InputRef::Op(0)]),
                op(vec![InputRef::Op(1)]),
            ],
            result: 2,
        };
        assert_eq!(wave_levels(&input), vec![0, 1, 2]);
    }
}
