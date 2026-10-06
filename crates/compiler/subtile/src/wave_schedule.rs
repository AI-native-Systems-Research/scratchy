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
//! This is target-neutral tape substrate: [`wave_order`] lays a tape's ops out level by level, so
//! a target whose barriers drain everything in flight runs independent branches between the same
//! barriers; the spyre reroll can consume the same levels when its launch-group construction moves
//! tape-side.

use crate::handoff::LoweredDecode;
use crate::lower::{ArchOp, InputRef, LoweringInput};
use crate::tape_colouring::{ColourRule, OutputAlias};

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

/// `l` with its ops in WAVE ORDER: by level, ties in tape order — each op as early as what it
/// reads allows, so independent branches interleave instead of running one after the other.
///
/// Data edges alone do not order a step that writes over a buffer in place (the target's `rule`
/// says which: any output that aliases an operand, and a norm that may fold the step it reads
/// writes over that step's delta) after the steps still reading that buffer's earlier value. So
/// such a step's level is also past every earlier reader of any value living in a buffer it may
/// write. A read no operand names ([`LoweredDecode::unnamed_reads`]) is an edge like any other.
/// Nor do data edges order a fold's driver after the steps it absorbed, whose operands its command
/// reads: each `(ahead, op)` pair — `ahead` before `op` in tape order — keeps `ahead` ahead. An
/// op the target dispatches nothing for (`free`: a step a fold computes inside another, a view)
/// takes no level of its own: its readers may run at its level. The result is a permutation of the
/// ops: the same steps compute the same values.
pub fn wave_order(
    l: &LoweredDecode,
    rule: fn(&ArchOp) -> ColourRule,
    ahead: &[(usize, usize)],
    free: &[bool],
) -> LoweredDecode {
    let ops = &l.input.ops;
    let n = ops.len();
    let producer = |i: usize, k: usize| match ops[i].inputs.get(k) {
        Some(InputRef::Op(j)) => Some(*j),
        _ => None,
    };
    let aliased = |i: usize| match rule(&ops[i].op).output {
        OutputAlias::Operand(k) | OutputAlias::UnitScale(k) | OutputAlias::PeeledWhenNormed(k) => {
            Some(k.index())
        }
        OutputAlias::AbsorbedDelta | OutputAlias::Fresh => None,
    };
    let writes_over = |i: usize| !matches!(rule(&ops[i].op).output, OutputAlias::Fresh);
    // The op whose buffer each op's output lives in.
    let mut owner: Vec<usize> = (0..n).collect();
    for i in 0..n {
        if let Some(j) = aliased(i).and_then(|k| producer(i, k)) {
            owner[i] = owner[j];
        }
    }
    // Every reader of each buffer, in tape order.
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, od) in ops.iter().enumerate() {
        for input in &od.inputs {
            if let InputRef::Op(j) = input {
                readers[owner[*j]].push(i);
            }
        }
    }
    // `wave_levels`, plus the edges data does not carry: in tape order, every level an op waits
    // on is final when it is reached.
    let mut kept_ahead: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, op) in ahead.iter().chain(&l.unnamed_reads) {
        kept_ahead[op].push(a);
    }
    let mut level = vec![0usize; n];
    // The first level an op's readers may take: past its own, or at it when it dispatches nothing.
    let mut ready = vec![0usize; n];
    for i in 0..n {
        for &a in &kept_ahead[i] {
            level[i] = level[i].max(ready[a]);
        }
        let mut over: Vec<usize> = Vec::new();
        for input in &ops[i].inputs {
            if let InputRef::Op(j) = input {
                level[i] = level[i].max(ready[*j]);
                if writes_over(i) {
                    over.push(owner[*j]);
                    // A norm folding the step it reads writes over that step's delta.
                    let delta = rule(&ops[*j].op).delta_when_absorbed;
                    over.extend(
                        delta
                            .and_then(|k| producer(*j, k.index()))
                            .map(|d| owner[d]),
                    );
                }
            }
        }
        for b in over {
            for &r in readers[b].iter().take_while(|r| **r < i) {
                level[i] = level[i].max(level[r] + 1);
            }
        }
        ready[i] = level[i] + usize::from(!free.get(i).copied().unwrap_or(false));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (level[i], i));
    let mut at = vec![0usize; n];
    for (new, &old) in order.iter().enumerate() {
        at[old] = new;
    }
    let remap = |x: &InputRef| match x {
        InputRef::Op(j) => InputRef::Op(at[*j]),
        InputRef::Ext(e) => InputRef::Ext(*e),
    };
    let mut x = l.clone();
    x.input.ops = order
        .iter()
        .map(|&i| {
            let mut od = ops[i].clone();
            od.inputs = od.inputs.iter().map(remap).collect();
            od
        })
        .collect();
    x.input.result = at[l.input.result];
    x.op_tiles = order.iter().map(|&i| l.op_tiles[i]).collect();
    x.op_expansion = order.iter().map(|&i| l.op_expansion[i]).collect();
    x.norm_gain_add_tiles = (l.norm_gain_add_tiles.iter())
        .map(|(op, tile)| (at[*op], *tile))
        .collect();
    x.unnamed_reads = (l.unnamed_reads.iter())
        .map(|(w, r)| (at[*w], at[*r]))
        .collect();
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::OpDesc;
    use crate::subtile_ir::{EwKind, SubOp};

    fn op(inputs: Vec<InputRef>) -> OpDesc {
        OpDesc {
            op: SubOp::Elementwise(EwKind::Add),
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

    const MUL: SubOp<crate::subtile_ir::NeoX, crate::subtile_ir::Arch> =
        SubOp::Elementwise(EwKind::Mul);

    /// An add writes over its operand 1 in place; every other op writes a fresh buffer.
    fn in_place_add(op: &ArchOp) -> ColourRule {
        match op {
            SubOp::Elementwise(EwKind::Add) => {
                ColourRule::FRESH.output(OutputAlias::Operand(crate::tape_steps::OperandIx(1)))
            }
            _ => ColourRule::FRESH,
        }
    }

    fn decode(ops: Vec<OpDesc>) -> LoweredDecode {
        let n = ops.len();
        LoweredDecode {
            input: LoweringInput {
                sources: vec![],
                result: n - 1,
                ops,
            },
            bindings: vec![],
            op_tiles: (0..n as u32).map(|t| Some((t, 0))).collect(),
            norm_gain_add_tiles: Default::default(),
            op_expansion: vec![None; n],
            unnamed_reads: Vec::new(),
        }
    }

    fn mul(inputs: Vec<InputRef>) -> OpDesc {
        OpDesc {
            op: MUL,
            m: 1,
            inputs,
        }
    }

    /// The ops of `l` by their provenance tile.
    fn tiles(l: &LoweredDecode) -> Vec<u32> {
        l.op_tiles.iter().map(|t| t.unwrap().0).collect()
    }

    #[test]
    fn independent_branches_interleave_and_operands_follow_their_ops() {
        use InputRef::{Ext, Op};
        // a: 0 → 1 → 2; b: 3 → 4; 5 joins them.
        let l = decode(vec![
            mul(vec![Ext(0)]),
            mul(vec![Op(0)]),
            mul(vec![Op(1)]),
            mul(vec![Ext(0)]),
            mul(vec![Op(3)]),
            mul(vec![Op(2), Op(4)]),
        ]);
        let w = wave_order(&l, in_place_add, &[], &[]);
        assert_eq!(tiles(&w), [0, 3, 1, 4, 2, 5]);
        // An op kept ahead of another holds it back: 3 stays after 2.
        let kept = wave_order(&l, in_place_add, &[(2, 3)], &[]);
        assert_eq!(tiles(&kept), [0, 1, 2, 3, 4, 5]);
        // The join reads the two ops it read, wherever they went; it is still the result.
        assert_eq!(w.input.ops[5].inputs, [Op(4), Op(3)]);
        assert_eq!(w.input.result, 5);
    }

    #[test]
    fn an_in_place_write_waits_for_the_earlier_readers_of_what_it_overwrites() {
        use InputRef::{Ext, Op};
        // 0 is read by 3, at the end of a chain, and then overwritten in place by the add 4 — whose
        // data alone would put it at level 1.
        let l = decode(vec![
            mul(vec![Ext(0)]),
            mul(vec![Ext(0)]),
            mul(vec![Op(1)]),
            mul(vec![Op(2), Op(0)]),
            op(vec![Ext(0), Op(0)]),
            mul(vec![Op(3), Op(4)]),
        ]);
        let w = wave_order(&l, in_place_add, &[], &[]);
        let at = |tile: u32| tiles(&w).iter().position(|t| *t == tile).unwrap();
        assert!(at(3) < at(4), "{:?}", tiles(&w));
        // A fresh-buffer op with the same operands moves up to its data level.
        let fresh = wave_order(&l, |_| ColourRule::FRESH, &[], &[]);
        assert_eq!(tiles(&fresh), [0, 1, 2, 4, 3, 5]);
    }

    #[test]
    fn a_read_no_operand_names_is_an_edge() {
        use InputRef::{Ext, Op};
        // 2 reads what 0 staged, through storage none of its operands names.
        let mut l = decode(vec![
            mul(vec![Ext(0)]),
            mul(vec![Op(0)]),
            mul(vec![Ext(0)]),
            mul(vec![Op(1), Op(2)]),
        ]);
        assert_eq!(
            tiles(&wave_order(&l, |_| ColourRule::FRESH, &[], &[])),
            [0, 2, 1, 3]
        );
        l.unnamed_reads = vec![(1, 2)];
        let w = wave_order(&l, |_| ColourRule::FRESH, &[], &[]);
        assert_eq!(tiles(&w), [0, 1, 2, 3]);
        // The pair follows its ops.
        assert_eq!(w.unnamed_reads, [(1, 2)]);
    }
}
