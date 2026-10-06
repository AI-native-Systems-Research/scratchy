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

/// Per op of `n`, the op its `(op, next)` links chain to: itself when it has none.
fn chain_ends(n: usize, links: &[(usize, usize)]) -> Vec<usize> {
    let mut next: Vec<usize> = (0..n).collect();
    for &(op, to) in links {
        next[op] = to;
    }
    (0..n)
        .map(|mut i| {
            for _ in 0..n {
                if next[i] == i {
                    return i;
                }
                i = next[i];
            }
            panic!("a fold's links chain into a cycle at op {i}")
        })
        .collect()
}

/// `l` with its ops in WAVE ORDER: by level, ties in tape order — each op as early as what it
/// reads allows, so independent branches interleave instead of running one after the other.
///
/// Data edges alone do not order a step that writes over a buffer in place (the target's `rule`
/// says which: any output that aliases an operand, and a norm that may fold the step it reads
/// writes over that step's delta) after the steps still reading that buffer's earlier value. So
/// such a step's level is also past every earlier reader of any value living in a buffer it may
/// write. A read no operand names ([`LoweredDecode::unnamed_reads`]) is an edge like any other.
///
/// Nor do data edges order a fold's driver after the steps it absorbed, whose operands its command
/// reads: each `(step, driver)` of `absorbed` — `step` before `driver` in tape order — keeps the
/// step ahead, and what the step reads, the driver's command reads. And a fold's epilogue is its
/// driver's command's own write: each `(epilogue, driver)` of `epilogues` runs at its driver's
/// level, the two kept together in tape order, and a step reading either runs past them — so the
/// folds the tape order made fold again in wave order, never split into a command each. An op the
/// target dispatches nothing for (`free`: a step a fold computes inside another, a view) takes no
/// level of its own: its readers may run at its level. The result is a permutation of the ops: the
/// same steps compute the same values.
pub fn wave_order(
    l: &LoweredDecode,
    rule: fn(&ArchOp) -> ColourRule,
    absorbed: &[(usize, usize)],
    epilogues: &[(usize, usize)],
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
    // The command each op runs in: the driver its epilogues chain up to.
    let command = chain_ends(n, epilogues);
    // Each command's first op in tape order: where its members sit among their level's.
    let mut first: Vec<usize> = (0..n).collect();
    for i in (0..n).rev() {
        first[command[i]] = first[command[i]].min(i);
    }
    let dispatches = |c: usize| usize::from(!free.get(c).copied().unwrap_or(false));
    // Every op's constraints from outside its command: the commands it reads (data, a read no
    // operand names, the steps its fold absorbed), and the readers of what it writes over in place.
    let mut kept_ahead: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(w, r) in absorbed.iter().chain(&l.unnamed_reads) {
        kept_ahead[r].push(w);
    }
    // The op whose command makes each op's reads: an absorbed step reads in its driver's.
    let reads_in = chain_ends(n, absorbed);
    let mut after: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for i in 0..n {
        let mut over: Vec<usize> = Vec::new();
        for input in &ops[i].inputs {
            if let InputRef::Op(j) = input {
                after[i].push((command[*j], dispatches(command[*j])));
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
        for &w in &kept_ahead[i] {
            after[i].push((command[w], dispatches(command[w])));
        }
        for b in over {
            for &r in readers[b].iter().take_while(|r| **r < i) {
                let x = reads_in[r];
                if x < i {
                    after[i].push((command[x], 1));
                }
            }
        }
        after[i].retain(|&(c, _)| c != command[i]);
    }
    // `wave_levels` over commands: a command waits on every command any member waits on. A
    // command's members can follow an op reading another member in tape order, so the levels are
    // raised until none moves.
    let mut level = vec![0usize; n];
    let settled = (0..=n).any(|_| {
        let mut moved = false;
        for i in 0..n {
            let at = (after[i].iter())
                .map(|&(c, past)| level[c] + past)
                .max()
                .unwrap_or(0);
            if at > level[command[i]] {
                level[command[i]] = at;
                moved = true;
            }
        }
        !moved
    });
    assert!(
        settled,
        "a fold's command waits on its own epilogue's reader"
    );
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (level[command[i]], first[command[i]], i));
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
        let w = wave_order(&l, in_place_add, &[], &[], &[]);
        assert_eq!(tiles(&w), [0, 3, 1, 4, 2, 5]);
        // An op kept ahead of another holds it back: 3 stays after 2.
        let kept = wave_order(&l, in_place_add, &[(2, 3)], &[], &[]);
        assert_eq!(tiles(&kept), [0, 1, 2, 3, 4, 5]);
        // The join reads the two ops it read, wherever they went; it is still the result.
        assert_eq!(w.input.ops[5].inputs, [Op(4), Op(3)]);
        assert_eq!(w.input.result, 5);
    }

    #[test]
    fn an_epilogue_runs_with_its_driver_and_its_readers_run_past_them() {
        use InputRef::{Ext, Op};
        // Gemma 4's MoE entry: one command (driver 5) computes the norm 1 (absorbed), the add 2
        // others read (its epilogue) and the norm 5; the router 3 and the dense gate/up 4 read the
        // add, the experts 6 read the norm 5 and the router, the down 7 the gate/up; 8 joins.
        let l = decode(vec![
            mul(vec![Ext(0)]),
            mul(vec![Op(0)]),
            mul(vec![Op(1), Ext(0)]),
            mul(vec![Op(2)]),
            mul(vec![Op(2)]),
            mul(vec![Op(2)]),
            mul(vec![Op(5), Op(3)]),
            mul(vec![Op(4)]),
            mul(vec![Op(6), Op(7)]),
        ]);
        // The absorbed norm alone kept ahead, the norm 5 lands among the add's readers, away from
        // the add: a fold of the reordered tape can no longer take it.
        let absorbed = [false, true, false, false, false, false, false, false, false];
        let apart = wave_order(&l, |_| ColourRule::FRESH, &[(1, 5)], &[], &absorbed);
        assert_eq!(tiles(&apart), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        // As one command, the three run together and every reader of the add follows them.
        let whole = wave_order(&l, |_| ColourRule::FRESH, &[(1, 5)], &[(2, 5)], &absorbed);
        assert_eq!(tiles(&whole), [0, 1, 2, 5, 3, 4, 6, 7, 8]);
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
        let w = wave_order(&l, in_place_add, &[], &[], &[]);
        let at = |tile: u32| tiles(&w).iter().position(|t| *t == tile).unwrap();
        assert!(at(3) < at(4), "{:?}", tiles(&w));
        // A fresh-buffer op with the same operands moves up to its data level.
        let fresh = wave_order(&l, |_| ColourRule::FRESH, &[], &[], &[]);
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
            tiles(&wave_order(&l, |_| ColourRule::FRESH, &[], &[], &[])),
            [0, 2, 1, 3]
        );
        l.unnamed_reads = vec![(1, 2)];
        let w = wave_order(&l, |_| ColourRule::FRESH, &[], &[], &[]);
        assert_eq!(tiles(&w), [0, 1, 2, 3]);
        // The pair follows its ops.
        assert_eq!(w.unnamed_reads, [(1, 2)]);
    }
}
