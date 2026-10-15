// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! SPLAT SINK — move every whole-tile constant to just before its first use.
//!
//! A program's scalar constants reach the emulator as `tensor.splat`s — one full-tile tensor
//! per constant per tile shape — and the producer's canonicalization hoists every one of them
//! to the function entry. Each is an LX tile on the per-core 2 MiB scratchpad from the moment
//! it is bound until its last use, so hoisting makes EVERY block's constants co-resident with
//! the first block's working set.
//!
//! The row-block law (`ktir_superdsc::superdsc_opspec::rows_per_block`) sizes a block so ONE
//! block's live set fits LX; it cannot see constants that belong to a LATER block but are
//! already resident. MEASURED (gemma-4-12b emu prefill, m = 19, `gelu_fwd` over `[19, 15360]`
//! blocked 11 + 8): the kernel's three constants (`0.044715`, `2·√(2/π)`, `1.0`) were splatted
//! at `[11, 15360]` AND at `[8, 15360]`, all six at the entry, so the 11-row block ran with the
//! tail's three tiles beside its own — `6 · splat + x = 2,088,960 B` resident, and the block's
//! `divf` overflowed the 2,097,152 B scratchpad.
//!
//! Sinking is a pure reordering: a splat has no side effect and its only operand is a scalar
//! defined before it, so moving it down to the op that first reads it changes no value and no
//! dependency. With every splat born inside its block, a block's constants die with the block
//! (consume-on-last-use) before the next block's are bound.
//!
//! `tensor.empty` is the same class (a pure, operand-free tile allocation) and is sunk by the
//! same rule. Applied at the emulator's execution entry only — the card's KTIR is untouched, and
//! on the card a scalar constant is a register operand, never an LX tile.

use ktir_core::arena::Arena;
use ktir_core::ir::{Attr, IRModule, Operation, Ssa};
use ktir_core::opkind::OpKind;
use std::collections::HashMap;

/// The ops this pass moves: pure tile materializations with no side effect.
fn sinkable(op: &Operation<'_>) -> bool {
    matches!(op.op_type, OpKind::TensorSplat | OpKind::TensorEmpty)
        && op.result.is_some()
        && op.regions.is_empty()
}

/// Every SSA `op` reads — operands, `Attr::Ssas`, and anything read inside its regions.
fn reads(op: &Operation<'_>, out: &mut Vec<Ssa>) {
    out.extend_from_slice(op.operands);
    for (_, attr) in op.attributes {
        if let Attr::Ssas(xs) = attr {
            out.extend_from_slice(xs);
        }
    }
    for region in op.regions {
        for inner in *region {
            reads(inner, out);
        }
    }
}

/// Reorder one op list: each sinkable op moves to just before the first later op that reads
/// it. Recurses into regions. Returns `None` when nothing moved (the list is left as is).
fn sink_list<'a>(a: &'a Arena, ops: &'a [Operation<'a>]) -> Option<&'a [Operation<'a>]> {
    let mut changed = false;
    // Regions first, so a nested body is in its sunk form before its parent is placed.
    let ops: Vec<Operation<'a>> = ops
        .iter()
        .map(|op| {
            if op.regions.is_empty() {
                return *op;
            }
            let mut any = false;
            let bodies: Vec<&'a [Operation<'a>]> = op
                .regions
                .iter()
                .map(|r| match sink_list(a, r) {
                    Some(new) => {
                        any = true;
                        new
                    }
                    None => *r,
                })
                .collect();
            if any {
                changed = true;
                Operation {
                    regions: a.regions(bodies),
                    ..*op
                }
            } else {
                *op
            }
        })
        .collect();
    // First top-level reader of every value.
    let mut first_read: HashMap<Ssa, usize> = HashMap::new();
    let mut buf = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        buf.clear();
        reads(op, &mut buf);
        for v in &buf {
            first_read.entry(*v).or_insert(i);
        }
    }
    // Sinkable op -> the index it must precede (a dead one stays where it is).
    let mut parked: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut moved = vec![false; ops.len()];
    for (i, op) in ops.iter().enumerate() {
        if !sinkable(op) {
            continue;
        }
        let r = op.result.expect("sinkable ops have a result");
        if let Some(&j) = first_read.get(&r)
            && j > i + 1
        {
            parked.entry(j).or_default().push(i);
            moved[i] = true;
        }
    }
    if !moved.iter().any(|&m| m) {
        return changed.then(|| a.ops(ops));
    }
    let mut out = Vec::with_capacity(ops.len());
    for (j, op) in ops.iter().enumerate() {
        if let Some(srcs) = parked.get(&j) {
            for &i in srcs {
                out.push(ops[i]);
            }
        }
        if !moved[j] {
            out.push(*op);
        }
    }
    Some(a.ops(out))
}

/// Apply the splat sink to every function in `module`, in place. Returns the number of
/// functions whose op order changed.
pub fn apply_splat_sink<'a>(a: &'a Arena, module: &mut IRModule<'a>) -> usize {
    let mut rewritten = 0usize;
    for func in module.functions.values_mut() {
        if let Some(new) = sink_list(a, func.operations) {
            func.operations = new;
            rewritten += 1;
        }
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_builder::Ops;

    /// The measured gelu shape in miniature: two blocks, each with its own constant splatted at
    /// its own tile shape, both hoisted to the entry. After the sink each splat sits directly
    /// before the op that reads it, and every other op keeps its relative order.
    #[test]
    fn hoisted_splats_sink_to_their_first_reader() {
        let mut o = Ops::new();
        let c_a = o.op(Some("%ca"), OpKind::ArithConstant, &[]);
        let s_a = o.op(Some("%sa"), OpKind::TensorSplat, &["%ca"]);
        let c_b = o.op(Some("%cb"), OpKind::ArithConstant, &[]);
        let s_b = o.op(Some("%sb"), OpKind::TensorSplat, &["%cb"]);
        let x = o.op(Some("%x"), OpKind::KtdpLoad, &["%t0"]);
        let ya = o.op(Some("%ya"), OpKind::ArithMulf, &["%x", "%sa"]);
        let st_a = o.op(None, OpKind::KtdpStore, &["%ya", "%t1"]);
        let y = o.op(Some("%y"), OpKind::KtdpLoad, &["%t2"]);
        let yb = o.op(Some("%yb"), OpKind::ArithMulf, &["%y", "%sb"]);
        let st_b = o.op(None, OpKind::KtdpStore, &["%yb", "%t3"]);
        let body = vec![c_a, s_a, c_b, s_b, x, ya, st_a, y, yb, st_b];
        let f = o.func("gelu_like", &[], body, (1, 1, 1));
        let mut module = IRModule {
            functions: [("gelu_like".to_string(), f)].into_iter().collect(),
            aliases: Default::default(),
        };
        assert_eq!(apply_splat_sink(o.arena(), &mut module), 1);
        let got: Vec<Option<Ssa>> = module.functions["gelu_like"]
            .operations
            .iter()
            .map(|op| op.result)
            .collect();
        let want: Vec<Option<Ssa>> = [
            Some("%ca"),
            Some("%cb"),
            Some("%x"),
            Some("%sa"),
            Some("%ya"),
            None,
            Some("%y"),
            Some("%sb"),
            Some("%yb"),
            None,
        ]
        .into_iter()
        .map(|n| n.map(|n| o.ssa(n)))
        .collect();
        assert_eq!(got, want, "each splat sits directly before its first reader");
    }

    /// A program with nothing to move is reported unchanged and left byte-identical.
    #[test]
    fn an_already_sunk_program_is_untouched() {
        let mut o = Ops::new();
        let x = o.op(Some("%x"), OpKind::KtdpLoad, &["%t0"]);
        let c = o.op(Some("%c"), OpKind::ArithConstant, &[]);
        let s = o.op(Some("%s"), OpKind::TensorSplat, &["%c"]);
        let y = o.op(Some("%y"), OpKind::ArithMulf, &["%x", "%s"]);
        let st = o.op(None, OpKind::KtdpStore, &["%y", "%t1"]);
        let f = o.func("sunk", &[], vec![x, c, s, y, st], (1, 1, 1));
        let before: Vec<Operation<'static>> = f.operations.to_vec();
        let mut module = IRModule {
            functions: [("sunk".to_string(), f)].into_iter().collect(),
            aliases: Default::default(),
        };
        assert_eq!(apply_splat_sink(o.arena(), &mut module), 0);
        assert_eq!(module.functions["sunk"].operations, before.as_slice());
    }
}
