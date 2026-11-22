// SPDX-License-Identifier: Apache-2.0
//! The `dl16tofp32` lane stagger, carried through a lowered program.
//!
//! The card's widen writes value-exact but lane-permuted output (the measured stagger,
//! [`ElementArrangement::Dl16ToFp32`], its permutation `DeviceGen::DL16TOFP32_PERMUTATION`). A
//! pointwise op reads its operands lane-for-lane, so every buffer it reads beside a widened value
//! must hold the same permutation: a host-staged table there is staged already permuted
//! ([`stagger`]), and a computed value in the other order is a program the card cannot run
//! correctly — refused by name. Which op permutes is the device-facts table's row
//! (`ktir_core::ddl_binds::output_arrangement`), never logic here.
//!
//! [`staged_arrangements`] answers, for a lowered graph, the arrangement each parameter must be
//! staged in: every buffer is a class of one arrangement, a pointwise op joins its lane-varying
//! operands and its result into one class, the widen's input is standard and its result the
//! table's, a reduction across lanes is order-free and answers standard, and every buffer the
//! device reads or writes as another program's input or output is standard. A class holding both
//! is a refusal; a class of host-staged tables nothing fixes is standard.

use ktir_core::ddl_binds::{DeviceGen, DeviceOp, output_arrangement};
pub use ktir_core::ddl_binds::ElementArrangement;

use crate::graph::{Graph, Operand, Store};
use crate::opfunc::{Axis, Family, OpFunc};

/// Why a program's buffers cannot hold one arrangement each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrangementError {
    /// Node `node` reads or writes, lane-for-lane, a buffer that must be standard beside one that
    /// must be permuted.
    Mixed { node: usize },
}

/// `words` — a logical-order f32 word stream — in the [`ElementArrangement::Dl16ToFp32`] buffer
/// order: lane `i` of each 64-word block written at the permutation's position `i`. The one
/// staging of a permuted table; a tail short of a whole block stays in place.
pub fn stagger<G: DeviceGen>(words: &[f32]) -> Vec<f32> {
    let mut out = words.to_vec();
    let (blocks, _) = out.as_chunks_mut::<64>();
    for block in blocks {
        let logical = *block;
        for (i, word) in logical.iter().enumerate() {
            block[G::DL16TOFP32_PERMUTATION[i] as usize] = *word;
        }
    }
    out
}

/// Union-find over buffers (the parameters, then the intermediates), each class carrying the arrangement fixed on it, if any.
struct Classes {
    parent: Vec<usize>,
    fixed: Vec<Option<ElementArrangement>>,
}

impl Classes {
    fn find(&mut self, i: usize) -> usize {
        let p = self.parent[i];
        if p == i {
            return i;
        }
        let r = self.find(p);
        self.parent[i] = r;
        r
    }

    fn fix(&mut self, i: usize, a: ElementArrangement) -> bool {
        let r = self.find(i);
        match self.fixed[r] {
            Some(b) => a == b,
            None => {
                self.fixed[r] = Some(a);
                true
            }
        }
    }

    fn join(&mut self, i: usize, j: usize) -> bool {
        let (a, b) = (self.find(i), self.find(j));
        if a == b {
            return true;
        }
        let merged = match (self.fixed[a], self.fixed[b]) {
            (Some(x), Some(y)) if x != y => return false,
            (x, y) => x.or(y),
        };
        self.parent[a] = b;
        self.fixed[b] = merged;
        true
    }
}

/// Does `o` hold one value across a stick's lanes — a constant, or a view with no contiguous
/// run (one value per stick, repeated) — so its order within a block is no order at all?
fn lane_uniform(o: &Operand) -> bool {
    matches!(o.store, Store::Const(_))
        || (o.view.tile.is_none() && !o.view.axes.iter().any(|a| a.is_unit()))
}

/// The arrangement each parameter of `g` must be staged in. `staged[p]` says parameter `p` is a
/// host-staged table, free to be staged in either order; every other parameter is another
/// program's buffer and standard.
pub fn staged_arrangements<G: DeviceGen>(
    g: &Graph,
    staged: &[bool],
) -> Result<Vec<ElementArrangement>, ArrangementError> {
    let params = g.params.len();
    let n = params + g.locals.len();
    let mut c = Classes {
        parent: (0..n).collect(),
        fixed: vec![None; n],
    };
    let index = |o: &Operand| match o.store {
        Store::Param(p) => Some(p),
        Store::Local(l) => Some(params + l),
        Store::Const(_) => None,
    };
    for p in 0..params {
        if !staged.get(p).copied().unwrap_or(false) {
            c.fix(p, ElementArrangement::Standard);
        }
    }
    let row = |op: DeviceOp| output_arrangement::<G>(op);
    for (ni, node) in g.nodes.iter().enumerate() {
        let mixed = ArrangementError::Mixed { node: ni };
        let out = index(&node.output);
        let ins: Vec<Option<usize>> = node
            .inputs
            .iter()
            .map(|o| (!lane_uniform(o)).then(|| index(o)).flatten())
            .collect();
        let gathered = node.inputs.iter().any(|o| o.gather.is_some()) || node.output.gather.is_some();
        let ok = match (node.func, node.func.family()) {
            // The widen permutes a standard input by its row.
            (OpFunc::Widen, _) => ins.iter().flatten().all(|&i| c.fix(i, ElementArrangement::Standard))
                && out.is_none_or(|o| c.fix(o, row(DeviceOp::Dl16ToFp32))),
            // The narrow writes its row's arrangement whatever it reads.
            (OpFunc::Narrow, _) => out.is_none_or(|o| c.fix(o, row(DeviceOp::Fp32ToDl16))),
            // A copy through an index, a restick and a contraction move lanes between sticks: they
            // read and write standard buffers.
            (OpFunc::Restick, _) | (_, Family::Contraction) => ins
                .iter()
                .flatten()
                .chain(out.iter())
                .all(|&i| c.fix(i, ElementArrangement::Standard)),
            _ if gathered => ins
                .iter()
                .flatten()
                .chain(out.iter())
                .all(|&i| c.fix(i, ElementArrangement::Standard)),
            // A reduction across the lanes reads the block's multiset: its result is standard.
            (OpFunc::Reduce(_, Axis::Stick), _) => {
                out.is_none_or(|o| c.fix(o, ElementArrangement::Standard))
            }
            // Every other op reads and writes lane-for-lane: one class.
            _ => {
                let mut members = ins.iter().flatten().copied().chain(out);
                match members.next() {
                    Some(first) => members.all(|m| c.join(first, m)),
                    None => true,
                }
            }
        };
        if !ok {
            return Err(mixed);
        }
    }
    Ok((0..params)
        .map(|p| {
            let r = c.find(p);
            c.fixed[r].unwrap_or(ElementArrangement::Standard)
        })
        .collect())
}
