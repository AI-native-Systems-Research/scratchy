// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! `module.walk(...)` and friends: the traversal primitives every pass uses.
//!
//! MLIR's `walk` visits an op and then its regions, and a pass that MUTATES while
//! walking corrupts the walker -- which is why nearly every C++ pass here says
//! "collect first, rewrite after". These helpers make that the only shape
//! available: [`walk`] and [`walk_mut`] give read/modify access to ops in place,
//! and structural edits go through [`OpPath`], collected first.

use crate::ir::*;
use std::collections::HashMap;

/// WHERE AN OP IS, as a path from the module root: `[i]` is `module.ops[i]`,
/// `[i, r, j]` is region `r` of that op, op `j`, and so on.
///
/// A path rather than an index, because a pass that erases an op invalidates
/// every FLAT index after it but only the paths that share its prefix -- and the
/// erase order (descending) then makes even those safe.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpPath(pub Vec<(usize, usize)>);

impl OpPath {
    /// `(region index, op index)` pairs. The first pair's region index is always
    /// 0 (the module's implicit region).
    pub fn root(idx: usize) -> OpPath {
        OpPath(vec![(0, idx)])
    }

    pub fn child(&self, region: usize, idx: usize) -> OpPath {
        let mut v = self.0.clone();
        v.push((region, idx));
        OpPath(v)
    }

    pub fn parent(&self) -> Option<OpPath> {
        if self.0.len() <= 1 {
            return None;
        }
        Some(OpPath(self.0[..self.0.len() - 1].to_vec()))
    }

    /// This op's index within its own block.
    pub fn index(&self) -> usize {
        self.0.last().expect("a path is never empty").1
    }
}

/// Every op in the module, in program order, with its path.
pub fn paths(module: &Module) -> Vec<OpPath> {
    fn count(ops: &[Op], out: &mut usize) {
        *out += ops.len();
        for o in ops {
            for r in &o.regions {
                count(&r.ops, out);
            }
        }
    }
    fn walk_region(ops: &[Op], parent: &OpPath, region: usize, out: &mut Vec<OpPath>) {
        for (i, op) in ops.iter().enumerate() {
            let p = parent.child(region, i);
            out.push(p.clone());
            for (r, inner) in op.regions.iter().enumerate() {
                walk_region(&inner.ops, &p, r, out);
            }
        }
    }
    // ONE exact allocation, not amortized growth: the fixed-point passes call this
    // once per rewrite (their `while find` shape), so every mid-walk realloc was
    // paid O(times) over the same module -- at kernel scale (~160k ops) the realloc
    // traffic alone dominated the pass (measured: `RawVec::grow_one` + `realloc`
    // ~60% of a sample).
    let mut n = 0usize;
    count(&module.ops, &mut n);
    let mut out = Vec::with_capacity(n);
    for (i, op) in module.ops.iter().enumerate() {
        let p = OpPath::root(i);
        out.push(p.clone());
        for (r, region) in op.regions.iter().enumerate() {
            walk_region(&region.ops, &p, r, &mut out);
        }
    }
    out
}

/// The op at `path`.
pub fn at<'m>(module: &'m Module, path: &OpPath) -> Option<&'m Op> {
    let (_, first) = path.0[0];
    let mut op = module.ops.get(first)?;
    for (r, i) in &path.0[1..] {
        op = op.regions.get(*r)?.ops.get(*i)?;
    }
    Some(op)
}

/// The op at `path`, mutably.
pub fn at_mut<'m>(module: &'m mut Module, path: &OpPath) -> Option<&'m mut Op> {
    let (_, first) = path.0[0];
    let mut op = module.ops.get_mut(first)?;
    for (r, i) in &path.0[1..] {
        op = op.regions.get_mut(*r)?.ops.get_mut(*i)?;
    }
    Some(op)
}

/// The block `path`'s op lives in.
pub fn block_mut<'m>(module: &'m mut Module, path: &OpPath) -> Option<&'m mut Vec<Op>> {
    if path.0.len() == 1 {
        return Some(&mut module.ops);
    }
    let (_, first) = path.0[0];
    let mut op = module.ops.get_mut(first)?;
    for (r, i) in &path.0[1..path.0.len() - 1] {
        op = op.regions.get_mut(*r)?.ops.get_mut(*i)?;
    }
    let (r, _) = path.0[path.0.len() - 1];
    Some(&mut op.regions.get_mut(r)?.ops)
}

/// The block `path`'s op lives in, shared -- the read-only twin of [`block_mut`].
pub fn block_ref<'m>(module: &'m Module, path: &OpPath) -> Option<&'m Vec<Op>> {
    if path.0.len() == 1 {
        return Some(&module.ops);
    }
    let (_, first) = path.0[0];
    let mut op = module.ops.get(first)?;
    for (r, i) in &path.0[1..path.0.len() - 1] {
        op = op.regions.get(*r)?.ops.get(*i)?;
    }
    let (r, _) = path.0[path.0.len() - 1];
    Some(&op.regions.get(r)?.ops)
}

/// Replace MANY ops in place, each with a run of zero or more ops, in ONE
/// rebuilding pass per block.
///
/// A batched rewrite that splices per op (`Vec::remove` + `Vec::insert`) moves
/// the whole tail of the block per rewrite -- quadratic at kernel scale, where a
/// single function body holds six figures of ops (measured: `memmove` was the
/// whole remaining cost of the descriptor walks). This rebuilds each touched
/// block once, by draining it and re-emitting ops in index order with the
/// replacements expanded.
///
/// `edits` are `(the path of an op being replaced, its replacement run)`. Every
/// edit's path must point at a DISTINCT op; edits to the same block may arrive
/// in any order. The replacement for an op may be EMPTY (an erase).
/// A block's identity: the path prefix of the op holding it, plus its region index.
type BlockKey = (Vec<(usize, usize)>, usize);

pub fn splice_many(module: &mut Module, edits: Vec<(OpPath, Vec<Op>)>) {
    // Group by block: the parent prefix, PLUS the region index the final pair's
    // region names (the last pair is `(region, index)` and the region belongs to
    // the op the prefix points at -- dropping it would merge sibling regions).
    let mut by_block: HashMap<BlockKey, Vec<(usize, Vec<Op>)>> = HashMap::new();
    for (path, ops) in edits {
        let (prefix, last) = path.0.split_at(path.0.len() - 1);
        let (region, i) = last[0];
        by_block
            .entry((prefix.to_vec(), region))
            .or_default()
            .push((i, ops));
    }
    // A representative path per block (prefix + the region, index 0), so
    // `block_mut` can find it.
    let mut blocks: Vec<BlockKey> = by_block.keys().cloned().collect();
    // DEEPEST BLOCKS FIRST. Splicing a block only invalidates paths that point
    // INTO it -- but a splice at prefix P also changes P's block's op COUNT,
    // which shifts every SIBLING index after it, and deeper blocks' prefixes are
    // named by those indices. Processing deepest-first means every block whose
    // prefix passes through P's block is already done when P's splice shifts it:
    // the pre-order sort puts a parent before its descendants, so this reverse
    // visits descendants first (the same law `unroll_constant_trip_loops` and
    // `walk::erase` state for their own rewrites).
    blocks.sort();
    blocks.reverse();
    for (prefix, region) in blocks {
        let mut probe = prefix.clone();
        probe.push((region, 0));
        let Some(mut edits) = by_block.remove(&(prefix, region)) else {
            continue;
        };
        edits.sort_by_key(|(i, _)| *i);
        let block = match block_mut(module, &OpPath(probe)) {
            Some(b) => b,
            None => continue,
        };
        let old = std::mem::take(block);
        let mut it = edits.into_iter().peekable();
        block.reserve(old.len());
        for (i, op) in old.into_iter().enumerate() {
            if it.peek().is_some_and(|(ei, _)| *ei == i) {
                let (_, replacement) = it.next().expect("peeked");
                block.extend(replacement);
            } else {
                block.push(op);
            }
        }
        // Edits past the end (should not happen for collected paths) are dropped;
        // a collected path's index is always inside its block.
    }
}

/// Visit every op, mutably, in program order. Structure must not change.
pub fn for_each_mut(module: &mut Module, mut f: impl FnMut(&mut Op)) {
    fn go(ops: &mut [Op], f: &mut impl FnMut(&mut Op)) {
        for op in ops.iter_mut() {
            f(op);
            for r in op.regions.iter_mut() {
                go(&mut r.ops, f);
            }
        }
    }
    go(&mut module.ops, &mut f);
}

/// Visit every op AND every region's block-argument list, mutably.
pub fn for_each_op_and_args_mut(
    module: &mut Module,
    mut f: impl FnMut(&mut Op),
    mut g: impl FnMut(&mut BlockArgs),
) {
    fn go(ops: &mut [Op], f: &mut impl FnMut(&mut Op), g: &mut impl FnMut(&mut BlockArgs)) {
        for op in ops.iter_mut() {
            f(op);
            for r in op.regions.iter_mut() {
                g(&mut r.args);
                go(&mut r.ops, f, g);
            }
        }
    }
    go(&mut module.ops, &mut f, &mut g);
}

/// Replace every USE of `from` with `to`, everywhere -- operands, and the
/// `iter_args` inits that are operands too.
///
/// `replaceAllUsesWith`. Results are NOT rewritten: an op still defines what it
/// defines, and rewriting a definition here is how a pass loses track of an op it
/// meant to erase.
pub fn replace_all_uses(module: &mut Module, from: Ssa, to: Ssa) {
    for_each_mut(module, |op| {
        for o in op.operands.iter_mut() {
            if *o == from {
                *o = to;
            }
        }
    });
}

/// Replace MANY values at once, in ONE walk -- the batched [`replace_all_uses`].
///
/// A pass that rewires N values by calling [`replace_all_uses`] N times walks the
/// whole module N times -- quadratic at kernel scale, where one function body
/// holds six figures of ops and a CSE pass rewires four figures of constants
/// (measured: `dedup_constants`' per-rewire walks dominated a multi-minute
/// expansion). This walks once, resolving every operand through the map.
///
/// Chains resolve transitively and in the order given: `(a→b, b→c)` rewrites an
/// `a` use to `c` when the entries are processed as a map lookup per operand --
/// they are NOT. Each operand gets ONE lookup, so a chain must be flattened by
/// the CALLER (follow `to` through the map until it is absent) if that is the
/// intended meaning. Callers today always pass disjoint pairs.
pub fn replace_all_uses_many(module: &mut Module, map: &HashMap<Ssa, Ssa>) {
    if map.is_empty() {
        return;
    }
    for_each_mut(module, |op| {
        for o in op.operands.iter_mut() {
            if let Some(to) = map.get(o) {
                *o = *to;
            }
        }
    });
}

/// Resolve every `to` through the map until it is absent -- the chain-flatten
/// that [`replace_all_uses_many`]'s contract requires of callers whose pairs
/// may chain (`a→b, b→c` must send an `a` use all the way to `c`). A cycle, if
/// a caller ever builds one, terminates at a value it has already visited only
/// by the caller's own map being cyclic -- the same loop would not have
/// terminated under per-victim `replace_all_uses` either.
pub fn flatten_rewire_map(map: &HashMap<Ssa, Ssa>) -> HashMap<Ssa, Ssa> {
    map.iter()
        .map(|(from, to)| {
            let mut end = *to;
            while let Some(next) = map.get(&end) {
                end = *next;
            }
            (*from, end)
        })
        .collect()
}

/// Every value USED anywhere in the module.
///
/// A value used only inside a nested region still counts, which is the whole
/// point: an argument reached only through an `affine.apply` over an induction
/// variable is LIVE, and a liveness walk that stops at the loop boundary drops
/// every attention bundle's key pointer.
pub fn used_values(module: &Module) -> std::collections::HashSet<Ssa> {
    let mut used = std::collections::HashSet::new();
    for op in module.ops_deep() {
        used.extend(op.operands.iter().copied());
    }
    used
}

/// Erase the ops at `victims`, deepest-and-last first so no surviving path shifts.
pub fn erase(module: &mut Module, victims: &[OpPath]) {
    let mut v: Vec<OpPath> = victims.to_vec();
    v.sort();
    v.dedup();
    for path in v.into_iter().rev() {
        if let Some(block) = block_mut(module, &path) {
            let i = path.index();
            if i < block.len() {
                block.remove(i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nested() -> Module {
        let mut m = Module::new();
        let a = m.fresh();
        let b = m.fresh();
        let inner = Op::new(OpKind::ArithAddf).with_result(b, IrType::Scalar(DType::F16));
        let forr = Op::new(OpKind::ScfFor).with_region(Region {
            args: vec![],
            ops: vec![inner],
        });
        let func = Op::new(OpKind::TtFunc).with_region(Region {
            args: vec![],
            ops: vec![
                Op::new(OpKind::ArithConstant).with_result(a, IrType::Scalar(DType::F16)),
                forr,
                Op::new(OpKind::TtReturn),
            ],
        });
        m.ops.push(func);
        m
    }

    #[test]
    fn paths_reach_into_nested_regions_and_resolve_back() {
        let m = nested();
        let ps = paths(&m);
        let kinds: Vec<&str> = ps
            .iter()
            .map(|p| at(&m, p).unwrap().kind.spelling())
            .collect();
        assert_eq!(
            kinds,
            vec![
                "tt.func",
                "arith.constant",
                "scf.for",
                "arith.addf",
                "tt.return"
            ],
            "program order, regions included"
        );
    }

    #[test]
    fn erase_of_several_paths_removes_exactly_those() {
        let mut m = nested();
        let ps = paths(&m);
        // Erase the constant (index 1) and the addf inside the loop (index 3).
        let victims = vec![ps[1].clone(), ps[3].clone()];
        erase(&mut m, &victims);
        let kinds: Vec<&str> = paths(&m)
            .iter()
            .map(|p| at(&m, p).unwrap().kind.spelling())
            .collect();
        assert_eq!(kinds, vec!["tt.func", "scf.for", "tt.return"]);
    }

    #[test]
    fn uses_inside_a_loop_body_count_as_uses() {
        let mut m = Module::new();
        let outer = m.fresh();
        let inner = Op::new(OpKind::ArithAddf).with_operands([outer]);
        let forr = Op::new(OpKind::ScfFor).with_region(Region {
            args: vec![],
            ops: vec![inner],
        });
        m.ops.push(Op::new(OpKind::TtFunc).with_region(Region {
            args: vec![],
            ops: vec![
                Op::new(OpKind::ArithConstant).with_result(outer, IrType::Scalar(DType::F16)),
                forr,
            ],
        }));
        assert!(
            used_values(&m).contains(&outer),
            "a value used only inside a nested region is LIVE"
        );
    }
}
