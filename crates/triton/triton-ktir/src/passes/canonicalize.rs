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

//! `mlir::createCanonicalizerPass()`, ported to exactly what this pipeline needs.
//!
//! This is NOT a port of MLIR's whole canonicalizer -- it is a port of the effects
//! the pinned pipeline depends on, each one MEASURED against the golden rather than
//! assumed:
//!
//! 1. FOLD `index_cast` of an integer constant into an index constant. The golden's
//!    access tiles read `[%q, %c0]` where `%c0` is an INDEX constant; before
//!    canonicalize that operand was an `index_cast` of `%c0_i32`.
//! 2. DEDUPLICATE CONSTANTS, so the four descriptors' shared constants appear once.
//!    **Not** general CSE -- see [`dedup_constants`]; that distinction is measured.
//! 3. DCE pure ops whose results are unused -- the descriptor-typed
//!    `unrealized_conversion_cast` placeholders, and the i64 stride constants that
//!    only the erased `tt.make_tensor_descriptor` read.
//! 4. HOIST constants to the top of the function's entry block, which is what puts
//!    the multi-axis extent constant above the work loop in the golden.
//!
//! # THE TWO FIXES THIS PORT MUST KEEP
//!
//! DEAD i32 GRID-INDEX CHAINS ARE REWIRED **AND ERASED**. Dead IR is not free: the
//! scheduler still hoists `muli(divsi(index_cast(...)))` into a view start, where it
//! is refused. Rewiring alone leaves the chain in the IR and the refusal stands, so
//! [`dce`] runs to a FIXED POINT over transitively-dead pure ops rather than one
//! sweep.
//!
//! AN ARGUMENT REACHED ONLY THROUGH AN `affine.apply` OVER AN INDUCTION VARIABLE IS
//! **LIVE**. Liveness here is [`walk::used_values`], which walks INTO regions --
//! treating a value used only inside a loop body as dead drops every attention
//! bundle's key pointer. The test at the bottom of this file is that regression.

use std::collections::HashMap;

use crate::ir::*;
use crate::passes::walk::{self, OpPath};
use crate::Result;

pub fn run(module: &mut Module) -> Result<()> {
    // Iterate the whole set to a fixed point: a fold enables a CSE, a CSE enables a
    // DCE, and a DCE can make a constant hoistable.
    for _ in 0..16 {
        let mut changed = false;
        changed |= fold_index_casts(module);
        changed |= dedup_constants(module);
        changed |= dce(module);
        if !changed {
            break;
        }
    }
    hoist_constants(module);
    // Hoisting can expose a further duplicate (two equal constants from different
    // blocks now sit in the same one).
    while dedup_constants(module) | dce(module) {}
    Ok(())
}

//===----------------------------------------------------------------------===//
// 1. Folding
//===----------------------------------------------------------------------===//

/// `arith.index_cast` of an integer constant folds to an index constant.
///
/// MEASURED, not assumed: the golden's `construct_access_tile %view[%q, %c0]` has
/// `%c0 = arith.constant 0 : index`, where `ConvertTTIRToKTDP` emitted an
/// `index_cast` of the i32 zero. Without this fold every tile carries an extra cast
/// and the op census is wrong by one per index.
fn fold_index_casts(module: &mut Module) -> bool {
    let mut changed = false;
    loop {
        let target = walk::paths(module).into_iter().find_map(|p| {
            let op = walk::at(module, &p)?;
            if op.kind != OpKind::ArithIndexCast || !matches!(op.result_type(), Some(IrType::Index))
            {
                return None;
            }
            let src = op.operands.first().copied()?;
            let k = super::dot_to_linalg::const_int(module, src)?;
            Some((p, op.results[0], k))
        });
        let Some((path, result, k)) = target else { break };
        // Rewrite the cast in place as the index constant, keeping its result name so
        // every use is already wired.
        {
            let op = walk::at_mut(module, &path).expect("path");
            op.kind = OpKind::ArithConstant;
            op.operands.clear();
            op.set_attr(AttrKey::Value, Attr::Int(k));
            op.result_types = vec![IrType::Index];
        }
        let _ = result;
        changed = true;
    }
    changed
}

//===----------------------------------------------------------------------===//
// 2. Constant deduplication -- NOT CSE
//===----------------------------------------------------------------------===//

/// A CONSTANT's key: its value and its type. Nothing else is deduplicated.
///
/// WHY ONLY CONSTANTS, and this is MEASURED rather than a simplification. The
/// golden keeps TWO identical `arith.index_cast %qo_offset_y_24 : i32 to index` ops
/// -- one feeding the Q load's access tile, one feeding the O store's. A general CSE
/// merges them, and the op count is then wrong by one for a reason that looks like a
/// lowering bug and is not.
///
/// `mlir::createCanonicalizerPass()` materializes and deduplicates constants through
/// `OperationFolder`; it does NOT run CSE. CSE is `mlir::createCSEPass()`, a separate
/// pass, and `make_ktir`'s pipeline does not include it. Porting the canonicalizer as
/// "canonicalize plus CSE" would be porting a pipeline the C++ does not have.
fn cse_key(op: &Op) -> Option<String> {
    if op.kind != OpKind::ArithConstant || !op.regions.is_empty() || op.results.len() != 1 {
        return None;
    }
    let mut attrs: Vec<String> = op
        .attrs
        .iter()
        .map(|(k, v)| format!("{}={:?}", k.spelling(), v))
        .collect();
    attrs.sort();
    Some(format!("{}|{:?}|{}", op.kind.spelling(), op.result_types, attrs.join(",")))
}

/// Replace later duplicate constants with the first one.
///
/// SCOPED TO ONE BLOCK AT A TIME. A definition in a loop body does not dominate a
/// use outside it, so merging across that boundary would move a value out of scope.
fn dedup_constants(module: &mut Module) -> bool {
    let mut changed = false;
    let mut blocks: Vec<Vec<OpPath>> = Vec::new();
    group_by_block(module, &mut blocks);
    for block in blocks {
        let mut seen: HashMap<String, Ssa> = HashMap::new();
        let mut victims: Vec<OpPath> = Vec::new();
        let mut rewires: Vec<(Ssa, Ssa)> = Vec::new();
        for path in block {
            let Some(op) = walk::at(module, &path) else { continue };
            let Some(key) = cse_key(op) else { continue };
            let result = op.results[0];
            match seen.get(&key) {
                Some(first) => {
                    rewires.push((result, *first));
                    victims.push(path);
                }
                None => {
                    seen.insert(key, result);
                }
            }
        }
        for (from, to) in rewires {
            walk::replace_all_uses(module, from, to);
            changed = true;
        }
        walk::erase(module, &victims);
    }
    changed
}

/// Every block's op paths, grouped, so CSE and hoisting stay inside one block.
fn group_by_block(module: &Module, out: &mut Vec<Vec<OpPath>>) {
    let mut by_parent: HashMap<Vec<(usize, usize)>, Vec<OpPath>> = HashMap::new();
    for p in walk::paths(module) {
        let key = p.0[..p.0.len() - 1].to_vec();
        by_parent.entry(key).or_default().push(p);
    }
    let mut keys: Vec<Vec<(usize, usize)>> = by_parent.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let mut v = by_parent.remove(&k).unwrap();
        v.sort();
        out.push(v);
    }
}

//===----------------------------------------------------------------------===//
// 3. DCE
//===----------------------------------------------------------------------===//

/// Erase pure ops whose results are unused, to a FIXED POINT.
///
/// The fixed point is the point. A one-sweep DCE leaves the second link of a dead
/// chain behind, and a dead `muli(divsi(index_cast(...)))` chain is not free: the
/// scheduler hoists it into a view start where it is refused. So iterate until
/// nothing more is dead -- rewired AND erased.
fn dce(module: &mut Module) -> bool {
    let mut changed = false;
    loop {
        let used = walk::used_values(module);
        let victims: Vec<OpPath> = walk::paths(module)
            .into_iter()
            .filter(|p| {
                let op = walk::at(module, p).expect("path");
                op.kind.is_pure()
                    && op.regions.is_empty()
                    && !op.results.is_empty()
                    && op.results.iter().all(|r| !used.contains(r))
            })
            .collect();
        if victims.is_empty() {
            return changed;
        }
        walk::erase(module, &victims);
        changed = true;
    }
}

//===----------------------------------------------------------------------===//
// 4. Constant hoisting
//===----------------------------------------------------------------------===//

/// Move every `arith.constant` to the top of the enclosing function's entry block.
///
/// MLIR's `OperationFolder` keeps materialized constants in the entry block of the
/// nearest region that is ISOLATED FROM ABOVE -- for this IR that is the function.
/// The observable effect in the golden is the multi-axis extent constant sitting
/// ABOVE the work loop rather than inside it, so the loop body's first op is the
/// corelet plan.
///
/// ORDER: constants keep their relative program order, and land before every
/// non-constant. Their ABSOLUTE order among themselves is not semantic -- a
/// constant has no operands and no side effects -- which is why
/// [`crate::text::diff`] compares a block's leading constants as a MULTISET and
/// everything else in order. Pretending to match MLIR's internal insertion order
/// would be a claim this port cannot make from the outside.
fn hoist_constants(module: &mut Module) {
    let func_indices: Vec<usize> = (0..module.ops.len())
        .filter(|i| module.ops[*i].kind == OpKind::TtFunc)
        .collect();
    for fi in func_indices {
        let mut hoisted: Vec<Op> = Vec::new();
        collect_constants(&mut module.ops[fi].regions[0].ops, &mut hoisted);
        // Constants already at the function top stay where they are; the ones lifted
        // out of nested regions join them, in program order.
        let block = &mut module.ops[fi].regions[0].ops;
        for (k, c) in hoisted.into_iter().enumerate() {
            block.insert(k, c);
        }
        // Re-sort so every constant precedes every non-constant, order preserved.
        let (mut consts, rest): (Vec<Op>, Vec<Op>) = block
            .drain(..)
            .partition(|o| o.kind == OpKind::ArithConstant && o.regions.is_empty());
        consts.extend(rest);
        *block = consts;
    }
}

/// Lift constants out of nested regions, in program order.
fn collect_constants(ops: &mut Vec<Op>, out: &mut Vec<Op>) {
    for op in ops.iter_mut() {
        for r in op.regions.iter_mut() {
            let mut keep: Vec<Op> = Vec::new();
            for inner in r.ops.drain(..) {
                if inner.kind == OpKind::ArithConstant && inner.regions.is_empty() {
                    out.push(inner);
                } else {
                    keep.push(inner);
                }
            }
            r.ops = keep;
            collect_constants(&mut r.ops, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::parse;

    #[test]
    fn index_cast_of_a_constant_folds_to_an_index_constant() {
        // The tile needs a CONSUMER, and it is a `ktdp.load` here for the same reason
        // the real pipeline has one: `ktdp.load` is impure, so it anchors the chain
        // against DCE. Without it the whole chain is (correctly) dead.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %i = arith.index_cast %c0_i32 : i32 to index
    %t = ktdp.construct_access_tile %q[%i] {access_tile_order = affine_map<(d0) -> (d0)>} : memref<64xf16> -> !ktdp.access_tile<64xindex>
    %l = ktdp.load %t : <64xindex> -> tensor<64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("arith.index_cast"), 0, "the cast folded away");
        // The tile now reads an INDEX constant.
        let tile = m
            .ops_deep()
            .into_iter()
            .find(|o| o.kind == OpKind::KtdpConstructAccessTile)
            .unwrap();
        let idx = tile.operands[1];
        let def = m.def_of(idx).unwrap();
        assert_eq!(def.kind, OpKind::ArithConstant);
        assert_eq!(def.result_type(), Some(&IrType::Index));
        // And the now-dead i32 constant is GONE, not merely unused.
        assert_eq!(get("arith.constant"), 1, "the dead i32 constant is erased too");
    }

    #[test]
    fn identical_constants_are_deduplicated_to_one() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %a = arith.constant 128 : i32
    %b = arith.constant 128 : i32
    %c = arith.constant 64 : i32
    %x = arith.muli %a, %c : i32
    %y = arith.muli %b, %c : i32
    %z = arith.addi %x, %y : i32
    tt.return %z : i32
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(get("arith.constant"), 2, "128 appears once, 64 once");
        // AND THE TWO IDENTICAL MULTIPLIES SURVIVE. `canonicalize` is not CSE: the
        // golden keeps two identical index_casts, and a port that merges them is
        // wrong by one op. See `cse_key`.
        assert_eq!(get("arith.muli"), 2, "canonicalize does NOT CSE non-constants");
    }

    /// THE REGRESSION THE BRIEF NAMES. A value used ONLY inside a loop body is LIVE.
    /// Treating it as dead drops every attention bundle's key pointer.
    #[test]
    fn a_value_used_only_inside_a_loop_body_is_live() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %keyptr = builtin.unrealized_conversion_cast %q : !tt.ptr<f16> to index
    %view = ktdp.construct_memory_view %keyptr, sizes: [64], strides: [1] : memref<64xf16>
    scf.for %i = %c0 to %c8 step %c1 {
      %t = ktdp.construct_access_tile %view[%i] {access_tile_order = affine_map<(d0) -> (d0)>} : memref<64xf16> -> !ktdp.access_tile<64xindex>
      %l = ktdp.load %t : <64xindex> -> tensor<64xf16>
    }
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        let c = m.census();
        let get = |n: &str| c.iter().find(|(k, _)| k == n).map(|(_, v)| *v).unwrap_or(0);
        assert_eq!(
            get("ktdp.construct_memory_view"),
            1,
            "the view is used only INSIDE the loop and must survive"
        );
        assert_eq!(
            get("builtin.unrealized_conversion_cast"),
            1,
            "and so must the pointer it is built from -- this is the key-pointer drop"
        );
    }

    /// THE OTHER FIX THE BRIEF NAMES. A dead i32 grid-index chain is rewired AND
    /// ERASED, transitively -- one sweep would leave the middle of the chain behind.
    #[test]
    fn a_dead_grid_index_chain_is_erased_transitively_not_just_rewired() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c4 = arith.constant 4 : i32
    %c64 = arith.constant 64 : i32
    %p = arith.constant 3 : index
    %ic = arith.index_cast %p : index to i32
    %d = arith.divsi %ic, %c4 : i32
    %mm = arith.muli %d, %c64 : i32
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        let c = m.census();
        let total: usize = c.iter().map(|(_, v)| *v).sum();
        // Only tt.func + tt.return remain: the whole chain, constants included, is
        // gone. A single-sweep DCE would leave divsi/muli or their constants.
        assert_eq!(
            c,
            vec![("tt.func".to_string(), 1), ("tt.return".to_string(), 1)],
            "the whole dead chain is erased, not merely disconnected (census {total})"
        );
    }

    #[test]
    fn a_constant_defined_in_a_loop_body_is_hoisted_above_the_loop() {
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c8 step %c1 {
      %e = arith.constant 2 : index
      %r = arith.remui %i, %e : index
      %t = ktdp.construct_access_tile %q[%r] {access_tile_order = affine_map<(d0) -> (d0)>} : memref<64xf16> -> !ktdp.access_tile<64xindex>
      %l = ktdp.load %t : <64xindex> -> tensor<64xf16>
    }
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        let top: Vec<&str> = m.kernel().unwrap().regions[0]
            .ops
            .iter()
            .map(|o| o.kind.spelling())
            .collect();
        assert_eq!(
            top,
            vec![
                "arith.constant",
                "arith.constant",
                "arith.constant",
                "arith.constant", // the extent, lifted out of the body
                "scf.for",
                "tt.return"
            ]
        );
        let body = &m.kernel().unwrap().regions[0].ops[4].regions[0].ops;
        assert_eq!(
            body.iter().map(|o| o.kind.spelling()).collect::<Vec<_>>(),
            vec!["arith.remui", "ktdp.construct_access_tile", "ktdp.load"]
        );
    }

    #[test]
    fn a_ktdp_load_whose_result_is_unused_is_kept() {
        // ktdp.load declares no memory effects, so MLIR's driver will NOT DCE it. A
        // port that erases it here diverges from the C++ by one op, for a reason no
        // diff would explain.
        let src = "\
module {
  tt.func public @k(%q: !tt.ptr<f16>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %t = ktdp.construct_access_tile %q[%c0] {access_tile_order = affine_map<(d0) -> (d0)>} : memref<64xf16> -> !ktdp.access_tile<64xindex>
    %l = ktdp.load %t : <64xindex> -> tensor<64xf16>
    tt.return
  }
}
";
        let mut m = parse::parse(src).unwrap();
        run(&mut m).unwrap();
        let c = m.census();
        assert_eq!(c.iter().find(|(k, _)| k == "ktdp.load").map(|(_, v)| *v), Some(1));
    }
}
